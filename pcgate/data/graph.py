"""Model-ready heterogeneous graph construction and structural node features.

Primary graph policy (proposal 6.3):
  * all permitted biological context relations, in both directions;
  * indication edges ONLY for the training partition (+ their reverse copies);
  * other drug-disease relations (contraindication, off-label use) are excluded by
    default so they cannot act as treatment-label proxies; if a config re-enables
    them, any copy of a validation/test indication pair is still dropped.

Node features (6.4): node-type one-hot, log1p per-relation incoming counts,
missing-neighbourhood indicators, log1p total degree.  No identity embeddings, so
held-out drugs/diseases are represented by the same shared transformations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..utils import sanitize_relation
from .splits import Split

IND = ("drug", "indication", "disease")
REV_IND = ("disease", "rev_indication", "drug")


@dataclass
class GraphBundle:
    num_nodes: dict
    edge_index: dict                    # edge_type -> LongTensor[2, E]
    x: dict                             # node_type -> FloatTensor[N, F]
    feature_names: list
    node_names: dict = field(default_factory=dict)   # type -> np.ndarray of names
    node_ids: dict = field(default_factory=dict)     # type -> np.ndarray of source ids

    @property
    def node_types(self):
        return sorted(self.num_nodes)

    @property
    def edge_types(self):
        return sorted(self.edge_index)

    def metadata(self):
        return (self.node_types, self.edge_types)

    def to(self, device):
        self.edge_index = {k: v.to(device) for k, v in self.edge_index.items()}
        self.x = {k: v.to(device) for k, v in self.x.items()}
        return self

    def masked_edges(self, train_pair_idx: torch.Tensor | None) -> dict:
        """Edge dict with the given training indication pairs (and reverse copies) removed."""
        if train_pair_idx is None or IND not in self.edge_index:
            return self.edge_index
        E = self.edge_index[IND].size(1)
        keep = torch.ones(E, dtype=torch.bool, device=self.edge_index[IND].device)
        keep[train_pair_idx] = False
        out = dict(self.edge_index)
        out[IND] = self.edge_index[IND][:, keep]
        out[REV_IND] = self.edge_index[REV_IND][:, keep]
        return out

    def without_edges(self, removals: dict) -> dict:
        """Edge dict with specific (src, dst) edges removed per edge type (used by explanations)."""
        out = dict(self.edge_index)
        for et, pairs in removals.items():
            if not pairs or et not in out:
                continue
            ei = out[et]
            n_dst = max(int(self.num_nodes[et[2]]), 1)
            code = ei[0] * n_dst + ei[1]
            rm = torch.tensor([s * n_dst + d for s, d in pairs], device=ei.device)
            out[et] = ei[:, ~torch.isin(code, rm)]
        return out


def build_graph(processed_dir: str | Path, split: Split, cfg: dict | None = None) -> GraphBundle:
    cfg = cfg or {}
    processed_dir = Path(processed_dir)
    exclude = {sanitize_relation(r) for r in cfg.get("exclude_relations", ["contraindication", "off-label use"])}
    ind_rel = sanitize_relation(cfg.get("indication_relation", "indication"))
    use_ind_feat = cfg.get("indication_degree_feature", True)
    keep_only = cfg.get("include_relations")  # optional whitelist

    nodes = pd.read_csv(processed_dir / "nodes.csv", dtype={"id": str, "name": str})
    num_nodes = nodes.groupby("type").size().to_dict()
    node_names = {t: g.sort_values("type_index")["name"].to_numpy() for t, g in nodes.groupby("type")}
    node_ids = {t: g.sort_values("type_index")["id"].to_numpy() for t, g in nodes.groupby("type")}
    edges = pd.read_csv(processed_dir / "edges.csv")

    heldout = {(int(a), int(b)) for a, b in np.concatenate([split.val, split.test])}
    edge_index: dict = {}
    for (rel, st, dt), g in edges.groupby(["relation", "src_type", "dst_type"], sort=True):
        if rel == ind_rel or rel in exclude:
            continue
        if keep_only is not None and rel not in keep_only:
            continue
        s = g["src"].to_numpy(np.int64)
        d = g["dst"].to_numpy(np.int64)
        if {st, dt} == {"drug", "disease"}:
            # Never let a copy of a held-out indication into the graph.
            dd = (s, d) if st == "drug" else (d, s)
            ok = np.array([(int(a), int(b)) not in heldout for a, b in zip(*dd)], dtype=bool)
            s, d = s[ok], d[ok]
        if len(s) == 0:
            continue
        if st == dt:
            ei = np.stack([np.concatenate([s, d]), np.concatenate([d, s])])
            edge_index[(st, rel, dt)] = torch.from_numpy(np.unique(ei, axis=1))
        else:
            edge_index[(st, rel, dt)] = torch.from_numpy(np.stack([s, d]))
            edge_index[(dt, "rev_" + rel, st)] = torch.from_numpy(np.stack([d, s]))

    tr = torch.from_numpy(np.asarray(split.train, dtype=np.int64).reshape(-1, 2))
    edge_index[IND] = tr.t().contiguous()            # column i == split.train[i]
    edge_index[REV_IND] = tr.flip(1).t().contiguous()

    x, names = make_features(num_nodes, edge_index, use_ind_feat)
    return GraphBundle(num_nodes, edge_index, x, names, node_names, node_ids)


def make_features(num_nodes: dict, edge_index: dict, use_indication: bool = True):
    """Structural node features. Always built on the CPU, whatever device the edge index is on:
    `run` re-verifies the features of an already-moved (possibly CUDA) graph, and mixing devices
    here raised "Expected all tensors to be on the same device"."""
    types = sorted(num_nodes)
    ets = sorted(et for et in edge_index if use_indication or et not in (IND, REV_IND))
    names = [f"type={t}" for t in types] + [f"log_in[{'|'.join(et)}]" for et in ets] \
        + [f"missing[{'|'.join(et)}]" for et in ets] + ["log_degree"]
    F = len(names)
    x = {}
    for ti, t in enumerate(types):
        n = num_nodes[t]
        f = torch.zeros(n, F)
        f[:, ti] = 1.0
        total = torch.zeros(n)
        for j, et in enumerate(ets):
            if et[2] != t:
                continue
            cnt = torch.bincount(edge_index[et][1].cpu(), minlength=n).float()
            f[:, len(types) + j] = torch.log1p(cnt)
            f[:, len(types) + len(ets) + j] = (cnt == 0).float()
            total += cnt
        f[:, -1] = torch.log1p(total)
        x[t] = f
    return x, names
