"""Path retrieval + perturbation sensitivity (proposal section 12).

Paths (only edges present in the training graph; target indication edges never used):
  Drug -> Gene/Protein -> Disease
  Drug -> Gene/Protein -> Pathway -> Gene/Protein -> Disease

Sensitivity:  Delta_P = f(d, s | G) - f(d, s | G \\ E_P), where E_P contains the path's
edges and their reverse copies.  Each Delta_P is compared with Delta for `n_random`
matched random removals: same number of edges, same edge types, endpoints in the
same log2-degree bucket where possible.
These deltas show what the model relies on; they are not biological evidence.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .data.graph import GraphBundle

def _adj(ei: torch.Tensor):
    a = defaultdict(set)
    for s, d in ei.t().tolist():
        a[s].add(d)
    return a


class PathExplainer:
    def __init__(self, model, graph: GraphBundle, device, max_paths: int = 5, n_random: int = 20, seed: int = 0):
        self.model, self.g, self.device = model, graph, device
        self.max_paths, self.n_random = max_paths, n_random
        self.rng = np.random.default_rng(seed)
        ei = graph.edge_index
        self.drug_prot = self._adj_between("drug_protein", "drug", "gene_protein")
        self.dis_prot = self._adj_between("disease_protein", "disease", "gene_protein")
        self.prot_path = self._adj_between("pathway_protein", "gene_protein", "pathway")
        self.path_prot = self._adj_between("pathway_protein", "pathway", "gene_protein")
        self.deg = {t: torch.zeros(n, dtype=torch.long) for t, n in graph.num_nodes.items()}
        for (st, _, dt), e in ei.items():
            self.deg[dt] += torch.bincount(e[1].cpu(), minlength=graph.num_nodes[dt])

    def _adj_between(self, rel, a, b):
        """Adjacency a -> b for relation `rel`, whichever direction it is stored in."""
        for (st, r, dt), e in self.g.edge_index.items():
            if r == rel and st == a and dt == b:
                return _adj(e.cpu())
            if r == rel and st == b and dt == a:
                return _adj(e.cpu().flip(0))
        return {}

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score(self, d, s, ei=None, return_alpha=False):
        self.model.eval()
        h, ctx = self.model.encode(self.g.x, ei if ei is not None else self.g.edge_index)
        dd = torch.tensor([d], device=self.device)
        ss = torch.tensor([s], device=self.device)
        logit, alpha = self.model.score(h, ctx, dd, ss, return_alpha=True)
        p = torch.sigmoid(logit).item()
        return (p, alpha) if return_alpha else p

    # ------------------------------------------------------------------ paths
    def paths(self, d, s):
        out = []
        targets = self.dis_prot.get(s, set())
        for g in sorted(self.drug_prot.get(d, ())):
            if g in targets:
                out.append([("drug", d), ("gene_protein", g), ("disease", s)])
        for g1 in sorted(self.drug_prot.get(d, ())):
            for pw in sorted(self.prot_path.get(g1, ())):
                for g2 in sorted(self.path_prot.get(pw, ()) & targets):
                    if g2 != g1:
                        out.append([("drug", d), ("gene_protein", g1), ("pathway", pw),
                                    ("gene_protein", g2), ("disease", s)])
            if len(out) > 200:
                break
        return out

    def _edges_of_path(self, p):
        """Map a node path to {edge_type: [(src, dst), ...]} including reverse copies."""
        rem = defaultdict(list)
        for (ta, a), (tb, b) in zip(p[:-1], p[1:]):
            for et in self.g.edge_index:
                if et[0] == ta and et[2] == tb:
                    rem[et].append((a, b))
                if et[0] == tb and et[2] == ta:
                    rem[et].append((b, a))
        return rem

    def _random_matched(self, rem):
        """Random edges of the same types, endpoint degrees in the same log2 bucket."""
        out = defaultdict(list)
        for et, pairs in rem.items():
            ei = self.g.edge_index[et].cpu()
            if ei.size(1) == 0:
                continue
            bs = torch.log2(self.deg[et[0]][ei[0]].float() + 1).floor()
            bd = torch.log2(self.deg[et[2]][ei[1]].float() + 1).floor()
            for (a, b) in pairs:
                tb_s = np.floor(np.log2(self.deg[et[0]][a].item() + 1))
                tb_d = np.floor(np.log2(self.deg[et[2]][b].item() + 1))
                cand = torch.nonzero((bs == tb_s) & (bd == tb_d)).flatten()
                if len(cand) == 0:
                    cand = torch.arange(ei.size(1))
                j = int(cand[self.rng.integers(0, len(cand))])
                out[et].append((int(ei[0, j]), int(ei[1, j])))
        return out

    def explain(self, d, s):
        base, alpha = self.score(d, s, return_alpha=True)
        res = {"drug": int(d), "disease": int(s), "score": base,
               "drug_name": str(self.g.node_names["drug"][d]), "disease_name": str(self.g.node_names["disease"][s]),
               "drug_degree": int(self.deg["drug"][d]), "disease_degree": int(self.deg["disease"][s])}
        if alpha is not None:
            res["gate_weights"] = {
                v: {"|".join(et): float(w) for et, w in zip(self.model.summ.rels[v], alpha[v][0].tolist())}
                for v in alpha}
        paths = self.paths(d, s)
        res["n_paths_found"] = len(paths)
        rows = []
        for p in paths[: self.max_paths]:
            rem = self._edges_of_path(p)
            delta = base - self.score(d, s, self.g.without_edges(rem))
            rnd = []
            for _ in range(self.n_random):
                rr = self._random_matched(rem)
                rnd.append(base - self.score(d, s, self.g.without_edges(rr)))
            rnd = np.asarray(rnd)
            rows.append({
                "path": " -> ".join(f"{t}:{self.g.node_names[t][i]}" for t, i in p),
                "n_edges_removed": sum(len(v) for v in rem.values()),
                "delta": float(delta),
                "random_delta_mean": float(rnd.mean()), "random_delta_std": float(rnd.std()),
                "empirical_p": float((np.sum(rnd >= delta) + 1) / (len(rnd) + 1)),
            })
        res["paths"] = rows
        return res
