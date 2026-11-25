"""Integrity / leakage checks that must pass before experiments (proposal 6.5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..utils import read_json, sha256_file
from .graph import IND, REV_IND, GraphBundle, make_features
from .splits import Split


def _pairs(a) -> set:
    return {(int(x), int(y)) for x, y in np.asarray(a).reshape(-1, 2)}


def run_checks(processed_dir: str | Path, split: Split, graph: GraphBundle, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    res: dict = {}
    tr, va, te = _pairs(split.train), _pairs(split.val), _pairs(split.test)
    held = va | te

    # 1. No target-pair overlap across partitions.
    res["1_no_pair_overlap"] = not (tr & va or tr & te or va & te)

    # 2. No (reverse) copies of held-out indications in ANY drug<->disease edge type.
    leaked = 0
    for et, ei in graph.edge_index.items():
        if {et[0], et[2]} != {"drug", "disease"}:
            continue
        e = ei.t().cpu().numpy()
        if et[0] == "disease":
            e = e[:, ::-1]
        leaked += len(_pairs(e) & held)
    res["2_no_heldout_copies_in_graph"] = leaked == 0
    res["2_leaked_edges"] = leaked
    ind_graph = _pairs(graph.edge_index[IND].t().cpu().numpy())
    rev_graph = _pairs(graph.edge_index[REV_IND].flip(0).t().cpu().numpy())
    res["2b_indication_edges_equal_train"] = ind_graph == tr and rev_graph == tr

    # 3. No held-out entity treatment labels in training.
    hd = set(split.heldout_drugs.tolist())
    hs = set(split.heldout_diseases.tolist())
    res["3_no_heldout_entity_labels_in_train"] = not any(d in hd or s in hs for d, s in tr)

    # 4. No treatment-bearing feature leakage: features must be reproducible from the
    #    training graph alone, and no feature may be derived from excluded relations.
    x_re, names = make_features(graph.num_nodes, graph.edge_index, cfg.get("indication_degree_feature", True))
    same = all(torch.allclose(x_re[t], graph.x[t].cpu()) for t in graph.x)
    banned = [n for n in names if "contraindication" in n or "off_label" in n]
    res["4_features_from_train_graph_only"] = bool(same) and not banned

    # 5. Consistent entity mappings across models (hash of nodes.csv recorded at preprocessing).
    audit = read_json(Path(processed_dir) / "audit.json")
    res["5_entity_mapping_hash_consistent"] = audit["nodes_sha256"] == sha256_file(Path(processed_dir) / "nodes.csv")

    # 6. Fixed evaluation pools: Split.load() verifies pool hashes; also check pool contents.
    ok = True
    for pool, part in ((split.val_pool, va), (split.test_pool, te)):
        pos = _pairs(pool.ap_pairs[pool.ap_labels == 1])
        unl = _pairs(pool.ap_pairs[pool.ap_labels == 0])
        ok &= pos == part
        ok &= not (unl & _pairs(split.all_known))
    res["6_fixed_eval_pools_valid"] = bool(ok)

    res["all_passed"] = all(v for k, v in res.items() if k[0].isdigit() and isinstance(v, bool))
    return res
