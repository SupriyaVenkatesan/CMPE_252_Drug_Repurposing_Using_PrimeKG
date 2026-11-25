"""Metrics on the fixed evaluation pools (proposal section 11).

Main:      AP (positives vs the pre-sampled 1:10 unlabeled pool), Recall@10 (macro over diseases)
Secondary: AUROC, Precision@10, NDCG@10, Recall@20
All ranking metrics are macro-averaged over evaluable diseases (each disease weighted
equally); ties are broken with a fixed-seed random permutation so popularity-style
scorers with many ties are not favoured by candidate ordering.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .data.splits import EvalPool

Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (drug_idx, disease_idx) -> scores


def make_nn_scorer(model, graph, device, chunk: int = 1 << 16, edge_index=None) -> Scorer:
    model.eval()
    with torch.no_grad():
        h, ctx = model.encode(graph.x, edge_index if edge_index is not None else graph.edge_index)

    def scorer(d, s):
        out = []
        with torch.no_grad():
            for i in range(0, len(d), chunk):
                dd = torch.as_tensor(d[i:i + chunk], device=device)
                ss = torch.as_tensor(s[i:i + chunk], device=device)
                out.append(model.score(h, ctx, dd, ss).float().cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)
    return scorer


def make_pop_scorer(pop) -> Scorer:
    return lambda d, s: pop.score_pairs(d, s).numpy()


def evaluate_pool(scorer: Scorer, pool: EvalPool, ks=(10, 20), tie_seed: int = 0) -> dict:
    res: dict = {}
    # ---- AP / AUROC on fixed positive + unlabeled pairs
    sc = scorer(pool.ap_pairs[:, 0], pool.ap_pairs[:, 1])
    y = pool.ap_labels
    res["ap"] = float(average_precision_score(y, sc)) if y.sum() and (1 - y).sum() else float("nan")
    res["auroc"] = float(roc_auc_score(y, sc)) if y.sum() and (1 - y).sum() else float("nan")

    # ---- per-disease ranking
    rng = np.random.default_rng(tie_seed)
    cand = pool.cand_drugs
    tie = rng.random(len(cand))
    per = {f"recall@{k}": [] for k in ks}
    per["precision@10"], per["ndcg@10"], pool_sizes = [], [], []
    for qi, s in enumerate(pool.queries):
        pos = pool.pos_drugs[pool.pos_ptr[qi]:pool.pos_ptr[qi + 1]]
        excl = pool.excl_drugs[pool.excl_ptr[qi]:pool.excl_ptr[qi + 1]]
        keep = ~np.isin(cand, excl)
        c, t = cand[keep], tie[keep]
        scores = scorer(c, np.full(len(c), s, dtype=np.int64))
        order = np.lexsort((t, -scores))
        ranked = c[order]
        is_pos = np.isin(ranked, pos)
        pool_sizes.append(len(c))
        for k in ks:
            per[f"recall@{k}"].append(is_pos[:k].sum() / len(pos))
        per["precision@10"].append(is_pos[:10].sum() / 10)
        top = is_pos[:10]
        gains = top / np.log2(np.arange(2, 2 + len(top)))
        ideal = (1.0 / np.log2(np.arange(2, 2 + min(len(pos), 10)))).sum()
        per["ndcg@10"].append(gains.sum() / ideal)
    for k, v in per.items():
        res[k] = float(np.mean(v)) if v else float("nan")
    res["n_queries"] = int(len(pool.queries))
    res["n_positive_pairs"] = int(pool.ap_labels.sum())
    res["n_ap_pairs"] = int(len(pool.ap_labels))
    res["mean_pool_size"] = float(np.mean(pool_sizes)) if pool_sizes else 0.0
    res["candidate_universe"] = int(len(cand))
    res["per_query"] = {"disease": pool.queries.tolist(),
                        "recall@10": [float(v) for v in per["recall@10"]],
                        "ndcg@10": [float(v) for v in per["ndcg@10"]]}
    return res
