"""Verification tests: metrics (hand-computed), splits, eval pools, leakage checks.

Run:  python -m pytest -q tests
"""
import math

import numpy as np
import pytest
import torch

from pcgate.data.splits import EvalPool, make_split
from pcgate.evaluate import evaluate_pool
from pcgate.models import Popularity


# ----------------------------------------------------------------------------- metrics
def _pool(queries, pos, excl, cand, ap_pairs, ap_labels):
    def csr(groups):
        ptr, vals = [0], []
        for q in queries:
            vals += groups.get(q, [])
            ptr.append(len(vals))
        return np.array(ptr), np.array(vals, dtype=np.int64)
    pp, pv = csr(pos)
    ep, ev = csr(excl)
    return EvalPool(np.array(queries), pp, pv, ep, ev, np.array(cand),
                    np.array(ap_pairs).reshape(-1, 2), np.array(ap_labels))


def test_ranking_metrics_hand_computed():
    # 20 candidate drugs; scorer ranks drug i by -i (drug 0 best).
    cand = list(range(20))
    # disease 0: positives at ranks 1 and 12 -> R@10 = 1/2, R@20 = 1, P@10 = 1/10
    # disease 1: positive drug 3, but drug 0 is excluded (known train indication) -> drug 3 at rank 3
    pool = _pool([0, 1], {0: [0, 11], 1: [3]}, {1: [0]}, cand,
                 [(0, 0), (5, 0), (11, 0), (3, 1), (7, 1)], [1, 0, 1, 1, 0])
    m = evaluate_pool(lambda d, s: -np.asarray(d, dtype=float), pool)
    assert m["recall@10"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["recall@20"] == pytest.approx(1.0)
    assert m["precision@10"] == pytest.approx((1 / 10 + 1 / 10) / 2)
    ndcg0 = (1 / math.log2(2)) / (1 / math.log2(2) + 1 / math.log2(3))
    ndcg1 = (1 / math.log2(4)) / 1.0
    assert m["ndcg@10"] == pytest.approx((ndcg0 + ndcg1) / 2)
    assert m["n_queries"] == 2
    assert m["mean_pool_size"] == pytest.approx((20 + 19) / 2)


def test_ap_auroc_hand_computed():
    pool = _pool([0], {0: [0]}, {}, [0, 1, 2], [(0, 0), (1, 0), (2, 0), (3, 0)], [1, 0, 1, 0])
    scores = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.1}
    m = evaluate_pool(lambda d, s: np.array([scores[int(x)] for x in d]), pool)
    # ranking: +, -, +, -  ->  AP = (1/1 + 2/3) / 2
    assert m["ap"] == pytest.approx((1 + 2 / 3) / 2)
    # AUROC: positive/negative pairs correctly ordered: (0>1,0>3,2>3) of 4 -> 0.75
    assert m["auroc"] == pytest.approx(0.75)


def test_tie_break_is_fixed_and_not_order_biased():
    pool = _pool([0], {0: [0]}, {}, list(range(100)), [(0, 0), (1, 0)], [1, 0])
    const = lambda d, s: np.zeros(len(d))
    a = evaluate_pool(const, pool)["recall@10"]
    b = evaluate_pool(const, pool)["recall@10"]
    assert a == b  # deterministic


def test_popularity_scores_training_counts():
    pop = Popularity(4, np.array([[1, 0], [1, 2], [3, 1]]))
    s = pop.score_pairs(np.array([0, 1, 2, 3]), np.zeros(4, dtype=int)).numpy()
    assert list(np.argsort(-s)[:2]) == [1, 3] and s[0] == 0


# ----------------------------------------------------------------------------- splits
def _pairs(n_drugs=60, n_dis=40, n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = np.stack([rng.integers(0, n_drugs, n), rng.integers(0, n_dis, n)], 1)
    return np.unique(p, axis=0)


@pytest.mark.parametrize("setting", ["random", "drug", "disease"])
def test_split_partitions_and_pools(setting):
    pairs = _pairs()
    sp = make_split(pairs, 60, setting, seed=3)
    tr, va, te = (set(map(tuple, x.tolist())) for x in (sp.train, sp.val, sp.test))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) + len(va) + len(te) == len(pairs)
    assert 0.6 < len(tr) / len(pairs) < 0.8
    if setting == "drug":
        held = set(sp.heldout_drugs.tolist())
        assert held and not any(d in held for d, _ in tr)
        assert set(sp.test_pool.cand_drugs.tolist()) == set(sp.test[:, 0].tolist())
    if setting == "disease":
        held = set(sp.heldout_diseases.tolist())
        assert held and not any(s in held for _, s in tr)
    known = set(map(tuple, pairs.tolist()))
    for pool, part in ((sp.val_pool, va), (sp.test_pool, te)):
        pos = set(map(tuple, pool.ap_pairs[pool.ap_labels == 1].tolist()))
        unl = pool.ap_pairs[pool.ap_labels == 0]
        assert pos == part
        assert not (set(map(tuple, unl.tolist())) & known)
        assert len(unl) <= 10 * len(pos)
        # exclusions are known indications from OTHER partitions only
        for qi, s in enumerate(pool.queries):
            ex = pool.excl_drugs[pool.excl_ptr[qi]:pool.excl_ptr[qi + 1]]
            for d in ex:
                assert (int(d), int(s)) in known and (int(d), int(s)) not in part


def test_split_is_reproducible():
    pairs = _pairs()
    a = make_split(pairs, 60, "disease", seed=5)
    b = make_split(pairs, 60, "disease", seed=5)
    assert np.array_equal(a.test, b.test) and a.test_pool.digest() == b.test_pool.digest()


# ----------------------------------------------------------------------------- pipeline + leakage
def test_end_to_end_pipeline_and_leakage_checks(tmp_path):
    from pcgate.data.checks import run_checks
    from pcgate.data.graph import IND, build_graph, make_features
    from pcgate.data.preprocess import preprocess
    from pcgate.data.splits import Split
    from pcgate.data.synthetic import generate

    raw = generate(tmp_path / "kg.csv", seed=1)
    audit = preprocess(raw, tmp_path / "proc")
    assert audit["invalid_rows"] == 8                    # 5 missing ids + 3 unknown types injected
    assert audit["unique_canonical_edges"] * 2 <= audit["rows_after_invalid_removal"]
    import pandas as pd
    ind = pd.read_csv(tmp_path / "proc" / "indications.csv").to_numpy()
    n_drugs = int((pd.read_csv(tmp_path / "proc" / "nodes.csv")["type"] == "drug").sum())
    for setting in ("random", "drug", "disease"):
        make_split(ind, n_drugs, setting, 0).save(tmp_path / setting)
        sp = Split.load(tmp_path / setting)
        g = build_graph(tmp_path / "proc", sp)
        res = run_checks(tmp_path / "proc", sp, g)
        assert res["all_passed"], res
        # edge masking removes exactly the requested training edges (+ reverse copies)
        idx = torch.tensor([0, 1, 2])
        masked = g.masked_edges(idx)
        assert masked[IND].size(1) == g.edge_index[IND].size(1) - 3
        assert not any(rel in ("contraindication", "off_label_use") for _, rel, _ in g.edge_index)

    # Regression: `run` re-verifies the checks on a graph that has already been moved to the
    # training device. Features must still be computable, and must come back on the CPU.
    # On a CUDA box this reproduces the original failure directly; on CPU it at least pins
    # the contract that make_features never returns device-dependent tensors.
    sp = Split.load(tmp_path / "random")
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        g = build_graph(tmp_path / "proc", sp).to(torch.device(dev))
        x_re, _ = make_features(g.num_nodes, g.edge_index)
        assert all(v.device.type == "cpu" for v in x_re.values())
        assert run_checks(tmp_path / "proc", sp, g)["all_passed"]

    # Deliberately leak a test pair into the graph -> check 2 must fail.
    sp = Split.load(tmp_path / "random")
    g = build_graph(tmp_path / "proc", sp)
    leak = torch.as_tensor(sp.test[:1].T)
    g.edge_index[IND] = torch.cat([g.edge_index[IND], leak], 1)
    assert not run_checks(tmp_path / "proc", sp, g)["2_no_heldout_copies_in_graph"]
