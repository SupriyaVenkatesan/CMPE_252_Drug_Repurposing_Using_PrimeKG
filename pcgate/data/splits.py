"""Leakage-controlled split generation and fixed evaluation pools (proposal 10.1-10.5, 11.1).

Settings
  random   unique indication pairs split ~70/15/15
  drug     drugs that have indications are split into groups; all of a held-out drug's
           indications go to its partition (non-treatment context edges are kept)
  disease  same for diseases

Splits are produced on unique (drug, disease) pairs, i.e. BEFORE any inverse edges
are created; graph construction later adds reverse edges only for training pairs.

Evaluation pools (fixed and saved, identical for every model):
  * ranking queries: every disease with >=1 positive in the partition.
      random / disease : candidates = all eligible drugs, minus that disease's known
                         indications from the other partitions
      drug             : candidates = the drugs of the held-out drug group
  * AP / AUROC: every positive + `unlabeled_ratio` unlabeled pairs per positive drawn
    from the same candidate pool and never a known indication in any partition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils import read_json, sha256_array, write_json

SETTINGS = ("random", "drug", "disease")


@dataclass
class EvalPool:
    queries: np.ndarray        # disease ids, one per ranking query
    pos_ptr: np.ndarray        # CSR into pos_drugs
    pos_drugs: np.ndarray
    excl_ptr: np.ndarray       # CSR into excl_drugs (known non-partition indications)
    excl_drugs: np.ndarray
    cand_drugs: np.ndarray     # candidate drug universe for this partition
    ap_pairs: np.ndarray       # [N, 2] (drug, disease)
    ap_labels: np.ndarray      # [N] 1 = held-out indication, 0 = unlabeled

    def arrays(self, prefix: str) -> dict:
        return {f"{prefix}_{k}": v for k, v in self.__dict__.items()}

    @classmethod
    def from_npz(cls, z, prefix: str) -> "EvalPool":
        return cls(**{k: z[f"{prefix}_{k}"] for k in cls.__dataclass_fields__})

    def digest(self) -> str:
        return sha256_array(*[getattr(self, k) for k in self.__dataclass_fields__])


@dataclass
class Split:
    setting: str
    seed: int
    train: np.ndarray  # [N, 2] (drug, disease)
    val: np.ndarray
    test: np.ndarray
    heldout_drugs: np.ndarray     # drugs whose labels are hidden from training (drug setting)
    heldout_diseases: np.ndarray  # diseases whose labels are hidden from training (disease setting)
    val_pool: EvalPool
    test_pool: EvalPool
    all_known: np.ndarray  # every known indication [N, 2] (used to filter negatives)

    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "split.npz", train=self.train, val=self.val, test=self.test,
            heldout_drugs=self.heldout_drugs, heldout_diseases=self.heldout_diseases,
            all_known=self.all_known, **self.val_pool.arrays("val"), **self.test_pool.arrays("test"))
        write_json({
            "setting": self.setting, "seed": self.seed,
            "n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test),
            "n_heldout_drugs": len(self.heldout_drugs), "n_heldout_diseases": len(self.heldout_diseases),
            "val_queries": len(self.val_pool.queries), "test_queries": len(self.test_pool.queries),
            "val_ap_pairs": len(self.val_pool.ap_pairs), "test_ap_pairs": len(self.test_pool.ap_pairs),
            "test_candidate_universe": len(self.test_pool.cand_drugs),
            "pair_sha256": sha256_array(self.train, self.val, self.test),
            "val_pool_sha256": self.val_pool.digest(),
            "test_pool_sha256": self.test_pool.digest(),
        }, d / "manifest.json")

    @classmethod
    def load(cls, d: str | Path) -> "Split":
        d = Path(d)
        man = read_json(d / "manifest.json")
        z = np.load(d / "split.npz")
        s = cls(setting=man["setting"], seed=man["seed"], train=z["train"], val=z["val"], test=z["test"],
                heldout_drugs=z["heldout_drugs"], heldout_diseases=z["heldout_diseases"],
                val_pool=EvalPool.from_npz(z, "val"), test_pool=EvalPool.from_npz(z, "test"),
                all_known=z["all_known"])
        # Fixed evaluation pools: refuse to run if the pools changed on disk.
        assert s.val_pool.digest() == man["val_pool_sha256"], "val pool hash mismatch"
        assert s.test_pool.digest() == man["test_pool_sha256"], "test pool hash mismatch"
        return s


def _group_split(ids: np.ndarray, frac, rng):
    ids = rng.permutation(np.unique(ids))
    n = len(ids)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]


def make_split(pairs: np.ndarray, n_drugs: int, setting: str, seed: int,
               frac=(0.7, 0.15, 0.15), unlabeled_ratio: int = 10,
               eligible_drugs: np.ndarray | None = None) -> Split:
    assert setting in SETTINGS, setting
    rng = np.random.default_rng(seed)
    pairs = np.unique(pairs, axis=0)
    empty = np.zeros(0, dtype=np.int64)
    held_drugs, held_dis = empty, empty
    if setting == "random":
        perm = rng.permutation(len(pairs))
        n_tr = int(round(frac[0] * len(pairs)))
        n_va = int(round(frac[1] * len(pairs)))
        tr, va, te = pairs[perm[:n_tr]], pairs[perm[n_tr:n_tr + n_va]], pairs[perm[n_tr + n_va:]]
        va_drugs = te_drugs = None
    else:
        col = 0 if setting == "drug" else 1
        g_tr, g_va, g_te = _group_split(pairs[:, col], frac, rng)
        tr = pairs[np.isin(pairs[:, col], g_tr)]
        va = pairs[np.isin(pairs[:, col], g_va)]
        te = pairs[np.isin(pairs[:, col], g_te)]
        held = np.concatenate([g_va, g_te])
        if setting == "drug":
            held_drugs = np.sort(held)
            va_drugs, te_drugs = np.sort(g_va), np.sort(g_te)
        else:
            held_dis = np.sort(held)
            va_drugs = te_drugs = None

    all_drugs = np.arange(n_drugs) if eligible_drugs is None else np.asarray(eligible_drugs)
    known = {(int(a), int(b)) for a, b in pairs}
    val_pool = _make_pool(va, [tr, te], va_drugs if va_drugs is not None else all_drugs,
                          known, unlabeled_ratio, rng)
    test_pool = _make_pool(te, [tr, va], te_drugs if te_drugs is not None else all_drugs,
                           known, unlabeled_ratio, rng)
    return Split(setting, seed, tr, va, te, held_drugs, held_dis, val_pool, test_pool, pairs)


def _csr(groups: dict, keys: np.ndarray):
    ptr = [0]
    vals = []
    for k in keys:
        v = sorted(groups.get(int(k), ()))
        vals.extend(v)
        ptr.append(len(vals))
    return np.asarray(ptr, dtype=np.int64), np.asarray(vals, dtype=np.int64)


def _make_pool(part: np.ndarray, others: list, cand: np.ndarray, known: set,
               ratio: int, rng) -> EvalPool:
    pos: dict = {}
    for d, s in part:
        pos.setdefault(int(s), set()).add(int(d))
    excl: dict = {}
    cand_set = set(int(c) for c in cand)
    for o in others:
        for d, s in o:
            if int(s) in pos and int(d) in cand_set:
                excl.setdefault(int(s), set()).add(int(d))
    queries = np.asarray(sorted(pos), dtype=np.int64)
    pos_ptr, pos_drugs = _csr(pos, queries)
    excl_ptr, excl_drugs = _csr(excl, queries)

    ap_pairs, ap_labels = [], []
    cand = np.asarray(sorted(cand_set), dtype=np.int64)
    for s in queries:
        for d in sorted(pos[int(s)]):
            ap_pairs.append((d, int(s)))
            ap_labels.append(1)
            got, tries = 0, 0
            while got < ratio and tries < ratio * 50:
                tries += 1
                c = int(cand[rng.integers(0, len(cand))])
                if (c, int(s)) in known:
                    continue
                ap_pairs.append((c, int(s)))
                ap_labels.append(0)
                got += 1
    return EvalPool(queries, pos_ptr, pos_drugs, excl_ptr, excl_drugs, cand,
                    np.asarray(ap_pairs, dtype=np.int64).reshape(-1, 2),
                    np.asarray(ap_labels, dtype=np.int64))
