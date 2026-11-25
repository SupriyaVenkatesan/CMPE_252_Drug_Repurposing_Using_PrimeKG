"""Gate analysis and evaluation-time gate controls (Member 2; proposal RQ3).

    python -m pcgate gates --config configs/primekg_eligible_frozen.yaml            # disease setting
    python -m pcgate gates --config ... --setting random --seeds 0 1 2

Trained controls (separate models, run by `pcgate run`):
    uniform        same relation summaries, alpha = 1/|R_v|         -> is LEARNED weighting needed?
    param_matched  same summaries, flat projection, matched #params -> is it just extra capacity?

Evaluation-time controls on the TRAINED gated model (no retraining, same checkpoint):
    gated              the model as trained
    shuffled_query     q_ds is replaced by the query of a random TRAINING pair, so relation
                       weights no longer match the pair being scored
    mean_query         q_ds is replaced by the mean training-pair query: weights then depend only
                       on the endpoint's own summaries (node-conditioned, not pair-conditioned)
    uniform_weights    alpha forced to 1/|R_v| inside the trained gated model
If the gate's benefit comes from pair conditioning, shuffled_query and mean_query should
hurt; if they do not, the gate is acting as a node-level relation weighting at best.

Gate statistics (from the test AP pool: test positives + sampled unlabeled pairs):
    mean alpha per relation, split by label (positive / unlabeled), for drug and disease ends
    normalised entropy H(alpha)/log|R_avail|  (1 = uniform, 0 = one relation)
    within-drug variation: for each drug scored against >= 2 diseases, the mean std of its drug-side
        alpha across those diseases.  Exactly 0 for any node-only weighting; > 0 only if the
        drug's weights change with the disease, i.e. the gate is really pair-conditioned.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .evaluate import evaluate_pool, make_nn_scorer
from .models import ENDPOINTS, GateDecoder
from .utils import write_json

CONTROLS = ("gated", "shuffled_query", "mean_query", "uniform_weights")


class _QueryTransform(nn.Module):
    """Replaces the gate query q_ds with one that does not belong to the scored pair.

    The replacement comes from a fixed bank of queries of TRAINING pairs, so the control does not
    depend on how the evaluator batches pairs (a ranking batch holds one disease and every drug,
    and a batch mean there would still carry that disease's information).
      shuffled_query  q of a random training pair       (pair information destroyed)
      mean_query      mean q over the training bank     (one constant query for every pair)
    """

    def __init__(self, inner: nn.Module, mode: str, bank: torch.Tensor, seed: int = 0):
        super().__init__()
        self.inner, self.mode, self.bank = inner, mode, bank
        self.mean = bank.mean(0, keepdim=True)
        self.gen = torch.Generator().manual_seed(seed)

    def forward(self, x):
        q = self.inner(x)
        if self.mode == "shuffled_query":
            idx = torch.randint(len(self.bank), (len(q),), generator=self.gen).to(q.device)
            return self.bank[idx]
        if self.mode == "mean_query":
            return self.mean.expand_as(q)
        return q


@torch.no_grad()
def query_bank(model, graph, train_pairs: np.ndarray, device, n: int = 4096, seed: int = 0):
    """Gate queries q_ds of up to n training positives (computed on the full training graph)."""
    model.eval()
    rng = np.random.default_rng(seed)
    p = train_pairs[rng.choice(len(train_pairs), min(n, len(train_pairs)), replace=False)]
    h, _ = model.encode(graph.x, graph.edge_index)
    hd, hs = h["drug"][torch.as_tensor(p[:, 0], device=device)], h["disease"][torch.as_tensor(p[:, 1], device=device)]
    return model.decoder.q(torch.cat([hd, hs, hd * hs], -1))


@contextlib.contextmanager
def gate_control(model, mode: str, bank: torch.Tensor | None = None, seed: int = 0):
    """Temporarily apply an evaluation-time control to a trained gated model."""
    dec = model.decoder
    if not isinstance(dec, GateDecoder) or dec.mode != "gated":
        raise ValueError("gate controls need a trained `gated` model")
    if mode == "gated":
        yield model
        return
    if mode == "uniform_weights":
        dec.mode = "uniform"
        try:
            yield model
        finally:
            dec.mode = "gated"
        return
    if bank is None:
        raise ValueError(f"{mode} needs a query bank (see query_bank)")
    orig = dec.q
    dec.q = _QueryTransform(orig, mode, bank, seed)
    try:
        yield model
    finally:
        dec.q = orig


@torch.no_grad()
def gate_weights(model, graph, pairs: np.ndarray, device, chunk: int = 1 << 15):
    """alpha for each pair: {'drug': [P, R_d], 'disease': [P, R_s]} plus availability masks."""
    model.eval()
    h, ctx = model.encode(graph.x, graph.edge_index)
    out = {v: [] for v in ENDPOINTS}
    for i in range(0, len(pairs), chunk):
        d = torch.as_tensor(pairs[i:i + chunk, 0], device=device)
        s = torch.as_tensor(pairs[i:i + chunk, 1], device=device)
        _, alpha = model.score(h, ctx, d, s, return_alpha=True)
        for v in ENDPOINTS:
            out[v].append(alpha[v].float().cpu())
    alpha = {v: torch.cat(out[v]).numpy() for v in ENDPOINTS}
    masks = {"drug": ctx["drug"][1][torch.as_tensor(pairs[:, 0], device=device)].cpu().numpy(),
             "disease": ctx["disease"][1][torch.as_tensor(pairs[:, 1], device=device)].cpu().numpy()}
    return alpha, masks


def _norm_entropy(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    k = m.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(np.where(a > 0, a * np.log(np.clip(a, 1e-12, None)), 0.0)).sum(1)
        return np.where(k > 1, H / np.log(np.maximum(k, 2)), np.nan)


def within_drug_variation(alpha_drug: np.ndarray, drugs: np.ndarray) -> float:
    vals = []
    for d in np.unique(drugs):
        idx = np.flatnonzero(drugs == d)
        if len(idx) >= 2:
            vals.append(alpha_drug[idx].std(0).mean())
    return float(np.mean(vals)) if vals else float("nan")


def analyse(cfg, load_model_fn, setting: str, seeds, device, log=print) -> Path:
    res = Path(cfg["paths"]["results"])
    out_dir = res / "gates" / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl_rows, rel_rows, summ = [], [], {}
    for seed in seeds:
        split, graph, model = load_model_fn(cfg, setting, "gated", seed, device)
        pool = split.test_pool
        pairs, labels = pool.ap_pairs, pool.ap_labels
        # ---------------- evaluation-time controls (same checkpoint, same pool, same metric code)
        bank = query_bank(model, graph, split.train, device, seed=seed)
        for mode in CONTROLS:
            with gate_control(model, mode, bank=bank, seed=seed):
                m = evaluate_pool(make_nn_scorer(model, graph, device), pool)
            ctrl_rows.append({"setting": setting, "seed": seed, "control": mode,
                              **{k: m[k] for k in ("ap", "recall@10", "auroc", "ndcg@10", "precision@10")}})
            log(f"  [{setting}] seed {seed} {mode:16s} AP={m['ap']:.4f}  R@10={m['recall@10']:.4f}")
        # ---------------- gate statistics
        alpha, masks = gate_weights(model, graph, pairs, device)
        s = {}
        for v in ENDPOINTS:
            rels = [et[1] for et in model.summ.rels[v]]
            ent = _norm_entropy(alpha[v], masks[v])
            s[v] = {"n_relations": len(rels),
                    "mean_norm_entropy": float(np.nanmean(ent)),
                    "frac_pairs_no_relation": float((masks[v].sum(1) == 0).mean())}
            for j, r in enumerate(rels):
                for lab, name in ((1, "positive"), (0, "unlabeled")):
                    sel = labels == lab
                    avail = masks[v][sel, j] > 0
                    rel_rows.append({"setting": setting, "seed": seed, "endpoint": v, "relation": r,
                                     "label": name, "mean_alpha": float(alpha[v][sel, j].mean()),
                                     "mean_alpha_when_available": float(alpha[v][sel, j][avail].mean())
                                     if avail.any() else float("nan"),
                                     "availability": float(avail.mean())})
        s["within_drug_alpha_std"] = within_drug_variation(alpha["drug"], pairs[:, 0])
        s["within_disease_alpha_std"] = within_drug_variation(alpha["disease"], pairs[:, 1])
        summ[f"seed{seed}"] = s
        del model, graph, split
    ctrl = pd.DataFrame(ctrl_rows)
    rel = pd.DataFrame(rel_rows)
    ctrl.to_csv(out_dir / "gate_controls.csv", index=False)
    rel.to_csv(out_dir / "gate_relations.csv", index=False)
    write_json(summ, out_dir / "gate_summary.json")
    _write_md(ctrl, rel, summ, setting, out_dir / "gates.md")
    log(f"wrote {out_dir}")
    return out_dir / "gates.md"


def _write_md(ctrl, rel, summ, setting, path):
    L = [f"# Gate analysis - {setting}-held-out", "",
         "Evaluation-time controls applied to the trained gated checkpoints (same pools, same metric code).",
         "Δ is control − gated, per seed, then averaged.", "",
         "| control | AP (mean ± sd) | R@10 (mean ± sd) | ΔAP | ΔR@10 | ΔAP sign across seeds |",
         "|---|---|---|---|---|---|"]
    base = ctrl[ctrl.control == "gated"].set_index("seed")
    for mode in CONTROLS:
        c = ctrl[ctrl.control == mode].set_index("seed")
        dap = (c["ap"] - base["ap"]).dropna()
        dr = (c["recall@10"] - base["recall@10"]).dropna()
        sign = "—" if mode == "gated" else ("all −" if (dap < 0).all() else "all +" if (dap > 0).all() else "mixed")
        L.append(f"| {mode} | {c['ap'].mean():.4f} ± {c['ap'].std(ddof=0):.4f} | "
                 f"{c['recall@10'].mean():.4f} ± {c['recall@10'].std(ddof=0):.4f} | "
                 f"{dap.mean():+.4f} | {dr.mean():+.4f} | {sign} |")
    L += ["", "## Is the gate pair-conditioned?", "",
          "Within-drug std: how much a drug's relation weights change across the diseases it is scored "
          "against (0 = node-only weighting).", "",
          "| seed | drug entropy | disease entropy | within-drug α std | within-disease α std |",
          "|---|---|---|---|---|"]
    for k, s in summ.items():
        L.append(f"| {k} | {s['drug']['mean_norm_entropy']:.3f} | {s['disease']['mean_norm_entropy']:.3f} | "
                 f"{s['within_drug_alpha_std']:.2e} | {s['within_disease_alpha_std']:.2e} |")
    L += ["", "## Mean relation weight (averaged over seeds)", ""]
    piv = rel.groupby(["endpoint", "relation", "label"])["mean_alpha"].mean().unstack("label").reset_index()
    for v in ENDPOINTS:
        p = piv[piv.endpoint == v].sort_values("positive", ascending=False)
        L += [f"### {v} endpoint", "", "| relation | α positives | α unlabeled |", "|---|---|---|"]
        L += [f"| {r.relation} | {r.positive:.3f} | {r.unlabeled:.3f} |" for r in p.itertuples()]
        L.append("")
    L.append("Weights describe what the model relies on; they are not biological evidence.")
    Path(path).write_text("\n".join(L), encoding="utf-8")