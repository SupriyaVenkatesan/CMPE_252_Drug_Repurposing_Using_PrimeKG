"""Result aggregation (proposal 11.3-11.4): per-seed values, mean +- std, paired seed
differences (gated - baseline) and bootstrap CIs that resample held-out diseases."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import read_json

METRICS = ["ap", "recall@10", "auroc", "precision@10", "ndcg@10", "recall@20"]
ORDER = ["popularity", "sage", "hgt", "uniform", "param_matched", "gated"]
# Contrasts the study is designed around (proposal E4), plus every learned model against the
# popularity reference. Pairs whose models have no results are skipped, so the table appears as
# soon as any two comparable models exist instead of staying empty until the gate is trained.
DESIGNED = [("gated", "hgt"), ("gated", "uniform"), ("gated", "param_matched"),
            ("hgt", "sage"), ("hgt", "popularity"), ("sage", "popularity")]


def _comparisons(models: set) -> list[tuple[str, str]]:
    out = [p for p in DESIGNED if p[0] in models and p[1] in models]
    if "popularity" in models:
        for m in ORDER:
            if m not in ("popularity",) and m in models and (m, "popularity") not in out:
                out.append((m, "popularity"))
    return out


def _seed_pairs(seeds_a: list, seeds_b: list) -> list[tuple]:
    """Pair runs for a paired difference. A deterministic model (popularity) has a single run,
    so each of the other model's seeds is compared against that same reference run; otherwise
    seeds are matched one-to-one."""
    if len(seeds_b) == 1:
        return [(s, seeds_b[0]) for s in seeds_a]
    if len(seeds_a) == 1:
        return [(seeds_a[0], s) for s in seeds_b]
    return [(s, s) for s in sorted(set(seeds_a) & set(seeds_b))]


def _load(results: Path) -> list[dict]:
    return [read_json(p) for p in sorted(results.glob("*/*/seed*/metrics.json"))]


def _bootstrap_diff(runs_a, runs_b, n_boot, rng):
    """CI for macro Recall@10 difference, resampling held-out diseases (seed-averaged)."""
    qa = runs_a[0]["per_query"]["disease"]
    if any(r["per_query"]["disease"] != qa for r in runs_a + runs_b):
        return float("nan"), float("nan")
    a = np.mean([r["per_query"]["recall@10"] for r in runs_a], 0)
    b = np.mean([r["per_query"]["recall@10"] for r in runs_b], 0)
    diff = a - b
    if len(diff) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(results: Path, n_boot: int = 1000) -> Path:
    runs = _load(results)
    if not runs:
        raise SystemExit(f"no metrics.json under {results} - nothing to summarise. "
                         f"Run `pcgate run` for this config first (if it failed, fix that first: "
                         f"an empty results directory is a symptom, not the cause).")
    rows = []
    for r in runs:
        row = {"setting": r["setting"], "model": r["model"], "seed": r["seed"]}
        row.update({m: r[m] for m in METRICS})
        row.update({"train_time_s": r["train"]["train_time_s"], "peak_memory_mb": r["train"].get("peak_memory_mb"),
                    "throughput_pairs_per_s": r["throughput_pairs_per_s"], "params": r["train"]["params"]["total"],
                    "epochs": r["train"].get("epochs_run", 0), "n_queries": r["n_queries"],
                    "mean_pool_size": r["mean_pool_size"], "n_test_pairs": r["n_positive_pairs"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results / "all_runs.csv", index=False)
    rng = np.random.default_rng(0)

    lines = ["# Results summary", "",
             "Metrics measure recovery of withheld documented indications against unlabeled pairs; "
             "they do not measure therapeutic efficacy.", ""]
    for setting, g in df.groupby("setting", sort=False):
        lines += [f"## Setting: {setting}", "",
                  f"Test pairs: {int(g['n_test_pairs'].iloc[0])}, evaluable diseases (queries): "
                  f"{int(g['n_queries'].iloc[0])}, mean candidate pool: {g['mean_pool_size'].iloc[0]:.0f}", ""]
        hdr = "| model | n seeds | " + " | ".join(METRICS) + " | params | train s | peak MB | pairs/s |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        models = [m for m in ORDER if m in set(g["model"])] + sorted(set(g["model"]) - set(ORDER))
        for m in models:
            x = g[g["model"] == m]
            cells = [f"{x[k].mean():.4f} ± {x[k].std(ddof=0):.4f}" for k in METRICS]
            lines.append(f"| {m} | {len(x)} | " + " | ".join(cells) +
                         f" | {int(x['params'].mean())} | {x['train_time_s'].mean():.1f} | "
                         f"{x['peak_memory_mb'].mean():.0f} | {x['throughput_pairs_per_s'].mean():.0f} |")
        lines.append("")
        # paired seed differences + bootstrap CI
        comp_lines = []
        for a, b in _comparisons(set(g["model"])):
            xa, xb = g[g["model"] == a].set_index("seed"), g[g["model"] == b].set_index("seed")
            pairs = _seed_pairs(sorted(xa.index), sorted(xb.index))
            if not pairs:
                continue
            d_ap = [xa.loc[i, "ap"] - xb.loc[j, "ap"] for i, j in pairs]
            d_r = [xa.loc[i, "recall@10"] - xb.loc[j, "recall@10"] for i, j in pairs]
            sel = lambda m, ss: [r for r in runs if r["setting"] == setting and r["model"] == m
                                 and r["seed"] in ss]
            lo, hi = _bootstrap_diff(sel(a, {i for i, _ in pairs}), sel(b, {j for _, j in pairs}),
                                     n_boot, rng)
            sign = "all +" if all(v > 0 for v in d_ap) else "all −" if all(v < 0 for v in d_ap) else "mixed"
            comp_lines.append(
                f"| {a} − {b} | {len(pairs)} | {', '.join(f'{v:+.4f}' for v in d_ap)} | {np.mean(d_ap):+.4f} | "
                f"{sign} | {np.mean(d_r):+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        if comp_lines:
            lines += ["Paired differences. A deterministic model (popularity) has one run, so each "
                      "seed of the other model is compared against it; otherwise seeds are matched "
                      "one-to-one. 'ΔAP sign' says whether the difference holds across every seed.", "",
                      "| comparison | n | ΔAP per seed | mean ΔAP | ΔAP sign | mean ΔR@10 | 95% bootstrap CI ΔR@10 (diseases) |",
                      "|---|---|---|---|---|---|---|"] + comp_lines + [""]
    out = results / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
