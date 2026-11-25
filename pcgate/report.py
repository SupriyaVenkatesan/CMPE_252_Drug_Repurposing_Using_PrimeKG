"""Result aggregation (proposal 11.3-11.4): per-seed values, mean +- std, paired seed
differences (gated - baseline) and bootstrap CIs that resample held-out diseases."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import read_json

METRICS = ["ap", "recall@10", "auroc", "precision@10", "ndcg@10", "recall@20"]
ORDER = ["popularity", "sage", "hgt", "uniform", "param_matched", "gated"]
COMPARISONS = [("gated", "hgt"), ("gated", "uniform"), ("gated", "param_matched"), ("hgt", "sage")]


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
        for a, b in COMPARISONS:
            xa, xb = g[g["model"] == a].set_index("seed"), g[g["model"] == b].set_index("seed")
            seeds = sorted(set(xa.index) & set(xb.index))
            if not seeds:
                continue
            d_ap = [xa.loc[s, "ap"] - xb.loc[s, "ap"] for s in seeds]
            d_r = [xa.loc[s, "recall@10"] - xb.loc[s, "recall@10"] for s in seeds]
            ra = [r for r in runs if r["setting"] == setting and r["model"] == a and r["seed"] in seeds]
            rb = [r for r in runs if r["setting"] == setting and r["model"] == b and r["seed"] in seeds]
            lo, hi = _bootstrap_diff(ra, rb, n_boot, rng)
            comp_lines.append(
                f"| {a} − {b} | {len(seeds)} | {', '.join(f'{v:+.4f}' for v in d_ap)} | {np.mean(d_ap):+.4f} | "
                f"{', '.join(f'{v:+.4f}' for v in d_r)} | {np.mean(d_r):+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        if comp_lines:
            lines += ["Paired seed differences (same split, same seeds):", "",
                      "| comparison | seeds | ΔAP per seed | mean ΔAP | ΔR@10 per seed | mean ΔR@10 | 95% bootstrap CI ΔR@10 (diseases) |",
                      "|---|---|---|---|---|---|---|"] + comp_lines + [""]
    out = results / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
