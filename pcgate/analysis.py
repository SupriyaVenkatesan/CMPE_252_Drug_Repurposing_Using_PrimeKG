"""Analysis tooling.

  schema_report   dataset pin + schema + usage-terms report from audit.json / provenance.json
  profile         memory / runtime study per model and hidden size
  coverage        held-out coverage, degree of evaluated entities, degree-stratified Recall@10
  pool_diagnostic is the candidate pool itself separable? (drug-eligibility artefact)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data.splits import SETTINGS, Split
from .utils import read_json, write_json


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


# ============================================================================ Week 1
def schema_report(raw_csv: str, processed: str, out_dir: str | None = None) -> str:
    """Write the dataset pin / schema / usage-terms report.

    Writes to `out_dir` (the results directory) rather than next to the processed data, which
    is gitignored — this is a reportable deliverable and needs to reach the repository.
    """
    processed = Path(processed)
    audit = read_json(processed / "audit.json")
    prov_p = Path(raw_csv).parent / "provenance.json"
    prov = read_json(prov_p) if prov_p.exists() else {}
    L = ["# Dataset pin, schema and usage terms", ""]
    L += ["## Pinned snapshot", "",
          "| field | value |", "|---|---|",
          f"| dataset | {prov.get('dataset', 'unknown (no provenance.json)')} |",
          f"| file | {prov.get('file', Path(raw_csv).name)} |",
          f"| source URL | {prov.get('url', '-')} |",
          f"| download date (UTC) | {prov.get('download_date_utc', '-')} |",
          f"| SHA-256 | `{audit['raw_sha256']}` |",
          f"| size (bytes) | {prov.get('size_bytes', '-')} |",
          f"| nodes.csv SHA-256 | `{audit['nodes_sha256']}` |",
          f"| edges.csv SHA-256 | `{audit['edges_sha256']}` |", ""]
    L += ["## Usage terms", "",
          "* The PrimeKG GitHub repository's MIT licence covers **software and tools only**.",
          "* Dataset use is governed by the terms on the Harvard Dataverse record "
          "(doi:10.7910/DVN/IXA7BM) and by the licences of PrimeKG's upstream sources. "
          "Record the Dataverse terms text here when the file is downloaded.",
          f"* Provenance note: {prov.get('data_terms_note', '-')}", "",
          "No patient records, paid data or the original HGTDR dataset are used.", ""]
    L += ["## Raw -> clean counts", "", "| stage | count |", "|---|---|",
          f"| raw rows | {audit['raw_rows']} |",
          f"| invalid rows (logged to invalid_rows.csv) | {audit['invalid_rows']} |",
          f"| rows after invalid removal | {audit['rows_after_invalid_removal']} |",
          f"| duplicate / reverse-copy rows removed | {audit['duplicate_or_reverse_rows_removed']} |",
          f"| unique canonical edges | {audit['unique_canonical_edges']} |",
          f"| unique indication pairs | {audit['unique_indication_pairs']} |",
          f"| drugs with >=1 indication | {audit['drugs_with_indication']} |",
          f"| diseases with >=1 indication | {audit['diseases_with_indication']} |", ""]
    if audit.get("note_edge_count"):
        L += ["", "**Why the edge count differs from a naive rows / 2.** " + audit["note_edge_count"], ""]
        by_rel = audit.get("edges_with_multiple_display_relations_by_relation", {})
        if by_rel:
            L += ["Affected relations: " + ", ".join(f"`{k}` ({v})" for k, v in by_rel.items()),
                  "", "These keep one edge each; only the `display_relation` sub-type is dropped, "
                  "and no model uses that column.", ""]
    if audit.get("invalid_reasons"):
        L += ["Invalid-row reasons: " + ", ".join(f"`{k}` x{v}" for k, v in audit["invalid_reasons"].items()), ""]
    miss = {k: v for k, v in audit["missing_values_per_column"].items() if v}
    L += [f"Missing values per column (raw): {miss if miss else 'none'}", ""]
    L += ["## Node types", "", "| node type | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(audit["node_type_counts"].items(), key=lambda kv: -kv[1])]
    L += ["", "The combined `gene/protein` type is kept as released (sanitised to `gene_protein`). "
          "No new node types or inferred associations are created.", ""]
    L += ["## Edge types (canonical direction; the graph adds `rev_*` copies)", "",
          "| edge type | count |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in sorted(audit["edge_type_counts_clean"].items(), key=lambda kv: -kv[1])]
    L += ["", "## Drug-disease relations", "",
          "| relation | edges | overlap with indication pairs | used in graph? |", "|---|---|---|---|"]
    for r, n in audit.get("drug_disease_relations", {}).items():
        ov = audit["indication_overlap_with_other_drug_disease_relations"].get(r, 0)
        L.append(f"| {r} | {n} | {ov} | no (treatment-label proxy) |")
    L.append(f"| indication | {audit['unique_indication_pairs']} | - | training partition only |")
    L += ["", "Contraindication and off-label relations are never merged into indication labels.", ""]
    out = Path(out_dir or processed) / "dataset_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return str(out)


# ============================================================================ Week 3
def _peak_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**20
    except Exception:
        return float("nan")


def profile(cfg: dict, graph, split, device, models=("sage", "hgt", "gated"), hiddens=(64, 128),
            batch_sizes=(256, 1024), steps: int = 5) -> pd.DataFrame:
    from .evaluate import make_nn_scorer
    from .models import LinkModel
    from .train import NegativeSampler

    in_dim = next(iter(graph.x.values())).size(1)
    pos_all = torch.as_tensor(split.train, device=device)
    sampler = NegativeSampler(split, graph.num_nodes["drug"], graph.num_nodes["disease"], device)
    rows = []
    for m in models:
        for H in hiddens:
            for bs in batch_sizes:
                mcfg = dict(cfg.get("model", {}), hidden=H, decoder_hidden=H)
                torch.manual_seed(0)
                model = LinkModel(m, graph.metadata(), in_dim, mcfg, graph.num_nodes).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                times = []
                for _ in range(steps):
                    idx = torch.randint(len(pos_all), (min(bs, len(pos_all)),), device=device)
                    pos = pos_all[idx]
                    neg = sampler(pos, 5)
                    t0 = time.perf_counter()
                    h, ctx = model.encode(graph.x, graph.masked_edges(idx))
                    lp = model.score(h, ctx, pos[:, 0], pos[:, 1])
                    ln = model.score(h, ctx, neg[:, 0], neg[:, 1])
                    loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
                        F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                scorer = make_nn_scorer(model, graph, device)
                n = 50_000
                d = np.random.default_rng(0).integers(0, graph.num_nodes["drug"], n)
                s = np.random.default_rng(1).integers(0, graph.num_nodes["disease"], n)
                t0 = time.perf_counter()
                scorer(d, s)
                thr = n / (time.perf_counter() - t0)
                steps_per_epoch = int(np.ceil(len(pos_all) / bs))
                rows.append({"model": m, "hidden": H, "batch_size": bs,
                             "params": model.param_report()["total"],
                             "step_s_median": float(np.median(times[1:] if len(times) > 1 else times)),
                             "est_epoch_s": float(np.median(times[1:] if len(times) > 1 else times)) * steps_per_epoch,
                             "peak_mem_mb": _peak_mb(device), "score_pairs_per_s": thr,
                             "device": device.type})
    return pd.DataFrame(rows)


# ============================================================================ pool diagnostic
def pool_diagnostic(cfg, load_fn, results: Path) -> str:
    """How much of AP/AUROC is explained by drug *eligibility* alone?

    Only 1,801 of PrimeKG's 7,957 drug-type nodes ever carry an indication, but the unlabeled
    pool is drawn from all of them. A scorer that merely answers "is this drug ever indicated
    for anything?" can therefore separate most positives from most unlabeled pairs without
    using the disease at all.

    Three measurements on the SAME fixed pools the models were scored on:
      * share of unlabeled pairs whose drug has no indication anywhere (trivially rejectable);
      * an *eligibility oracle* that scores 1 for any ever-indicated drug and 0 otherwise -
        it ignores the disease completely, so its AP is the ceiling reachable by the artefact;
      * popularity restricted to eligible drugs, reported as lift over prevalence (AP / base
        rate) because dropping negatives changes the base rate and raw APs stop being comparable.

    Read-only: nothing on disk is modified, and no model checkpoint is needed.
    """
    from .evaluate import evaluate_pool, make_pop_scorer
    from .models import Popularity

    rows, rank_rows = [], []
    for setting in SETTINGS:
        split, graph = load_fn(cfg, setting, torch.device("cpu"))
        pool = split.test_pool
        n_drugs = graph.num_nodes["drug"]
        eligible = np.zeros(n_drugs, dtype=bool)
        eligible[np.unique(split.all_known[:, 0])] = True

        pairs, labels = pool.ap_pairs, pool.ap_labels
        unl = pairs[labels == 0]
        pos = pairs[labels == 1]
        rng = np.random.default_rng(0)
        jitter = rng.random(n_drugs) * 1e-6          # deterministic tie-break, same for every scorer

        # --- eligibility oracle: knows only "is this drug ever indicated", never the disease
        oracle = lambda d, s: eligible[np.asarray(d)].astype(float) + jitter[np.asarray(d)]
        o = evaluate_pool(oracle, pool)

        # --- popularity, on the full pool and on the eligible-only subset of the same pool
        deg = torch.zeros(n_drugs, dtype=torch.long)
        for (st, _, dt), e in graph.edge_index.items():
            if dt == "drug":
                deg += torch.bincount(e[1].cpu(), minlength=n_drugs)
        pop = make_pop_scorer(Popularity(n_drugs, split.train, deg))
        full = evaluate_pool(pop, pool)

        keep = eligible[pairs[:, 0]]
        sub = type(pool)(**{**pool.__dict__, "ap_pairs": pairs[keep], "ap_labels": labels[keep],
                            "cand_drugs": pool.cand_drugs[eligible[pool.cand_drugs]]})
        restricted = evaluate_pool(pop, sub)

        prev_full = labels.mean()
        prev_sub = sub.ap_labels.mean()
        rows.append({
            "setting": setting,
            "cand_drugs": len(pool.cand_drugs),
            "eligible": int(eligible[pool.cand_drugs].sum()),
            "unlabeled_pairs": len(unl),
            "unlab_drug_never_indicated_%": 100 * float((~eligible[unl[:, 0]]).mean()),
            "oracle_ap": o["ap"], "oracle_auroc": o["auroc"], "oracle_lift": o["ap"] / prev_full,
            "pop_ap_full": full["ap"], "pop_lift_full": full["ap"] / prev_full,
            "pop_ap_eligible": restricted["ap"], "pop_lift_eligible": restricted["ap"] / prev_sub,
            "prevalence_full": float(prev_full), "prevalence_eligible": float(prev_sub),
        })
        rank_rows.append({"setting": setting, "metric": "popularity Recall@10",
                          "all_drugs": full["recall@10"], "eligible_only": restricted["recall@10"],
                          "oracle_all_drugs": o["recall@10"]})
        assert len(pos) == int(labels.sum())

    df = pd.DataFrame(rows)
    rank = pd.DataFrame(rank_rows)
    L = ["# Candidate-pool diagnostic: the drug-eligibility artefact", "",
         pool_diagnostic.__doc__.split("Three measurements")[0].strip(), "",
         "**Lift = AP / prevalence.** A random scorer has lift 1.0 whatever the base rate, so lift is",
         "comparable across pools while raw AP is not.", "",
         "## Pool composition and AP decomposition", "", _md_table(df), "",
         "## Ranking metrics (unaffected by the unlabeled sample, affected by the candidate set)", "",
         _md_table(rank), ""]
    out = results / "pool_diagnostic.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results / "pool_diagnostic.csv", index=False)
    out.write_text("\n".join(L), encoding="utf-8")
    return str(out)


# ============================================================================ Week 6
def _degrees(graph, ntype):
    deg = np.zeros(graph.num_nodes[ntype])
    for (st, _, dt), ei in graph.edge_index.items():
        if dt == ntype:
            deg += np.bincount(ei[1].cpu().numpy(), minlength=len(deg))
    return deg


def coverage(cfg: dict, load_fn, results: Path, n_buckets: int = 3) -> str:
    """Coverage of held-out labels and degree-stratified Recall@10 per model."""
    L = ["# Coverage and degree analysis", "",
         "Evaluable = held-out disease with at least one positive in the partition and a non-empty candidate pool. "
         "Diseases are bucketed by their degree in the training graph (tertiles) to expose uneven coverage.", ""]
    cov_rows, strat_rows = [], []
    for setting in SETTINGS:
        try:
            split, graph = load_fn(cfg, setting, torch.device("cpu"))
        except FileNotFoundError:
            continue
        te = split.test
        deg_s = _degrees(graph, "disease")
        deg_d = _degrees(graph, "drug")
        q = split.test_pool.queries
        cov_rows.append({
            "setting": setting, "test_pairs": len(te),
            "test_diseases": int(len(np.unique(te[:, 1]))), "evaluable_diseases": int(len(q)),
            "test_drugs": int(len(np.unique(te[:, 0]))),
            "candidate_universe": int(len(split.test_pool.cand_drugs)),
            "median_pos_per_disease": float(np.median(np.diff(split.test_pool.pos_ptr))) if len(q) else 0.0,
            "median_test_disease_degree": float(np.median(deg_s[q])) if len(q) else 0.0,
            "median_test_drug_degree": float(np.median(deg_d[np.unique(te[:, 0])])) if len(te) else 0.0,
            "test_diseases_zero_train_labels": int(sum(1 for s in q if s not in set(split.train[:, 1].tolist()))),
        })
        if len(q) == 0:
            continue
        edges = np.unique(np.quantile(deg_s[q], np.linspace(0, 1, n_buckets + 1)))
        bucket = np.clip(np.searchsorted(edges, deg_s[q], side="right") - 1, 0, max(len(edges) - 2, 0))
        bmap = dict(zip(q.tolist(), bucket.tolist()))
        for mp in sorted((results / setting).glob("*/seed*/metrics.json")):
            m = read_json(mp)
            pq = m["per_query"]
            for b in sorted(set(bucket.tolist())):
                vals = [r for s, r in zip(pq["disease"], pq["recall@10"]) if bmap.get(s) == b]
                strat_rows.append({"setting": setting, "model": m["model"], "seed": m["seed"],
                                   "degree_bucket": f"[{edges[b]:.0f}, {edges[min(b + 1, len(edges) - 1)]:.0f}]",
                                   "n_diseases": len(vals), "recall@10": float(np.mean(vals)) if vals else float("nan")})
    cov = pd.DataFrame(cov_rows)
    L += ["## Coverage per split", "", _md_table(cov), ""]
    if strat_rows:
        st = pd.DataFrame(strat_rows)
        agg = (st.groupby(["setting", "model", "degree_bucket"], sort=False)
               .agg(n_diseases=("n_diseases", "first"), recall10_mean=("recall@10", "mean"),
                    recall10_std=("recall@10", "std"), seeds=("seed", "nunique")).reset_index())
        agg["recall10_std"] = agg["recall10_std"].fillna(0.0)
        L += ["## Recall@10 by test-disease degree bucket (mean ± std over seeds)", "", _md_table(agg), ""]
        st.to_csv(results / "coverage_by_degree.csv", index=False)
    cov.to_csv(results / "coverage.csv", index=False)
    out = results / "coverage.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return str(out)
