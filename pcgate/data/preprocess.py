"""PrimeKG download, audit, identifier mapping, cleaning and target extraction.

Stages (proposal section 6.1):
  Data audit         -> audit.json         (columns, missing ids, relation labels, endpoint types)
  Identifier mapping -> nodes.csv          (stable type-specific indices + source identifiers)
  Cleaning           -> edges.csv, invalid_rows.csv (deduplicated canonical edge list, logged rejects)
  Target extraction  -> indications.csv    (unique documented drug -> disease indication pairs)
Every output is produced by code; nothing is edited by hand.
"""
from __future__ import annotations

import datetime as _dt
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import sanitize_relation, sanitize_type, sha256_file, write_json

PRIMEKG_KG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"  # kg.csv
REQUIRED_COLUMNS = [
    "relation", "display_relation",
    "x_index", "x_id", "x_type", "x_name", "x_source",
    "y_index", "y_id", "y_type", "y_name", "y_source",
]
KNOWN_TYPES = {
    "drug", "disease", "gene/protein", "effect/phenotype", "pathway",
    "biological_process", "molecular_function", "cellular_component",
    "anatomy", "exposure",
}


def download_primekg(raw_dir: str | Path, url: str = PRIMEKG_KG_URL) -> dict:
    """Download kg.csv and write a provenance record (version, date, SHA-256)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "kg.csv"
    if not out.exists():
        print(f"Downloading {url} -> {out} (~1 GB) ...")
        part = out.with_suffix(".csv.part")
        # Dataverse rejects Python's default User-Agent with HTTP 403.
        req = urllib.request.Request(url, headers={"User-Agent": "pcgate-downloader/0.1"})
        with urllib.request.urlopen(req) as r, open(part, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        part.replace(out)
    prov = {
        "dataset": "PrimeKG (Harvard Dataverse doi:10.7910/DVN/IXA7BM)",
        "file": "kg.csv",
        "url": url,
        "download_date_utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
        "sha256": sha256_file(out),
        "size_bytes": out.stat().st_size,
        "data_terms_note": "Dataset terms are separate from the MIT software licence of the PrimeKG repo; "
                           "see the Dataverse record for data terms.",
    }
    write_json(prov, raw_dir / "provenance.json")
    return prov


def _type_rank(t: str) -> tuple:
    order = {"drug": 0, "disease": 1}
    return (order.get(t, 2), t)


def preprocess(raw_csv: str | Path, out_dir: str | Path, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    raw_csv, out_dir = Path(raw_csv), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indication_relation = cfg.get("indication_relation", "indication")

    df = pd.read_csv(raw_csv, low_memory=False, dtype={"x_id": str, "y_id": str})
    audit: dict = {"raw_file": str(raw_csv), "raw_sha256": sha256_file(raw_csv),
                   "raw_rows": int(len(df)), "columns": list(df.columns)}
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    audit["missing_columns"] = missing_cols
    if missing_cols:
        raise ValueError(f"kg.csv missing required columns: {missing_cols}")

    # ---------------- audit ----------------
    audit["missing_values_per_column"] = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    audit["relation_counts_raw"] = df["relation"].value_counts().to_dict()
    audit["display_relation_counts_raw"] = df["display_relation"].value_counts().to_dict()
    endpoint = (df["relation"] + " | " + df["x_type"].astype(str) + " -> " + df["y_type"].astype(str))
    audit["relation_endpoint_types_raw"] = endpoint.value_counts().to_dict()
    audit["node_types_raw"] = sorted(set(df["x_type"].dropna()) | set(df["y_type"].dropna()))

    # ---------------- cleaning: invalid rows ----------------
    invalid_reason = pd.Series("", index=df.index)
    for c in ["relation", "x_id", "y_id", "x_type", "y_type"]:
        invalid_reason[df[c].isna()] += f"missing_{c};"
    unknown_type = ~df["x_type"].isin(KNOWN_TYPES) | ~df["y_type"].isin(KNOWN_TYPES)
    invalid_reason[unknown_type] += "unknown_node_type;"
    self_loop = (df["x_id"] == df["y_id"]) & (df["x_type"] == df["y_type"])
    invalid_reason[self_loop] += "self_loop;"
    bad = invalid_reason != ""
    rej = df[bad].copy()
    rej["reason"] = invalid_reason[bad]
    rej.to_csv(out_dir / "invalid_rows.csv", index=False)
    df = df[~bad].copy()
    for side in ("x", "y"):
        df[f"{side}_source"] = df[f"{side}_source"].fillna("NA").astype(str)
        df[f"{side}_name"] = df[f"{side}_name"].fillna("").astype(str)
        df[f"{side}_id"] = df[f"{side}_id"].astype(str)
    audit["invalid_rows"] = int(bad.sum())
    audit["invalid_reasons"] = rej["reason"].value_counts().to_dict()

    # ---------------- identifier mapping ----------------
    # Node key = (type, source, id).  PrimeKG ids are only unique within a source.
    xs = df[["x_type", "x_source", "x_id", "x_name"]].set_axis(["type", "source", "id", "name"], axis=1)
    ys = df[["y_type", "y_source", "y_id", "y_name"]].set_axis(["type", "source", "id", "name"], axis=1)
    nodes = (pd.concat([xs, ys], ignore_index=True)
             .drop_duplicates(subset=["type", "source", "id"])
             .assign(type=lambda d: d["type"].map(sanitize_type))
             .sort_values(["type", "source", "id"], kind="mergesort")
             .reset_index(drop=True))
    nodes["type_index"] = nodes.groupby("type").cumcount()
    nodes["global_id"] = np.arange(len(nodes))
    nodes = nodes[["global_id", "type", "type_index", "source", "id", "name"]]
    nodes.to_csv(out_dir / "nodes.csv", index=False)

    key = pd.MultiIndex.from_frame(nodes[["type", "source", "id"]])
    lookup = pd.Series(nodes["type_index"].to_numpy(), index=key)

    def map_side(t, s, i):
        mi = pd.MultiIndex.from_arrays([t.map(sanitize_type), s, i])
        return lookup.reindex(mi).to_numpy()

    e = pd.DataFrame({
        "relation": df["relation"].map(sanitize_relation).to_numpy(),
        "x_type": df["x_type"].map(sanitize_type).to_numpy(),
        "x": map_side(df["x_type"], df["x_source"], df["x_id"]),
        "y_type": df["y_type"].map(sanitize_type).to_numpy(),
        "y": map_side(df["y_type"], df["y_source"], df["y_id"]),
    })
    assert not np.isnan(e["x"].astype(float)).any() and not np.isnan(e["y"].astype(float)).any()
    e["x"] = e["x"].astype(np.int64)
    e["y"] = e["y"].astype(np.int64)

    # ---------------- canonical orientation + de-duplication ----------------
    # PrimeKG stores each undirected relationship twice (x->y and y->x).
    # Canonical form: drug first, disease second, then other types alphabetically;
    # same-type edges oriented (min, max).
    swap = np.array([_type_rank(a) > _type_rank(b) for a, b in zip(e["x_type"], e["y_type"])])
    same = e["x_type"].to_numpy() == e["y_type"].to_numpy()
    swap |= same & (e["x"].to_numpy() > e["y"].to_numpy())
    src_t = np.where(swap, e["y_type"], e["x_type"])
    dst_t = np.where(swap, e["x_type"], e["y_type"])
    src = np.where(swap, e["y"], e["x"])
    dst = np.where(swap, e["x"], e["y"])
    edges = pd.DataFrame({"relation": e["relation"], "src_type": src_t, "src": src,
                          "dst_type": dst_t, "dst": dst})
    before = len(edges)
    edges = edges.drop_duplicates().sort_values(
        ["relation", "src_type", "dst_type", "src", "dst"], kind="mergesort").reset_index(drop=True)
    audit["rows_after_invalid_removal"] = int(before)
    audit["unique_canonical_edges"] = int(len(edges))
    audit["duplicate_or_reverse_rows_removed"] = int(before - len(edges))
    edges.to_csv(out_dir / "edges.csv", index=False)

    # ---------------- target extraction ----------------
    ind_rel = sanitize_relation(indication_relation)
    ind = edges[(edges["relation"] == ind_rel) & (edges["src_type"] == "drug") & (edges["dst_type"] == "disease")]
    ind = ind[["src", "dst"]].rename(columns={"src": "drug", "dst": "disease"}).drop_duplicates()
    ind = ind.sort_values(["drug", "disease"]).reset_index(drop=True)
    ind.to_csv(out_dir / "indications.csv", index=False)

    # Overlap of indications with other drug-disease relations (reported, never merged).
    dd = edges[(edges["src_type"] == "drug") & (edges["dst_type"] == "disease") & (edges["relation"] != ind_rel)]
    ind_keys = set(zip(ind["drug"], ind["disease"]))
    audit["drug_disease_relations"] = dd["relation"].value_counts().to_dict()
    audit["indication_overlap_with_other_drug_disease_relations"] = {
        r: int(sum((a, b) in ind_keys for a, b in zip(g["src"], g["dst"])))
        for r, g in dd.groupby("relation")
    }

    # ---------------- statistics ----------------
    audit["node_type_counts"] = nodes["type"].value_counts().to_dict()
    audit["relation_counts_clean"] = edges["relation"].value_counts().to_dict()
    audit["edge_type_counts_clean"] = (edges["src_type"] + " -[" + edges["relation"] + "]-> " + edges["dst_type"]) \
        .value_counts().to_dict()
    audit["unique_indication_pairs"] = int(len(ind))
    audit["drugs_with_indication"] = int(ind["drug"].nunique())
    audit["diseases_with_indication"] = int(ind["disease"].nunique())
    audit["nodes_sha256"] = sha256_file(out_dir / "nodes.csv")
    audit["edges_sha256"] = sha256_file(out_dir / "edges.csv")
    write_json(audit, out_dir / "audit.json")
    return audit
