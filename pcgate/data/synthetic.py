"""Generate a small synthetic knowledge graph in the exact PrimeKG `kg.csv` format.

Purpose: exercise the whole pipeline (audit -> splits -> models -> evaluation ->
explanations) on a laptop CPU before running on the real 8M-row PrimeKG file.
It is NOT a substitute for PrimeKG and no scientific claim should be drawn from it.

Planted signal (so that models have something to learn):
  * Proteins are grouped into latent modules; pathways and biological processes
    connect proteins of the same module.
  * Every disease has a module and a "mechanism": half the diseases are
    *protein-driven* (drugs treating them target proteins of the disease's
    module) and half are *phenotype-driven* (drugs treating them share side
    effects with the disease's phenotypes).  A pair-conditioned gate can in
    principle learn to weigh the protein vs phenotype relations per pair.
  * Like the real release, every edge is written in both directions and a few
    duplicate / malformed rows are injected to exercise the cleaning stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "relation", "display_relation",
    "x_index", "x_id", "x_type", "x_name", "x_source",
    "y_index", "y_id", "y_type", "y_name", "y_source",
]

NODE_COUNTS = {
    "drug": 400,
    "disease": 300,
    "gene/protein": 1500,
    "effect/phenotype": 400,
    "pathway": 120,
    "biological_process": 200,
    "molecular_function": 80,
    "cellular_component": 50,
    "anatomy": 40,
    "exposure": 30,
}
SOURCES = {
    "drug": "DrugBank", "disease": "MONDO", "gene/protein": "NCBI",
    "effect/phenotype": "HPO", "pathway": "REACTOME", "biological_process": "GO",
    "molecular_function": "GO", "cellular_component": "GO", "anatomy": "UBERON",
    "exposure": "CTD",
}


def generate(out_path: str | Path, seed: int = 7, n_modules: int = 8,
             indications_per_disease: tuple = (2, 10)) -> Path:
    rng = np.random.default_rng(seed)
    nodes = {}  # type -> DataFrame(index, id, name, module)
    gidx = 0
    for t, n in NODE_COUNTS.items():
        prefix = t.split("/")[0][:4].upper()
        nodes[t] = pd.DataFrame({
            "index": np.arange(gidx, gidx + n),
            "id": [f"{prefix}{i:05d}" for i in range(n)],
            "name": [f"{t.split('/')[0]}_{i}" for i in range(n)],
            "module": rng.integers(0, n_modules, size=n),
        })
        gidx += n

    rows: list[tuple] = []

    def add(rel, disp, xt, xi, yt, yi):
        x = nodes[xt].iloc[xi]
        y = nodes[yt].iloc[yi]
        r1 = (rel, disp, x["index"], x["id"], xt, x["name"], SOURCES[xt],
              y["index"], y["id"], yt, y["name"], SOURCES[yt])
        r2 = (rel, disp, y["index"], y["id"], yt, y["name"], SOURCES[yt],
              x["index"], x["id"], xt, x["name"], SOURCES[xt])
        rows.append(r1)
        rows.append(r2)

    def module_members(t, m):
        return np.flatnonzero(nodes[t]["module"].to_numpy() == m)

    def sample_edges(xt, yt, rel, disp, per_x, p_same, symmetric=False):
        nx, ny = len(nodes[xt]), len(nodes[yt])
        seen = set()
        for i in range(nx):
            m = nodes[xt]["module"].iat[i]
            same = module_members(yt, m)
            k = max(1, rng.poisson(per_x))
            for _ in range(k):
                if len(same) and rng.random() < p_same:
                    j = int(rng.choice(same))
                else:
                    j = int(rng.integers(0, ny))
                if symmetric and (j == i):
                    continue
                key = (min(i, j), max(i, j)) if symmetric else (i, j)
                if key in seen:
                    continue
                seen.add(key)
                add(rel, disp, xt, i, yt, j)

    # Biological context
    sample_edges("gene/protein", "gene/protein", "protein_protein", "ppi", 4, 0.7, symmetric=True)
    sample_edges("pathway", "gene/protein", "pathway_protein", "interacts with", 10, 0.85)
    sample_edges("biological_process", "gene/protein", "bioprocess_protein", "interacts with", 6, 0.75)
    sample_edges("molecular_function", "gene/protein", "molfunc_protein", "interacts with", 6, 0.5)
    sample_edges("cellular_component", "gene/protein", "cellcomp_protein", "interacts with", 8, 0.3)
    sample_edges("anatomy", "gene/protein", "anatomy_protein_present", "expression present", 15, 0.2)
    sample_edges("disease", "gene/protein", "disease_protein", "associated with", 5, 0.8)
    sample_edges("disease", "effect/phenotype", "disease_phenotype_positive", "phenotype present", 5, 0.8)
    sample_edges("disease", "disease", "disease_disease", "parent-child", 1, 0.6, symmetric=True)
    sample_edges("drug", "gene/protein", "drug_protein", "target", 3, 0.8)
    sample_edges("drug", "effect/phenotype", "drug_effect", "side effect", 4, 0.8)
    sample_edges("drug", "drug", "drug_drug", "synergistic interaction", 2, 0.3, symmetric=True)
    sample_edges("exposure", "disease", "exposure_disease", "linked to", 4, 0.5)

    # Treatment labels: mechanism-dependent module match.
    n_dis, n_drug = len(nodes["disease"]), len(nodes["drug"])
    mech = rng.integers(0, 2, size=n_dis)  # 0 protein-driven, 1 phenotype-driven
    # Drugs get a second, "phenotype" module used for the phenotype mechanism.
    drug_pheno_module = rng.integers(0, n_modules, size=n_drug)
    dis_mod = nodes["disease"]["module"].to_numpy()
    drug_mod = nodes["drug"]["module"].to_numpy()
    # Re-wire drug_effect signal: phenotype-driven link uses drug_pheno_module,
    # so add extra side-effect edges consistent with that module.
    for i in range(n_drug):
        cand = module_members("effect/phenotype", drug_pheno_module[i])
        if len(cand):
            for j in rng.choice(cand, size=min(3, len(cand)), replace=False):
                add("drug_effect", "side effect", "drug", i, "effect/phenotype", int(j))

    ind = set()
    contra = set()
    for s in range(n_dis):
        if mech[s] == 0:
            pool = np.flatnonzero(drug_mod == dis_mod[s])
        else:
            pool = np.flatnonzero(drug_pheno_module == dis_mod[s])
        k = rng.integers(*indications_per_disease)
        if len(pool):
            for d in rng.choice(pool, size=min(k, len(pool)), replace=False):
                if rng.random() < 0.8:
                    ind.add((int(d), s))
        # label noise: random indications
        if rng.random() < 0.3:
            ind.add((int(rng.integers(0, n_drug)), s))
        if rng.random() < 0.3:
            contra.add((int(rng.integers(0, n_drug)), s))
    for d, s in sorted(ind):
        add("indication", "indication", "drug", d, "disease", s)
    for d, s in sorted(contra - ind):
        add("contraindication", "contraindication", "drug", d, "disease", s)
    for d, s in list(sorted(ind))[:20]:  # a few off-label rows overlapping indications
        add("off-label use", "off-label use", "drug", d, "disease", s)

    df = pd.DataFrame(rows, columns=COLUMNS)
    # Inject cleaning cases: exact duplicates and malformed rows.
    dup = df.sample(n=50, random_state=seed)
    bad = df.sample(n=5, random_state=seed + 1).copy()
    bad["x_id"] = np.nan
    bad2 = df.sample(n=3, random_state=seed + 2).copy()
    bad2["y_type"] = "unknown_type"
    df = pd.concat([df, dup, bad, bad2], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
