# Drug Repurposing Using PrimeKG

**A pair-conditioned relation-gating extension to HGT for drug repurposing**

Tests whether a lightweight *pair-conditioned* relation gate improves a Heterogeneous Graph
Transformer at predicting drug–disease treatment links on [PrimeKG](https://github.com/mims-harvard/PrimeKG).
The gate uses both the candidate drug and the candidate disease to weight relation-specific
neighbourhood summaries at prediction time.

Relation-aware attention and query-conditioned weighting are not new (see HGTDR, MRDGNN). The
contribution is a careful, leakage-controlled implementation and evaluation of one specific
extension against strong baselines and matched controls.

> **Scope.** Outputs are ranked *computational hypotheses*. Metrics measure recovery of withheld,
> documented indications — not therapeutic efficacy. Nothing here is biologically or clinically validated.

## Research questions

1. Does the pair-conditioned gate predict treatment links better than standard HGT?
2. Does any improvement hold when treatment labels for some drugs or diseases are withheld?
3. Does it come from pair-conditioned weighting, or from extra parameters / access to relation summaries?
4. Can graph paths provide evidence supporting individual predictions?

## Models

| model | role |
|---|---|
| `popularity` | treatment-frequency reference (training indications per drug) |
| `sage` | GraphSAGE — graph learning without relation-specific aggregation |
| `hgt` | standard HGT — main heterogeneous-attention baseline |
| `uniform` | control: same relation summaries, equal weights instead of a learned gate |
| `param_matched` | control: non-gated decoder with the gated decoder's parameter count |
| `gated` | **proposed** pair-conditioned relation gate |

For a candidate pair (d, s) with shared encoder embeddings h:

- summaries m<sub>v,r</sub> = mean over N<sub>r</sub>(v) of W<sub>r</sub> h<sub>u</sub>, for v ∈ {d, s}
- query q<sub>ds</sub> = MLP<sub>q</sub>([h<sub>d</sub> ‖ h<sub>s</sub> ‖ h<sub>d</sub> ⊙ h<sub>s</sub>])
- gate a<sub>r</sub> = wᵀ tanh(A q<sub>ds</sub> + B m<sub>v,r</sub> + C e<sub>r</sub>), α = softmax over relations available to v
- context c<sub>v</sub> = Σ<sub>r</sub> α<sub>r</sub> m<sub>v,r</sub>
- score p̂<sub>ds</sub> = σ(MLP<sub>o</sub>[h<sub>d</sub> ‖ h<sub>s</sub> ‖ h<sub>d</sub> ⊙ h<sub>s</sub> ‖ c<sub>d</sub> ‖ c<sub>s</sub>])

All learned models share the encoder framework, features, splits, pools and metric code; only the
decoder differs.

## Dataset

PrimeKG `kg.csv` from [doi:10.7910/DVN/IXA7BM](https://doi.org/10.7910/DVN/IXA7BM), not included
here — `pcgate download` fetches it and records the date, size and SHA-256 in `provenance.json`.
The PrimeKG repository's MIT licence covers software only; dataset use follows the Dataverse terms.
Positives are documented `indication` pairs; everything else is *unlabeled*, not negative.
Contraindication and off-label relations are never merged into labels and are kept out of the graph.

## Evaluation

Three settings — random-pair, drug-held-out, disease-held-out — each with fixed, hash-verified
candidate pools shared by every model.

The candidate pool is the **1,801 drug nodes carrying at least one documented indication**, not all
7,957 drug-type nodes. With every node in the pool, ~78 % of unlabeled pairs involve drugs that are
never indicated, and a scorer ignoring the disease entirely reaches AUROC 0.89; `pcgate pooldiag`
measures this. The all-drug policy is retained as a secondary comparison (`configs/primekg.yaml`).

Main metrics are AP (1:10 positive:unlabeled) and macro Recall@10; secondary are AUROC,
Precision@10, NDCG@10, Recall@20, runtime, memory and throughput. Three seeds per learned model,
with paired per-seed differences and bootstrap CIs over held-out diseases.

Six integrity checks — partition overlap, reverse-copy leakage, held-out entity labels, feature
leakage, entity-mapping consistency, fixed pools — must pass before training, and are re-verified
at the start of each run against the config actually in use.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch                 # first, matching your platform: pytorch.org/get-started
pip install -r requirements.txt
```

Python 3.11, PyTorch 2.x, PyTorch Geometric 2.8. On Colab the notebook installs only what
the runtime lacks — do not install `requirements.txt` there.

```bash
python -m pcgate download   --config configs/primekg_eligible.yaml   # fetch + pin kg.csv (~1 GB)
python -m pcgate prepare    --config configs/primekg_eligible.yaml   # clean, split, leakage checks
python -m pcgate run        --config configs/primekg_eligible.yaml   # train + evaluate
python -m pcgate summarize  --config configs/primekg_eligible.yaml   # tables, paired diffs, CIs
```

Other commands: `schema` (dataset/licence report), `pooldiag` (pool sanity), `coverage`
(degree analysis), `candidates` (ranked hypotheses), `explain` (path case study), `profile`
(runtime/memory), `synth` (synthetic graph for a CPU smoke test via `configs/synthetic.yaml`).

`run` is resumable — it skips any run that already has `metrics.json`; add `--force` to redo one.
`prepare` reuses preprocessing when the pinned file's SHA-256 is unchanged.

**GPU:** open `colab/run_graphsage_colab.ipynb` on a Colab T4. Regenerate it with
`python colab/build_notebook.py` — never edit it by hand.

**Tests:** `python -m pytest -q tests` — metrics against hand-computed values, split partitioning,
pools, and the leakage checks including a deliberately injected leak.

## Layout

```
pcgate/data/   preprocess.py  splits.py  graph.py  checks.py  synthetic.py
pcgate/        models.py  train.py  evaluate.py  report.py  analysis.py  explain.py  cli.py
configs/       primekg_eligible.yaml (primary)  primekg.yaml (all-drug)  *_colab*.yaml  synthetic.yaml
colab/         build_notebook.py  run_graphsage_colab.ipynb
docs/          report.md  proposal.pdf
tests/         test_pipeline.py
data/ results/ generated locally; only small result tables are committed
```

## Reproducibility

Every transformation is scripted; configs record schema choices, excluded relations, pool policy,
seeds and split fractions. Raw-file, entity-mapping and pool hashes are stored and checked at load
time. Hyperparameters are chosen on validation data only, and the test set is evaluated after the
configuration is frozen. All models share one snapshot, split manifest, graph, pools, metric code
and tuning budget.

## Limitations

Unlabeled pairs may be real but undocumented treatments. Well-studied entities have richer
neighbourhoods, so results are reported by degree. Paths show what a model relies on, not
mechanism. Predictions say nothing about efficacy, safety or regulatory suitability.

## References

1. Chandak, Huang, Zitnik. Building a knowledge graph to enable precision medicine. *Scientific Data* 10, 67 (2023). https://doi.org/10.1038/s41597-023-01960-3
2. PrimeKG repository. https://github.com/mims-harvard/PrimeKG
3. PrimeKG dataset, Harvard Dataverse. https://doi.org/10.7910/DVN/IXA7BM
4. Gharizadeh et al. HGTDR: Advancing drug repurposing with heterogeneous graph transformers. *Bioinformatics* 40(7), btae349 (2024). https://doi.org/10.1093/bioinformatics/btae349
5. HGTDR source. https://github.com/bcb-sut/HGTDR
6. Hamilton, Ying, Leskovec. Inductive representation learning on large graphs. NeurIPS (2017). https://arxiv.org/abs/1706.02216
7. Schlichtkrull et al. Modeling relational data with graph convolutional networks. ESWC (2018). https://arxiv.org/abs/1703.06103
8. Hu, Dong, Wang, Sun. Heterogeneous graph transformer. WWW (2020). https://arxiv.org/abs/2003.01332
9. Huang et al. A foundation model for clinician-centered drug repurposing. *Nature Medicine* 30 (2024). https://doi.org/10.1038/s41591-024-03233-x
10. Liao et al. MRDGNN: multi-relational reasoning for predicting drug indications. *Computational Biology and Chemistry* (2026).
