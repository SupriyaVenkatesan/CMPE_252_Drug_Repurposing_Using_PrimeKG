# Drug Repurposing Using PrimeKG

**A pair-conditioned relation-gating extension to HGT for drug repurposing**

This project tests whether a lightweight, *pair-conditioned* relation gate improves a Heterogeneous Graph
Transformer (HGT) at predicting drug–disease treatment links on the [PrimeKG](https://github.com/mims-harvard/PrimeKG)
biomedical knowledge graph. The gate looks at both the candidate drug and the candidate disease, and uses them to
weight relation-specific neighbourhood summaries at prediction time.

We do not claim that relation-aware attention or query-conditioned weighting is new; HGTDR and MRDGNN already
explore these ideas. Our contribution is a careful, leakage-controlled implementation and evaluation of one
specific extension against strong baselines and matched controls.

> **Scope.** Outputs are ranked *computational hypotheses*. Metrics measure recovery of withheld, documented
> indications. They do not measure therapeutic efficacy, and nothing here is biologically or clinically validated.

---

## Research questions

1. Does the pair-conditioned gate predict treatment links better than standard HGT?
2. Does any improvement hold when treatment labels for some drugs or diseases are withheld?
3. Does any improvement come from pair-conditioned weighting, or just from extra parameters or access to relation summaries?
4. Can graph paths provide evidence that supports individual predictions?

## Approach

| Model | Role |
|---|---|
| Popularity | Treatment-frequency reference (number of training indications per drug) |
| GraphSAGE | Graph learning without relation-specific aggregation |
| Standard HGT | Main heterogeneous-attention baseline |
| Uniform-summary HGT | Control: same relation summaries, equal weights instead of a learned gate |
| Parameter-matched HGT | Control: non-gated decoder with the gated model's parameter count and the same summaries |
| **Gated HGT (proposed)** | Pair-conditioned relation gate on top of the shared HGT encoder |

**Gate.** For a candidate pair (d, s), with shared HGT embeddings h:

- relation summaries: m<sub>v,r</sub> = mean over N<sub>r</sub>(v) of W<sub>r</sub> h<sub>u</sub>, for v ∈ {d, s}
- pair query: q<sub>ds</sub> = MLP<sub>q</sub>([h<sub>d</sub> ‖ h<sub>s</sub> ‖ h<sub>d</sub> ⊙ h<sub>s</sub>])
- gate: a<sub>r</sub> = wᵀ tanh(A q<sub>ds</sub> + B m<sub>v,r</sub> + C e<sub>r</sub>), with α = softmax over the relations available to v
- context: c<sub>v</sub> = Σ<sub>r</sub> α<sub>r</sub> m<sub>v,r</sub>
- score: p̂<sub>ds</sub> = σ(MLP<sub>o</sub>[h<sub>d</sub> ‖ h<sub>s</sub> ‖ h<sub>d</sub> ⊙ h<sub>s</sub> ‖ c<sub>d</sub> ‖ c<sub>s</sub>])

All learned models share the encoder framework, input features, splits, candidate pools and metric code. Only
the decoder differs.

## Dataset

- **Source:** PrimeKG `kg.csv`, Harvard Dataverse, [doi:10.7910/DVN/IXA7BM](https://doi.org/10.7910/DVN/IXA7BM).
  Not included in this repository; `pcgate download` fetches it.
- **Pinned snapshot:** the download date, size and SHA-256 are recorded in `data/raw/primekg/provenance.json`.
  All experiment statistics are computed from the pinned snapshot.
- **Licensing:** the PrimeKG repository's MIT licence covers software only. Dataset use follows the Dataverse
  terms and the licences of PrimeKG's upstream sources.
- **Labels:** documented `indication` drug→disease pairs are positives, and all other pairs are *unlabeled*
  (positive–unlabeled setting). Contraindication and off-label-use relations are never merged into labels and
  are kept out of the graph, because they would act as treatment-label proxies.

## Data engineering

Every step is a script driven by a YAML config; nothing is edited by hand.

| Stage | Output |
|---|---|
| Audit: columns, missing ids, relation labels, endpoint types | `audit.json`, `dataset_report.md` |
| Identifier mapping: stable type-specific indices, source ids kept | `nodes.csv` |
| Cleaning: invalid rows logged; the two stored directions of each edge collapsed to one canonical edge | `edges.csv`, `invalid_rows.csv` |
| Target extraction: unique indication pairs | `indications.csv` |
| Splits: random-pair, drug-held-out and disease-held-out (70/15/15), made before any inverse edges exist | `splits/<setting>_s<seed>/` |
| Fixed evaluation pools: hash-checked every time they are loaded | in `split.npz` |
| Graph construction: context relations in both directions, training indications only | in-memory heterogeneous graph |
| Integrity checks: must pass before any experiment runs | `integrity_checks.json` |

**Leakage checks.**
1. No target pair appears in more than one partition.
2. No copy of a held-out indication, in either direction, appears in the graph.
3. No training labels exist for held-out drugs or diseases.
4. Features are computed from the training graph only.
5. Entity mappings are consistent across models.
6. Evaluation pools are fixed and identical for every model.

During training, each batch's own indication edges (and their reverse copies) are also removed from message passing.

## Evaluation

| Setting | Candidate pool |
|---|---|
| Random-pair | All eligible drugs for each test disease, minus that disease's known indications from other partitions |
| Drug-held-out | The held-out test drug group |
| Disease-held-out | All eligible drugs for each test disease, minus known indications from other partitions |

- **Main metrics:** Average Precision (each positive against 10 pre-sampled unlabeled pairs, the same pool for
  every model) and Recall@10, macro-averaged over test diseases.
- **Secondary metrics:** AUROC, Precision@10, NDCG@10, Recall@20, training time, peak memory and scoring throughput.
- **Uncertainty:** three seeds per learned model; per-seed paired differences; bootstrap CIs over held-out diseases.
- **Coverage:** evaluable entities and degree-stratified recall, so the effect of uneven graph coverage stays visible.
- **Explanations:** Drug → Gene/Protein → Disease and Drug → Gene/Protein → Pathway → Gene/Protein → Disease paths,
  scored by the change in prediction when the path's edges are removed, compared with matched random removals.

## Getting started

### Install

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch                 # or the CUDA build for your GPU
pip install torch_geometric pandas numpy scikit-learn pyyaml pytest psutil
```

Tested with Python 3.11, PyTorch 2.x and PyTorch Geometric 2.8.

### Run the full pipeline

```bash
python -m pcgate download   --config configs/primekg.yaml   # fetch + pin kg.csv (~1 GB)
python -m pcgate prepare    --config configs/primekg.yaml   # audit, clean, split, build graph, leakage checks
python -m pcgate schema     --config configs/primekg.yaml   # dataset / schema / usage-terms report
python -m pcgate run        --config configs/primekg.yaml   # train + evaluate all settings, models, seeds
python -m pcgate summarize  --config configs/primekg.yaml   # tables, paired differences, bootstrap CIs
python -m pcgate coverage   --config configs/primekg.yaml   # coverage and degree analysis
python -m pcgate candidates --config configs/primekg.yaml   # ranked candidate table
python -m pcgate explain    --config configs/primekg.yaml   # path explanation case study
python -m pcgate profile    --config configs/primekg.yaml   # memory / runtime study
```

`run` is resumable: it skips any run that already has `metrics.json`. Use `--only <models>` to run a subset,
e.g. `--only popularity sage`.

### Quick smoke test (CPU, a few minutes)

A small synthetic graph in PrimeKG's `kg.csv` format exercises the whole pipeline. Its numbers are not results.

```bash
python -m pcgate synth   --config configs/synthetic.yaml
python -m pcgate prepare --config configs/synthetic.yaml
python -m pcgate run     --config configs/synthetic.yaml
```

### Google Colab (GPU)

Open `colab/run_graphsage_colab.ipynb` with a T4 GPU runtime. The `configs/primekg_colab.yaml` config writes
results to Google Drive, so finished runs survive a disconnect.

### Tests

```bash
python -m pytest -q tests
```

The tests cover metrics against hand-computed values, split partitioning, evaluation pools, and the leakage
checks, including a deliberately injected leak that the checks must catch.

## Repository layout

```
pcgate/
  data/
    preprocess.py   download, audit, id mapping, cleaning, target extraction
    splits.py       random / drug / disease splits and fixed evaluation pools
    graph.py        heterogeneous graph construction and structural features
    checks.py       integrity and leakage checks
    synthetic.py    synthetic PrimeKG-format generator (smoke tests only)
  models.py         GraphSAGE, HGT, relation summaries, gated / uniform / parameter-matched decoders, popularity
  train.py          positive-unlabeled training with per-batch edge masking and early stopping
  evaluate.py       AP, AUROC, Recall@K, Precision@10, NDCG@10 on fixed pools
  report.py         seed summaries, paired differences, bootstrap CIs
  analysis.py       schema report, profiling, coverage analysis
  explain.py        path retrieval and perturbation sensitivity
  cli.py            command-line entry point (python -m pcgate ...)
configs/            primekg.yaml, primekg_colab.yaml, synthetic.yaml
colab/              Colab notebook
tests/              test suite
docs/               report drafts
data/, results/     generated locally (not committed)
```

## Reproducibility

- Every transformation is scripted; configs record schema choices, excluded relations, seeds and split fractions.
- Raw-file, entity-mapping and evaluation-pool hashes are stored and checked at load time.
- Hyperparameters are selected on validation data only; the test set is evaluated after the configuration is frozen.
- Every model is compared on the same snapshot, split manifests, graph edges, candidate pools, metric code and tuning budget.

## Limitations

- **Unlabeled ≠ negative:** some unlabeled pairs are real but undocumented treatments.
- **Uneven coverage:** well-studied entities have richer neighbourhoods; we report results by degree.
- **Explanations:** paths show what the model relies on, not proven biological mechanisms.
- **Clinical:** predictions say nothing about efficacy, safety or regulatory suitability.

## References

1. Chandak, P., Huang, K., Zitnik, M. *Building a knowledge graph to enable precision medicine.* Scientific Data 10, 67 (2023). https://doi.org/10.1038/s41597-023-01960-3
2. PrimeKG repository. https://github.com/mims-harvard/PrimeKG
3. PrimeKG dataset, Harvard Dataverse. https://doi.org/10.7910/DVN/IXA7BM
4. Gharizadeh, A., et al. *HGTDR: Advancing drug repurposing with heterogeneous graph transformers.* Bioinformatics 40(7), btae349 (2024). https://doi.org/10.1093/bioinformatics/btae349
5. HGTDR source code. https://github.com/bcb-sut/HGTDR
6. Hamilton, W. L., Ying, R., Leskovec, J. *Inductive representation learning on large graphs.* NeurIPS (2017). https://arxiv.org/abs/1706.02216
7. Schlichtkrull, M., et al. *Modeling relational data with graph convolutional networks.* ESWC (2018). https://arxiv.org/abs/1703.06103
8. Hu, Z., Dong, Y., Wang, K., Sun, Y. *Heterogeneous graph transformer.* WWW (2020). https://arxiv.org/abs/2003.01332
9. Huang, K., et al. *A foundation model for clinician-centered drug repurposing.* Nature Medicine 30 (2024). https://doi.org/10.1038/s41591-024-03233-x
10. Liao, Y., et al. *MRDGNN: A multi-relational reasoning framework for predicting drug indications via relational digraphs.* Computational Biology and Chemistry (2026).
