# Report — Dataset, Data Engineering, Metrics, Results

Sections 1–4 of the final report. Related work, architecture, gate controls and
explanation analysis are added separately.

All numbers come from the pinned PrimeKG snapshot: `kg.csv` from doi:10.7910/DVN/IXA7BM
(Dataverse file 6180620), downloaded 2026-11-17 02:08 UTC, 981,751,236 bytes, SHA-256
`0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69`.

---

## 1. Dataset

**Source and licensing.** PrimeKG, static public release on Harvard Dataverse [1, 3].
`pcgate download` fetches `kg.csv` and writes `provenance.json` with the URL, download date,
SHA-256 and size. The PrimeKG repository's MIT licence covers software only; dataset use follows
the Dataverse terms and the licences of PrimeKG's upstream sources. No patient records, paid data
or the original HGTDR dataset are used.

**Snapshot statistics** (computed from the pinned file, not quoted from the paper):

| quantity | value |
|---|---|
| raw rows | 8,100,498 (each relationship stored in both directions) |
| invalid rows | 0 |
| unique canonical edges | 4,050,064 (published: 4,050,249) |
| canonical edge types | 30 (plus a `rev_*` copy each in the graph) |
| nodes | 129,312 — biological_process 28,642; gene_protein 27,610; disease 17,080; effect_phenotype 15,311; anatomy 14,033; molecular_function 11,169; drug 7,957; cellular_component 4,176; pathway 2,516; exposure 818 |
| unique indication pairs | 9,388 (1,801 drugs, 1,363 diseases) |
| contraindication / off-label edges | 30,675 / 2,568; overlap with indications 87 / 125 pairs; both excluded from the graph |
| training graph | ≈8.03 M directed edges, 103 node features |

**The 185-edge gap against the published count is accounted for.** PrimeKG stores each
relationship twice, so raw rows ÷ 2 gives 4,050,249 — the published figure. Of those, 4,049,879
relationships appear exactly twice and **185 appear four times**: the same `drug_protein` pair
recorded under two different `display_relation` sub-types, such as WARS2–Tryptophan as both
*enzyme* and *target*, or SLCO1B1–Fluticasone propionate as both *carrier* and *transporter*.
They are not duplicate rows. The graph is keyed on `relation`, so each collapses to one edge,
giving 4,050,249 − 185 = 4,050,064.

The cost is that for those 185 pairs (0.36 % of 51,306 `drug_protein` edges) the sub-type
distinction is dropped. No model reads `display_relation`, so nothing downstream changes, but it
is a modelling choice rather than a cleaning artefact. `preprocess` now computes this directly
and `dataset_report.md` reports it, so the figure is derived rather than asserted.

**Schema.** The ten released node types are kept as published; the combined `gene/protein` type is
retained and only sanitised to `gene_protein` for use as a key. No new node types or inferred
associations are created.

**Targets.** Positives are the documented `indication` drug→disease pairs, de-duplicated. All other
pairs are *unlabeled*, not negative. Contraindication and off-label-use pairs are never merged into
the labels; the audit reports their overlap with indications.

## 2. Data engineering

Every transformation is a script driven by a config; nothing is edited by hand. Configs record the
excluded relations, feature policy, candidate-pool policy, split seed and fractions.

| stage | implementation | output |
|---|---|---|
| Audit | columns, missing values, relation labels, `relation \| x_type -> y_type` endpoint table | `audit.json` |
| Identifier mapping | node key = (type, source, id), so ids unique only within a source stay distinct; stable sort gives type-specific indices | `nodes.csv` |
| Cleaning | rows with missing relation/id/type, unknown node type or self-loop are logged with a reason and dropped; each relationship is put in a canonical direction and de-duplicated | `edges.csv`, `invalid_rows.csv` |
| Target extraction | unique `indication` pairs | `indications.csv` |
| Split generation | random-pair 70/15/15 over unique pairs; drug- and disease-held-out split *entity groups* 70/15/15. Splits are made on unique pairs **before** any inverse edge exists | `splits/<setting>_s<seed>[_<pool policy>]/` |
| Graph construction | context relations both directions; indication edges for the training partition only, plus reverse copies; contraindication and off-label excluded as treatment-label proxies | in memory |
| Validation | six integrity checks; `run` refuses to train if any fails | `integrity_checks.json` |

**Integrity checks** (`pcgate/data/checks.py`):

1. no target pair appears in more than one partition;
2. no copy of a validation/test indication, in either direction, appears in any drug↔disease edge type, and the graph's indication edges are exactly the training pairs;
3. no training pair involves a held-out drug or disease;
4. features are reproducible from the training graph alone, and none derives from an excluded relation;
5. the `nodes.csv` hash matches the hash recorded at preprocessing, so all models share one entity mapping;
6. evaluation pools match their stored hashes, every pool positive is a held-out label, and no "unlabeled" pair is a known indication.

The checks run twice: in `prepare`, and again at the start of `run` against the configuration
actually being trained, because an ablation can change the feature policy after `prepare` wrote its
result. `run` records its own result, including the feature count, beside the results.
`tests/test_pipeline.py` injects a leaked test pair and confirms check 2 fails.

**Training-time controls.** Unlabeled pairs are made by corrupting one endpoint, never land on a
known indication from any partition, and never use a held-out drug or disease. While a training
indication is scored, it and its reverse copy are removed from message passing for that batch.

## 3. Metrics

Implemented in `pcgate/evaluate.py`; `tests/test_pipeline.py` checks them against hand-computed values.

**Candidate pools.** Set by `data.eligible_drugs` and recorded in every split manifest. The
**primary** policy is `with_indication`: the 1,801 drug nodes carrying at least one documented
indication. Section 4.2 gives the evidence for it; the all-drug policy is kept as a secondary
comparison.

* *Random-pair and disease-held-out:* for each test disease, the eligible drugs minus that disease's known indications from the other partitions.
* *Drug-held-out:* the held-out test drug group. Reported separately because the pool differs.

**Main metrics.** Average Precision — every held-out positive against 10 pre-sampled unlabeled
pairs (1:10), fixed on disk and identical for every model; and Recall@10 per disease,
macro-averaged so each evaluable disease counts equally.

**Secondary.** AUROC on the AP pairs, Precision@10, NDCG@10 (ideal DCG over min(|P|, 10)),
Recall@20, training time, peak memory, scoring throughput.

**Ties** are broken by a fixed-seed random permutation; otherwise a scorer with many ties
(popularity in the drug-held-out setting, where every test drug has zero training labels) would be
ranked by arbitrary drug index.

**Aggregation and uncertainty.** Per-seed values, mean ± std over seeds, paired per-seed
differences, and a 95% bootstrap CI on the Recall@10 difference resampling held-out diseases on the
fixed split. That CI does not capture split-to-split variance. Diseases without a held-out positive
are not evaluable; coverage is reported separately.

**Interpretation.** Every metric measures recovery of *documented* indications that were withheld.
An unlabeled pair ranked highly may be a real but undocumented treatment. Scores are ranking
signals, not calibrated probabilities of clinical efficacy.

## 4. Results

### 4.1 Splits

All six integrity checks pass in every setting.

| setting | train / val / test pairs | evaluable test diseases | candidate pool | test diseases with 0 training labels | median positives/disease |
|---|---|---|---|---|---|
| random-pair | 6,572 / 1,408 / 1,408 | 643 | 1,801 | 84 | 1 |
| drug-held-out | 6,523 / 1,462 / 1,403 | 640 | 270 (test drug group) | 67 | 1 |
| disease-held-out | 6,381 / 1,476 / 1,531 | 205 | 1,801 | 205 (by construction) | 3 |

### 4.2 Candidate-pool diagnostic

Measured by `pcgate pooldiag` on the fixed pools (`results/*/pool_diagnostic.md`). The
*eligibility oracle* scores 1 for any drug ever indicated and 0 otherwise; it never looks at the
disease, so whatever it reaches is attainable without doing the task. Lift is AP ÷ prevalence, so a
random scorer always scores 1.0 and the figure survives a change of base rate.

| setting | unlabeled pairs whose drug is never indicated | oracle AUROC | popularity AP lift |
|---|---|---|---|
| random-pair, all drugs | 77.8 % | 0.890 | 6.8× |
| random-pair, eligible | 0.0 % | 0.499 | 4.0× |
| disease-held-out, all drugs | 77.9 % | 0.889 | 7.0× |
| disease-held-out, eligible | 0.0 % | 0.494 | 4.0× |
| drug-held-out (unchanged) | 0.0 % | 0.458 | 1.1× |

With all 7,957 drug nodes in the pool, ~78 % of unlabeled pairs are trivially rejectable and the
oracle alone reaches AUROC 0.89 — against 0.93 for popularity and 0.90 for GraphSAGE. Restricting
to eligible drugs returns the oracle to chance. **Recall@10 is unaffected either way** (0.1344 →
0.1344 random, 0.0823 → 0.0823 disease-held-out), because ineligible drugs already sort to the
bottom; the oracle's own Recall@10 is 0.003. Ranking metrics measured the task all along; AP and
AUROC on the all-drug pool largely did not.

Eligibility is derived from the indication labels of the whole release — label-set-level
information, not the individual held-out pair — and applied identically to every model. Deriving
it from training labels alone is not viable: held-out drugs have no training indications by
construction, so the drug-held-out pool would be empty, and elsewhere a test positive whose drug
appears only in test would be removed from its own pool.

Both protocols remain reproducible: splits live in separate directories and each manifest records
its policy. The pair partition is identical across them (`pair_sha256` matches), so only the
evaluation pools differ; the drug-held-out pool is byte-identical, and its results match to six
decimal places.

### 4.3 Baselines under the primary protocol

GraphSAGE: mean ± std over 3 seeds, Colab T4. Popularity is deterministic (one run).

| setting | model | AP | AUROC | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| random-pair | popularity | **0.361** | **0.724** | 0.134 | 0.131 |
| random-pair | GraphSAGE | 0.291 ± 0.007 | 0.684 | 0.134 ± 0.002 | 0.129 |
| drug-held-out | popularity | 0.097 | 0.545 | 0.043 | 0.021 |
| drug-held-out | GraphSAGE | 0.110 ± 0.006 | 0.564 | 0.043 ± 0.007 | 0.022 |
| disease-held-out | popularity | **0.363** | **0.742** | **0.082** | **0.101** |
| disease-held-out | GraphSAGE | 0.174 ± 0.015 | 0.604 | 0.053 ± 0.014 | 0.068 |

**Paired differences and uncertainty.** Popularity is deterministic, so each GraphSAGE seed is
compared against the same reference run. The bootstrap CI resamples held-out diseases on the fixed
split (`results/*/summary.md`).

| setting | ΔAP per seed | mean ΔAP | sign | mean ΔRecall@10 | 95% CI on ΔRecall@10 |
|---|---|---|---|---|---|
| random-pair | −0.062 / −0.076 / −0.072 | −0.070 | all − | −0.0003 | [−0.0063, +0.0058] |
| drug-held-out | +0.008 / +0.012 / +0.019 | +0.013 | all + | +0.0006 | [−0.0106, +0.0117] |
| disease-held-out | −0.201 / −0.193 / −0.172 | −0.189 | all − | −0.0290 | **[−0.0459, −0.0127]** |

Reading these together changes one conclusion from the raw table above:

* **Disease-held-out: GraphSAGE is reliably worse than popularity.** The AP deficit holds on every seed and the Recall@10 CI excludes zero. This is the one difference that is both consistent and statistically distinguishable.
* **Drug-held-out: the apparent GraphSAGE win is not established.** AP is higher on all three seeds, but the Recall@10 CI spans zero ([−0.011, +0.012]), so on the ranking metric the two are indistinguishable given disease-level variability. It should be described as "no worse than popularity", not as a win.
* **Random-pair: a tie on ranking.** AP is consistently lower, but Recall@10 is effectively identical and its CI spans zero.

The CI covers disease-level variability on one fixed split; it does not capture split-to-split
variance, so it bounds less uncertainty than it may appear to.

Every run selected its checkpoint within the first 4–10 epochs; in disease-held-out, validation
Recall@10 peaks by epoch 4 and then falls towards 0 while training loss keeps dropping.

### 4.4 Ablation: indication-derived node features

Each node carries a count of its *training* indication edges — 0 for every held-out disease, large
for every training disease. The ablation removes those 4 columns (103 → 99 features), changes
nothing else, and tests whether they act as a shortcut.

| setting | metric | with | without |
|---|---|---|---|
| disease-held-out | Recall@10 | **0.053 ± 0.014** | 0.005 ± 0.000 |
| disease-held-out | NDCG@10 | **0.068** | 0.005 |
| disease-held-out | AP | **0.174** | 0.116 |
| random-pair | AP | 0.291 | **0.359** |
| random-pair | AUROC | 0.684 | **0.746** |
| random-pair | NDCG@10 | **0.129** | 0.098 |
| drug-held-out | AP | 0.110 | 0.114 |

**Keep the feature.** Removing it does not fix disease-held-out; it collapses it, with Recall@10
falling to 0.005 — negative on all three seeds (−0.050/−0.034/−0.062). The feature's dominant
content appears to be drug-side popularity rather than a disease-side shortcut, and that is the
strongest honest signal available when a disease has no treatment history. The random-pair effect
is mixed (AP and AUROC up, NDCG down) and changes sign across seeds in the other settings.

The disease-held-out weakness therefore remains unexplained and is not a feature-policy artefact.

### 4.5 Coverage and degree

| setting | test pairs | evaluable diseases | median disease degree | diseases with 0 training labels |
|---|---|---|---|---|
| random-pair | 1,408 | 643 | 32 | 84 |
| drug-held-out | 1,403 | 640 | 29 | 67 |
| disease-held-out | 1,531 | 205 | 17 | 205 |

Popularity Recall@10 by test-disease degree tertile: random 0.117 / 0.159 / 0.126; drug-held-out
0.030 / 0.013 / 0.085; disease-held-out 0.071 / 0.081 / 0.095. Performance tends to rise with
degree in the held-out settings — the uneven-coverage effect. The median test disease has a single
held-out positive in two of three settings, so per-disease recall is 0 or 1 and variance is high;
this is why results are macro-averaged over at least 205 diseases and reported with bootstrap CIs.

### 4.6 Efficiency

GraphSAGE, hidden 64, batch 1024, 6.77 M parameters (most of them context-node embeddings).

| device | step (s) | ≈ epoch (s) | peak memory | scoring (pairs/s) |
|---|---|---|---|---|
| CPU | 8.33 | 58 | 3.4 GB (process) | 0.90 M |
| T4, hidden 64 | 0.56 | 3.9 | 2.6 GB (CUDA) | 0.23 M |
| T4, hidden 128 | 0.72 | 5.1 | 5.1 GB (CUDA) | 0.18 M |

A full run took 52–112 s on the T4; nine runs took about 13 minutes. Hidden 128 roughly doubles GPU
memory and is not justified by current results. On the synthetic pilot the gated decoder cost ~10 %
more parameters and step time than HGT but ~5× lower scoring throughput, because every scored pair
gathers [R × H] relation summaries for both endpoints; that is the overhead to report for the gate.

### 4.7 Candidate table

`pcgate candidates` lists, per disease with a documented indication, the top-10 drugs that are not
known indications in any partition, with source identifiers, score and (for gated models) the
highest-weighted relation per endpoint. GraphSAGE's lists span 66 distinct drugs against 22 for
popularity, but both are dominated by corticosteroids — hydrocortisone acetate, cortisone acetate,
prednisolone, methylprednisolone and dexamethasone each appear for over 1,000 diseases. Its
highest-scoring pair is corticosteroids for hypertension, which is pharmacologically implausible,
since corticosteroids can raise blood pressure. These are computational hypotheses only.

## 5. Limitations

* **Unlabeled ≠ negative.** Some unlabeled pairs are real but undocumented treatments, so precision-style metrics understate performance.
* **Uneven coverage.** Well-studied entities have richer neighbourhoods; results are reported by degree tertile.
* **Pool definition.** Eligibility uses label-set-level information; it is documented, applied identically to all models, and the all-drug protocol is retained for comparison.
* **Explanations.** Paths show what a model relies on, not biological mechanism.
* **Clinical.** Predictions say nothing about efficacy, safety or regulatory suitability.

## References

[1] Chandak, Huang, Zitnik. Building a knowledge graph to enable precision medicine. *Scientific Data* 10, 67 (2023). https://doi.org/10.1038/s41597-023-01960-3
[3] PrimeKG, Harvard Dataverse. https://doi.org/10.7910/DVN/IXA7BM
