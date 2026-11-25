# Report — Related Work, Architecture, Gate Controls, Explanations, Limitations (Member 2)

Sections of the final report owned by Member 2. They complement `docs/report.md` (dataset, data
engineering, metrics, quantitative results). Numbers marked **[fill: file]** are copied from the
named result file after the frozen runs; nothing here should be filled from memory.

---

## Related work

**Knowledge graph.** PrimeKG integrates 20 biomedical resources into a heterogeneous graph of diseases,
drugs, genes/proteins, phenotypes, pathways, exposures and anatomy, with indication,
contraindication and off-label edges between drugs and diseases [1–3]. We use one pinned release
and treat its `indication` edges as the only positive labels.

**Heterogeneous graph learning.** R-GCN learns one transformation per relation type [7].
The Heterogeneous Graph Transformer (HGT) [8] goes further: keys, queries, values and messages are
parameterised by source type, target type and relation, so attention over neighbours is
type-aware. GraphSAGE [6] aggregates neighbours without regard to relation type and serves here
as the relation-agnostic baseline.

**Drug repurposing on knowledge graphs.** HGTDR [4] applies HGT to drug repurposing on a
heterogeneous biomedical graph and scores drug–disease pairs from the encoder's embeddings.
TxGNN [9] trains on PrimeKG and stresses zero-shot prediction for diseases with few or no known
treatments — the motivation for our disease-held-out setting. MRDGNN [10] uses relation-aware,
query-conditioned attention for indication prediction; query-conditioned relation weighting is
therefore not new, and we position the gate as an incremental, carefully controlled extension.

**What differs here.** Standard HGT fixes a node's representation before the candidate partner is
known. Our gate re-weights relation-specific neighbourhood summaries *at scoring time, conditioned
on the pair* — the same drug can emphasise its protein targets for one disease and its
drug–drug neighbours for another. The contribution is the leakage-controlled evaluation of that one
change against matched controls, not a new architecture family.

**HGTDR reuse.** Reviewed in week 1. The public HGTDR repository [5] contains a notebook and
utilities targeting torch 1.13 / PyG 2.3 and the authors' own dataset; it ships **no licence file**,
so no reuse permission is granted and no HGTDR code is included. We implemented the published HGT
architecture with PyTorch Geometric's `HGTConv` (PyG 2.8), with our own PrimeKG loader and schema
mapping, and cite HGTDR for the method.

## Architecture

All learned models share one encoder framework, feature set, graph, splits, pools and metric
code; **only the decoder differs** (`pcgate/models.py`).

### Shared HGT encoder

Each node type t has an input projection x ↦ W_t x (+ a learnable identity embedding for
*context* node types only — genes, pathways, phenotypes, …; drugs and diseases never get identity
embeddings, so held-out drugs and diseases are represented only through shared transformations).
L = 2 `HGTConv` layers with H heads follow, each wrapped as

  h_v^(l+1) = h_v^(l) + Dropout(GELU(HGTConv(h^(l))_v)),

giving embeddings h_d (drug) and h_s (disease) of width H. Training indication edges that are
being scored in a batch are removed from message passing together with their reverse copies;
contraindication and off-label edges are never in the graph.

### Relation-specific summaries

For an endpoint v ∈ {d, s} and each relation r whose edges point into v's type,

  m_{v,r} = W_r · mean_{u ∈ N_r(v)} h_u      (W_r linear, so this equals the mean of W_r h_u)

with an availability mask μ_{v,r} = 1[|N_r(v)| > 0]. Unavailable relations get m = 0 and are
excluded from the gate's softmax.

### Pair query and pair-conditioned gate

  q_ds = MLP_q([h_d ‖ h_s ‖ h_d ⊙ h_s])
  a_{v,r} = wᵀ tanh(A q_ds + B m_{v,r} + C e_r)           (e_r: learned relation embedding)
  α_{v,·} = softmax over {r : μ_{v,r} = 1} of a_{v,r}
  c_v = Σ_r α_{v,r} m_{v,r}

An endpoint with no available relation gets c_v = 0 and a missing-indicator flag.

### Prediction and objective

  p̂_ds = σ(MLP_o[h_d ‖ h_s ‖ h_d ⊙ h_s ‖ c_d ‖ c_s ‖ miss_d ‖ miss_s])

trained positive–unlabeled: binary cross-entropy with positives and sampled unlabeled pairs
averaged separately (5 unlabeled per positive, made by corrupting one endpoint, never using
held-out entities or any known indication), Adam with weight decay, early stopping on
validation AP.

### Model family

| model | encoder | decoder input | question it answers |
|---|---|---|---|
| `hgt` | HGT | [h_d, h_s, h_d⊙h_s] | baseline: heterogeneous attention, no gate |
| `gated` | HGT | + c_d, c_s from the **pair-conditioned** gate | proposed |
| `uniform` | HGT | + c_v with α = 1/\|R_v\| | do summaries help *without* learned weighting? |
| `param_matched` | HGT | + Linear(concat_r μ·m_{v,r}) , widened to the gated decoder's size | is it just extra capacity? |

Parameter counts **[fill: `results/primekg_eligible/summary.md`, params column]**. The
parameter-matched decoder is sized to within 5 % of the gated decoder (checked in
`tests/test_member2_models.py`).

### Tuning and freezing

Bounded search, identical for `hgt` and `gated` (proposal 9.2: at most four development
configurations per model): hidden ∈ {64, 128} × learning rate ∈ {1e-3, 5e-4}, dropout 0.2,
2 layers, 2 heads, on the disease-held-out **validation** pool, seed 0, selected by validation AP
(`pcgate tune`). The selected values were frozen in `configs/primekg_eligible_frozen.yaml` before
any test evaluation. The controls inherit the gated model's values, so they differ from it only
in the decoder.

Selected configurations **[fill: `results/primekg_eligible/tuning/best.json`]**:

| model | hidden | lr | best val AP | val R@10 |
|---|---|---|---|---|
| hgt | | | | |
| gated | | | | |

## Gate controls (E4)

Two kinds of control, both on the disease-held-out setting first (proposal 9.2).

**Trained controls** — separate models, three seeds each, same frozen hyperparameters as `gated`:
`uniform` and `param_matched` (table above). Paired per-seed differences and bootstrap CIs over
held-out diseases come from `pcgate summarize`.

**Evaluation-time controls** — applied to the *same* trained gated checkpoints, same pools, same
metric code (`pcgate gates`, `pcgate/gates.py`):

* `shuffled_query`: q_ds replaced by the query of a random *training* pair — weights no longer
  match the scored pair.
* `mean_query`: q_ds replaced by the mean training-pair query — weights then depend only on each
  endpoint's own summaries (node-conditioned, not pair-conditioned).
* `uniform_weights`: α forced to 1/|R_v| inside the trained model.

Queries come from a fixed bank of training pairs, so the controls do not depend on how the
evaluator batches pairs. If the gate's benefit comes from pair conditioning, `shuffled_query` and
`mean_query` must hurt.

**Is the gate actually pair-conditioned?** For each drug scored against ≥ 2 diseases we take the
standard deviation of its drug-side α across those diseases. This is exactly 0 for any node-only
weighting. We also report the normalised entropy H(α)/log|R_avail| (1 = uniform).

Results **[fill: `results/primekg_eligible/summary.md` (disease section) and
`results/primekg_eligible/gates/disease/gates.md`]**:

| comparison (disease-held-out) | mean ΔAP | ΔAP sign over seeds | mean ΔR@10 | 95 % CI ΔR@10 |
|---|---|---|---|---|
| gated − hgt | | | | |
| gated − uniform | | | | |
| gated − param_matched | | | | |

| eval-time control | ΔAP vs gated | ΔR@10 vs gated |
|---|---|---|
| shuffled_query | | |
| mean_query | | |
| uniform_weights | | |

Interpretation rule, fixed before filling in the numbers: the gate is supported only if it beats
`hgt` consistently across seeds in at least one setting **and** beats both trained controls, and
`shuffled_query` / `mean_query` reduce its score. If within-drug α variation is near zero (as on
our short synthetic pilot, ~1e-5) the gate is behaving as a near-uniform weighting and any gain
must be attributed to the summaries, not to pair conditioning. A negative result is an acceptable
outcome (proposal 11.5).

## Explanation case study (E5)

Scope: ten pairs from the disease-held-out gated model (seed 0): the four strongest recovered test
indications, the three highest-scoring unlabeled pairs, and three test indications with the
least graph evidence (lowest combined endpoint degree) (`pcgate explain`).

**Paths.** Only edges present in the training graph, never the target indication:
Drug → Gene/Protein → Disease and Drug → Gene/Protein → Pathway → Gene/Protein → Disease; up to
five per pair.

**Sensitivity.** For a path P, Δ_P = f(d, s | G) − f(d, s | G \ E_P), where E_P holds the
path's edges and their reverse copies. Each Δ_P is compared against `n_random` = 20 matched random
removals (same number of edges, same edge types, endpoints in the same log2-degree bucket where
possible); the empirical p-value is (1 + #{random Δ ≥ Δ_P}) / (1 + n_random), so its floor is 1/21 ≈ 0.048.

Each case also lists the gate's top relation per endpoint and its weight
(`candidates_gated_*.csv`, `explanations.json`).

Results **[fill: `results/primekg_eligible/explanations.json`]**:

| case type | drug → disease | score | paths found | best Δ_P | random Δ (mean ± sd) | p |
|---|---|---|---|---|---|---|
| | | | | | | |

How to read this: a path whose removal lowers the score much more than matched random removals is
evidence the *model* relies on it. It is not evidence of a biological mechanism, and the absence
of a path (common for limited-evidence pairs) does not mean the prediction is wrong.

## Limitations (model side)

* **Pair conditioning may be weak.** A gate that learns near-uniform or node-only weights adds cost
  without the intended mechanism; we measure this directly instead of assuming it.
* **Scoring cost.** Every scored pair gathers R × H summaries for both endpoints; the gated decoder
  scored pairs ~5× slower than HGT on the synthetic pilot **[fill: `profile_gpu_t4.csv`]**.
* **Bounded tuning.** Four configurations per model on one setting and one seed; a model may be
  under-tuned for the other settings. The budget is identical for all models.
* **Single split.** One split seed per setting; seed variation covers optimisation only, and the
  bootstrap covers held-out diseases only.
* **Evaluation-time controls** perturb a model trained with the real gate; they show reliance, not
  what a model trained without pair conditioning would learn (the trained `uniform` control covers
  that).
* **Explanations** are perturbation-based on ten pairs; they describe model reliance, not
  mechanism, and path availability is biased toward well-studied entities.
* **Clinical.** Outputs are ranked computational hypotheses with no claim of efficacy or safety.

## References

[1] Chandak, Huang, Zitnik. Building a knowledge graph to enable precision medicine. *Scientific Data* 10, 67 (2023).
[2] PrimeKG repository. https://github.com/mims-harvard/PrimeKG
[3] PrimeKG dataset, Harvard Dataverse. https://doi.org/10.7910/DVN/IXA7BM
[4] Gharizadeh et al. HGTDR: Advancing drug repurposing with heterogeneous graph transformers. *Bioinformatics* 40(7), btae349 (2024).
[5] HGTDR source. https://github.com/bcb-sut/HGTDR
[6] Hamilton, Ying, Leskovec. Inductive representation learning on large graphs. NeurIPS (2017).
[7] Schlichtkrull et al. Modeling relational data with graph convolutional networks. ESWC (2018).
[8] Hu, Dong, Wang, Sun. Heterogeneous graph transformer. WWW (2020).
[9] Huang et al. A foundation model for clinician-centered drug repurposing. *Nature Medicine* 30 (2024).
[10] Liao et al. MRDGNN: multi-relational reasoning for predicting drug indications. *Computational Biology and Chemistry* (2026).