# Member 2 workflow — HGT, gated HGT, controls, explanations

Everything Member 2 owns, in the order of the project deadlines. Commands are run from the repo
root. GPU work runs in `colab/run_hgt_colab.ipynb`; the same commands work on any CUDA machine
with the non-Colab configs (`configs/primekg_eligible_*.yaml`).

## What Member 2 added

| file | purpose | milestone |
|---|---|---|
| `pcgate/tune.py` | bounded tuning on validation data only; writes the frozen config | Nov 18 |
| `pcgate/gates.py` | gate statistics + evaluation-time gate controls | Nov 26 |
| `pcgate/utils.py` → `run_cfg` | per-model frozen hyperparameters (`model_overrides`); controls inherit `gated` | Nov 18 |
| `pcgate/train.py` | records the effective hyperparameters in every `metrics.json` | Nov 18 |
| `pcgate/cli.py` | `tune` and `gates` commands | — |
| `configs/primekg_eligible_tune.yaml`, `configs/primekg_colab_eligible_tune.yaml` | tuning grid (2 × 2 = 4 configs per model) | Nov 18 |
| `configs/primekg_eligible_frozen.yaml` | **written by `tune`**, committed after tuning | Nov 18 |
| `configs/synthetic_member2.yaml` | CPU smoke test of the whole Member 2 path | Nov 15 |
| `configs/repro_member1.yaml`, `scripts/compare_runs.py` | reproduce a Member 1 experiment | Dec 2 |
| `scripts/make_colab_bundle.py` | zip code + configs (+ processed data) for Colab | — |
| `colab/build_hgt_notebook.py` → `colab/run_hgt_colab.ipynb` | GPU notebook for all Member 2 runs | Nov 23–26 |
| `tests/test_member2_models.py` | 12 tests: every variant trains, gate = distribution over available relations, gate is pair-conditioned, uniform is not, param-matched within 5 %, override fallback, controls restore the model | Nov 15 |
| `docs/report_member2.md` | related work, architecture, controls, explanations, limitations | Nov 29 |

The model code itself (`HGTEncoder`, `RelationSummaries`, `GateDecoder`, `FlatDecoder`) is in
`pcgate/models.py` and is shared with Member 1's pipeline.

## 0. Environment (once)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch                                        # matching your CUDA, see pytorch.org
pip install -r requirements.txt
python -m pytest -q tests                                # expect 22 passed
python -c "import torch; print(torch.cuda.is_available())"
```

## 1. CPU smoke test of the Member 2 path (~6 min, synthetic data, not a result)

```bash
python -m pcgate synth   --config configs/synthetic.yaml
python -m pcgate prepare --config configs/synthetic.yaml
python -m pcgate tune    --config configs/synthetic_member2.yaml --freeze-to configs/synthetic_member2_frozen.yaml
python -m pcgate run     --config configs/synthetic_member2_frozen.yaml
python -m pcgate summarize --config configs/synthetic_member2_frozen.yaml
python -m pcgate gates   --config configs/synthetic_member2_frozen.yaml
python -m pcgate explain --config configs/synthetic_member2_frozen.yaml
```

## 2. Real runs on Colab (T4)

```bash
python colab/build_hgt_notebook.py        # regenerate the notebook (never edit it by hand)
python scripts/make_colab_bundle.py       # -> pcgate_colab_bundle.zip
```

Upload `pcgate_colab_bundle.zip` to *My Drive*, open `colab/run_hgt_colab.ipynb` in Colab with a
T4, and run the sections in order. If `data/processed/primekg` is not in your bundle, section 4
downloads PrimeKG and runs `prepare` (same seed → same splits; hashes are checked).

| notebook section | command | deadline |
|---|---|---|
| 6 tuning | `pcgate tune --config configs/primekg_colab_eligible_tune.yaml --only hgt gated --freeze-to …frozen.yaml` | Nov 18 |
| 7 main | `pcgate run --config …frozen.yaml --only hgt gated` (3 settings × 3 seeds) | Nov 23 |
| 8 controls | `pcgate run --config …frozen.yaml --only uniform param_matched` (disease) | Nov 26 |
| 9 summaries | `pcgate summarize`, `pcgate coverage` | Nov 26 |
| 10 gates | `pcgate gates --setting disease` (and `random`) | Nov 26 |
| 11 explanations | `pcgate explain --model gated`, `pcgate candidates --model gated` | Nov 26 |

Rules: select on validation only; commit `configs/primekg_eligible_frozen.yaml` **before** looking at
test numbers; never re-tune afterwards.

## 3. Bring results back into the repo

Unzip `member2_results.zip` into `results/primekg_eligible/`. The `.gitignore` keeps only the small
tables (`summary.md`, `all_runs.csv`, `metrics.json`, tuning `trials.csv`/`best.json`, `gates/`,
`explanations.json`); checkpoints and the 3 MB candidate CSVs stay out.

```bash
unzip -o member2_results.zip -d results/primekg_eligible
cp results/primekg_eligible/primekg_eligible_frozen.yaml configs/
git add configs/primekg_eligible_frozen.yaml results/primekg_eligible
git status        # check that no model.pt or candidates_*.csv is staged
```

Then fill the **[fill: …]** tables in `docs/report_member2.md` from those files.

## 4. Reproducibility check (Dec 2)

```bash
python -m pcgate run --config configs/repro_member1.yaml
python scripts/compare_runs.py results/primekg_eligible results/repro_member1 --setting disease
```

Popularity must match exactly; GraphSAGE within 0.02 (GPU kernels are not bit-reproducible).
