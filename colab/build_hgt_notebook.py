"""Generate colab/run_hgt_colab.ipynb (Member 2: HGT, gated HGT, controls, gates, explanations).

Built, never hand-edited - same convention as colab/build_notebook.py.

    python colab/build_hgt_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

DRIVE = "/content/drive/MyDrive/pcgate/results"
CFG = "configs/primekg_colab_eligible"          # primary protocol, results on Drive

CELLS: list[tuple[str, str]] = [
("md", """# HGT and pair-conditioned gated HGT on PrimeKG — Colab T4 (Member 2)

**Before running:** push your branch from Codespaces, then here: Runtime → Change runtime type →
**T4 GPU**, and run top to bottom. Cell 2 clones the branch from GitHub (no upload needed).
PrimeKG is downloaded and prepared once; the processed data is cached on Drive for later sessions.

Everything is written to `My Drive/pcgate/results/primekg_eligible/`, next to Member 1's GraphSAGE and
popularity results, so `summarize` produces the paired comparisons for every model. Every step is
resumable: after a disconnect re-run cells 1–4, then continue where you stopped.

| section | what it does | T4 time (rough) |
|---|---|---|
| 1–4 | GPU, Drive, PyG, data + tests | 2–15 min |
| 5 | memory / runtime profile of hgt and gated | 3 min |
| 6 | **bounded tuning** on validation data → frozen config | 25–50 min |
| 7 | **main experiments**: hgt + gated × 3 settings × 3 seeds | 1–2 h |
| 8 | **trained controls**: uniform + param_matched on disease-held-out | 20–40 min |
| 9 | summaries, paired differences, coverage | 2 min |
| 10 | **gate analysis** + evaluation-time gate controls | 5 min |
| 11 | explanation case study + candidate table (gated) | 10–20 min |
| 12 | download results and the frozen configs | 1 min |"""),

("md", "## 1. Check the GPU"),
("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),

("md", """## 2. Mount Drive and get the code from GitHub

Set `BRANCH` to the branch you pushed from Codespaces. Re-run this after every new push.
(Offline alternative: upload `pcgate_colab_bundle.zip` to My Drive and set `USE_BUNDLE = True`.)"""),
("code", """BRANCH = 'Sree'
REPO = 'https://github.com/SupriyaVenkatesan/CMPE_252_Drug_Repurposing_Using_PrimeKG.git'
USE_BUNDLE = False
from google.colab import drive
drive.mount('/content/drive')
!rm -rf /content/pcgate
if USE_BUNDLE:
    !mkdir -p /content/pcgate && unzip -q /content/drive/MyDrive/pcgate_colab_bundle.zip -d /content/pcgate
else:
    !git clone -q --depth 1 -b {BRANCH} {REPO} /content/pcgate
%cd /content/pcgate
!git log --oneline -1 2>/dev/null; ls"""),

("md", "## 3. Install PyTorch Geometric (Colab already has torch, pandas, scikit-learn, psutil)"),
("code", """!pip install -q torch_geometric pyyaml
import torch, torch_geometric
print(torch.__version__, torch_geometric.__version__, 'cuda:', torch.cuda.is_available())"""),

("md", """## 4. Data, tests, and Member 1's committed results

First session: downloads PrimeKG (~1 GB) and runs `prepare` (same seed → same splits as Member 1,
hashes checked), then caches `data/processed` on Drive. Later sessions restore it from Drive."""),
("code", """import os
CACHE = '/content/drive/MyDrive/pcgate/data_processed_primekg'
if os.path.exists(CACHE + '/splits/disease_s0_with_indication/manifest.json'):
    !mkdir -p data/processed && cp -r {CACHE} data/processed/primekg && echo restored processed data from Drive
else:
    !python -m pcgate download --config CFG.yaml
    !python -m pcgate prepare  --config CFG.yaml
    !mkdir -p /content/drive/MyDrive/pcgate && cp -r data/processed/primekg {CACHE} && echo cached processed data on Drive
!python -m pytest -q tests -p no:cacheprovider
!python -c "from pcgate.utils import read_json; [print(s, read_json(f'data/processed/primekg/splits/{s}_s0_with_indication/integrity_checks.json')['all_passed']) for s in ('random','drug','disease')]"
# Member 1's sage/popularity metrics travel in the repo; copy without overwriting anything on Drive
!mkdir -p DRIVE_RESULTS/primekg_eligible && cp -rn results/primekg_eligible/* DRIVE_RESULTS/primekg_eligible/ 2>/dev/null; echo copied"""),

("md", "## 5. Memory / runtime profile (W3) for the HGT family"),
("code", "!python -m pcgate profile --config CFG.yaml --models hgt gated"),

("md", """## 6. Bounded tuning — validation data only (milestone: Nov 18)

Same 4 configurations for `hgt` and `gated`, on the disease-held-out validation pool, seed 0. The test
pool is never scored. `uniform` / `param_matched` inherit the gated values automatically.

Trials are saved to `tuning/trials.csv` on Drive after each run, so if the runtime disconnects,
re-running this cell skips finished trials and still rewrites the frozen config at the end.

**Commit the printed frozen config** (`configs/primekg_eligible_frozen.yaml`, downloaded in section 12)
— that is the "configurations frozen" checkpoint. Do not re-tune after looking at test results."""),
("code", """!python -m pcgate tune --config CFG_tune.yaml --only hgt gated --freeze-to CFG_frozen.yaml
# local twin of the frozen config (same overrides, repo paths) for committing to GitHub
import yaml, shutil
fz = yaml.safe_load(open('CFG_frozen.yaml'))
with open('configs/primekg_eligible_frozen.yaml', 'w') as f:
    f.write('# FROZEN configuration - written by `pcgate tune` (Colab) from validation data only.\\n')
    yaml.safe_dump({'base': 'primekg_eligible_tune.yaml', 'model_overrides': fz['model_overrides']}, f, sort_keys=False)
for p in ('CFG_frozen.yaml', 'configs/primekg_eligible_frozen.yaml'):
    shutil.copy(p, 'DRIVE_RESULTS/primekg_eligible/tuning/')
print(open('configs/primekg_eligible_frozen.yaml').read())"""),

("md", """### If the runtime restarted after tuning

`/content` is wiped on restart; the frozen configs are also on Drive. Run this before sections 7–11."""),
("code", """import os, shutil
src = 'DRIVE_RESULTS/primekg_eligible/tuning/'
for f in ('primekg_colab_eligible_frozen.yaml', 'primekg_eligible_frozen.yaml'):
    if os.path.exists(src + f): shutil.copy(src + f, 'configs/' + f); print('restored', f)"""),

("md", """## 7. Main experiments — HGT and gated HGT, all three settings, 3 seeds (milestone: Nov 23)

Checkpoints (`model.pt`) are saved next to each run's `metrics.json` on Drive."""),
("code", "!python -m pcgate run --config CFG_frozen.yaml --only hgt gated"),

("md", """## 8. Trained gate controls on disease-held-out (milestone: Nov 26)

`uniform`: same relation summaries, equal weights. `param_matched`: same summaries, non-gated decoder
with the gated decoder's parameter count. Both use the gated model's frozen hyperparameters."""),
("code", "!python -m pcgate run --config CFG_frozen.yaml --only uniform param_matched"),

("md", "## 9. Summaries, paired differences, bootstrap CIs, coverage by degree"),
("code", """!python -m pcgate summarize --config CFG_frozen.yaml
!python -m pcgate coverage  --config CFG_frozen.yaml"""),

("md", """## 10. Gate analysis — is the gain from pair-conditioned weighting? (RQ3)

Evaluation-time controls on the **same** gated checkpoints: `shuffled_query` (query from an unrelated
training pair), `mean_query` (one constant query → node-only weighting), `uniform_weights`.
Also reports relation weights, entropy, and how much a drug's weights change with the disease."""),
("code", """!python -m pcgate gates --config CFG_frozen.yaml --setting disease
!python -m pcgate gates --config CFG_frozen.yaml --setting random"""),

("md", """## 11. Limited explanation case study (E5) and candidate table

Path retrieval + perturbation against matched random removals, for ~10 pairs. Paths show what the
model relies on, not mechanism. Candidates are computational hypotheses, not validated treatments."""),
("code", """!python -m pcgate explain    --config CFG_frozen.yaml --model gated
!python -m pcgate candidates --config CFG_frozen.yaml --model gated | head -20"""),

("md", "## 12. Download results (no checkpoints) and the frozen configs"),
("code", """!cp configs/primekg_eligible_frozen.yaml CFG_frozen.yaml DRIVE_RESULTS/primekg_eligible/
!cd DRIVE_RESULTS/primekg_eligible && zip -qr /content/member2_results.zip . -x '*.pt'
from google.colab import files
files.download('/content/member2_results.zip')"""),
]


def build(path: str | Path = None) -> Path:
    path = Path(path or Path(__file__).with_name("run_hgt_colab.ipynb"))
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        src = src.replace("DRIVE_RESULTS", DRIVE).replace("CFG", CFG)
        c = {"cell_type": "markdown" if kind == "md" else "code",
             "id": f"cell{i:02d}", "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"wrote {build()}")