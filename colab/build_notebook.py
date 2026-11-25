"""Generate colab/run_graphsage_colab.ipynb from scratch.

The notebook is built, never hand-patched: an earlier edit filtered cells by a marker that
appeared only in the markdown headings, so stale *code* cells survived and were run by mistake.
Regenerating the whole file makes that class of error impossible.

    python colab/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

DRIVE = "/content/drive/MyDrive/pcgate/results"

CELLS: list[tuple[str, str]] = [
("md", """# GraphSAGE on PrimeKG — Colab T4

**Before running:** Runtime → Change runtime type → **T4 GPU**, and upload `pcgate_colab_bundle.zip`
to the top level of *My Drive*.

Results are written to `My Drive/pcgate/results/`. If the runtime disconnects, re-run cells 1–4 and
then continue; finished runs are skipped automatically, so nothing is duplicated or recomputed.

| section | what it does | time |
|---|---|---|
| 1–4 | set up, verify the pinned data | ~2 min |
| 5–8 | GraphSAGE on the **all-drug** pool (secondary protocol) | ~15 min |
| 9 | GraphSAGE on the **eligible-drug** pool (**primary** protocol) | ~15 min |
| 10 | ablation: no indication-derived features | ~13 min |"""),

("md", "## 1. Check the GPU"),
("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),

("md", "## 2. Mount Drive and unpack the bundle\n\nRe-run this after uploading a new bundle, otherwise the old code stays in `/content`."),
("code", """from google.colab import drive
drive.mount('/content/drive')
!rm -rf /content/pcgate && mkdir -p /content/pcgate
!unzip -q /content/drive/MyDrive/pcgate_colab_bundle.zip -d /content/pcgate
%cd /content/pcgate
!ls"""),

("md", "## 3. Install PyTorch Geometric\n\nColab already ships torch, pandas, scikit-learn and psutil."),
("code", """!pip install -q torch_geometric pyyaml
import torch, torch_geometric
print(torch.__version__, torch_geometric.__version__, 'cuda:', torch.cuda.is_available())"""),

("md", """## 4. Verify the pinned data

The tests cover metrics, splits, pools and the leakage checks. Loading a split re-verifies its
stored pool hashes, and `run` re-checks integrity against the config actually being trained."""),
("code", """!python -m pytest -q tests -p no:cacheprovider
!python -c "from pcgate.utils import read_json; [print(s, read_json(f'data/processed/primekg/splits/{s}_s0/integrity_checks.json')['all_passed']) for s in ('random','drug','disease')]"
!mkdir -p DRIVE_RESULTS/primekg
!cp -rn results/primekg/* DRIVE_RESULTS/primekg/ 2>/dev/null; echo done"""),

("md", """## 5. Secondary protocol — GraphSAGE on the all-drug pool

3 settings × 3 seeds. Already-finished runs are skipped, so re-running this costs nothing."""),
("code", "!python -m pcgate run --config configs/primekg_colab.yaml --only sage"),

("md", "## 6. Memory / runtime profile on the GPU"),
("code", "!python -m pcgate profile --config configs/primekg_colab.yaml --models sage"),

("md", "## 7. Summaries, coverage, candidate table"),
("code", """!python -m pcgate summarize  --config configs/primekg_colab.yaml
!python -m pcgate coverage   --config configs/primekg_colab.yaml
!python -m pcgate candidates --config configs/primekg_colab.yaml --model sage | head -20"""),

("md", """## 8. PRIMARY protocol — eligible-drug pool

Sections 5–7 ranked against all 7,957 drug-type nodes. Only 1,801 are ever indicated for anything,
so ~78 % of the unlabeled pairs were trivially rejectable and an *eligibility oracle* that ignores
the disease entirely reached **AUROC 0.89**. AP and AUROC were largely measuring that; Recall@10,
Precision@10 and NDCG@10 were never affected.

The pools are now the 1,801 eligible drugs. **The train/val/test pairs are identical** (verified by
hash), so this is purely an evaluation change — but validation pools changed too, so early stopping
may land on different checkpoints. Splits for both protocols ship in the bundle; no `prepare` needed.

Expect popularity to drop to AP ≈ 0.36 / AUROC ≈ 0.72, and the oracle to fall back to chance."""),
("code", """!python -m pcgate run       --config configs/primekg_colab_eligible.yaml --only popularity sage
!python -m pcgate pooldiag  --config configs/primekg_colab_eligible.yaml
!python -m pcgate summarize --config configs/primekg_colab_eligible.yaml
!python -m pcgate coverage  --config configs/primekg_colab_eligible.yaml"""),

("md", "### Pool comparison: all drugs vs eligible drugs, same models"),
("code", """import pandas as pd
old = pd.read_csv('DRIVE_RESULTS/primekg/all_runs.csv').assign(pool='all drugs (7957)')
new = pd.read_csv('DRIVE_RESULTS/primekg_eligible/all_runs.csv').assign(pool='eligible (1801)')
cmp = pd.concat([old, new]).groupby(['setting', 'model', 'pool'])[['ap', 'auroc', 'recall@10', 'ndcg@10']].mean()
print(cmp.round(4).to_string())
print('\\nRanking metrics should be ~unchanged; AP and AUROC should fall.')"""),

("md", """## 9. Ablation — no indication-derived node features

Every node carries a count of its *training* indication edges. That column is 0 for each held-out
disease and large for each training disease, so a model can learn "this disease already has known
treatments" instead of learning from biology — a plausible cause of the disease-held-out collapse
(GraphSAGE Recall@10 0.053 vs popularity 0.082; 0.016 on the lowest-degree tertile).

The feature is leakage-free (training edges only) and all six integrity checks pass either way, so
this is a design question, not a bug. The ablation runs under the **primary** protocol with the
feature off (103 → 99 features) and changes nothing else.

The integrity line below should read **99 features, 0 indication-derived**. Run section 8 first —
the comparison reads its results."""),
("code", """!python -m pcgate run       --config configs/primekg_colab_noindfeat.yaml --only sage
!python -m pcgate summarize --config configs/primekg_colab_noindfeat.yaml
!python -m pcgate coverage  --config configs/primekg_colab_noindfeat.yaml"""),

("md", "### Feature comparison: with vs without the indication features"),
("code", """import pandas as pd
with_f = pd.read_csv('DRIVE_RESULTS/primekg_eligible/all_runs.csv').assign(features='with (103)')
without = pd.read_csv('DRIVE_RESULTS/primekg_eligible_noindfeat/all_runs.csv').assign(features='without (99)')
cmp = pd.concat([with_f, without])
cmp = cmp[cmp.model == 'sage'].groupby(['setting', 'features'])[['ap', 'recall@10', 'auroc']].agg(['mean', 'std'])
print(cmp.round(4).to_string())"""),

("md", "## 10. Download every result"),
("code", """!cd DRIVE_RESULTS && zip -qr /content/primekg_results.zip . -x '*.pt'
from google.colab import files
files.download('/content/primekg_results.zip')"""),
]


def build(path: str | Path = None) -> Path:
    path = Path(path or Path(__file__).with_name("run_graphsage_colab.ipynb"))
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        src = src.replace("DRIVE_RESULTS", DRIVE)
        c = {"cell_type": "markdown" if kind == "md" else "code",
             "id": f"cell{i:02d}", "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": cells}
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    p = build()
    print(f"wrote {p} ({len(CELLS)} cells)")
