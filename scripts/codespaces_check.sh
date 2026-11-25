#!/usr/bin/env bash
# Everything Member 2 runs on CPU (GitHub Codespaces): tests + a synthetic end-to-end smoke test of
# tune -> frozen config -> run -> summarize -> gates -> explain. ~6-8 min on 4 cores.
# Synthetic data is NOT a result; real experiments run on the Colab GPU (colab/run_hgt_colab.ipynb).
#   bash scripts/codespaces_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "import torch, torch_geometric" 2>/dev/null || {
  echo ">> installing CPU PyTorch + requirements"
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  pip install -q -r requirements.txt
}
python -c "import torch, torch_geometric as g; print('torch', torch.__version__, '| pyg', g.__version__)"

echo ">> 1/4 unit tests";                 python -m pytest -q tests -p no:cacheprovider
echo ">> 2/4 synthetic data";             python -m pcgate synth   --config configs/synthetic.yaml
                                          python -m pcgate prepare --config configs/synthetic.yaml
C=configs/synthetic_member2.yaml; F=configs/synthetic_member2_frozen.yaml
echo ">> 3/4 tune -> freeze -> run";      python -m pcgate tune --config $C --freeze-to $F
                                          python -m pcgate run  --config $F
                                          python -m pcgate summarize --config $F > /dev/null && echo '   wrote results/synthetic_member2/summary.md'
echo ">> 4/4 gates + explanations";       python -m pcgate gates   --config $F > /dev/null && echo '   wrote results/synthetic_member2/gates/disease/gates.md'
                                          python -m pcgate explain --config $F > /dev/null && echo '   wrote results/synthetic_member2/explanations.json'
python colab/build_hgt_notebook.py
echo
echo "ALL CPU CHECKS PASSED. Synthetic output is in results/synthetic_member2/ (git-ignored)."
echo "Next: git add -A && git commit && git push origin Sree, then open colab/run_hgt_colab.ipynb in Colab."