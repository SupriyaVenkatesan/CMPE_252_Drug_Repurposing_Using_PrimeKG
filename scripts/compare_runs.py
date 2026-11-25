"""Compare metrics.json files of two results directories (reproducibility check).

    python scripts/compare_runs.py results/primekg_eligible results/repro_member1 --setting disease

Deterministic models (popularity) must match exactly. Learned models are compared with a tolerance:
GPU kernels (scatter/atomic adds) are not bit-reproducible, so small differences are expected; the
check flags anything larger than --tol (absolute) on the main metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METRICS = ("ap", "recall@10", "auroc", "ndcg@10", "precision@10", "recall@20")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("reproduction")
    ap.add_argument("--setting", default=None)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args(argv)
    ref, rep = Path(a.reference), Path(a.reproduction)
    ok, n = True, 0
    for f in sorted(rep.glob("*/*/seed*/metrics.json")):
        rel = f.relative_to(rep)
        if a.setting and rel.parts[0] != a.setting:
            continue
        g = ref / rel
        if not g.exists():
            print(f"[missing in reference] {rel}")
            ok = False
            continue
        r, p = json.loads(g.read_text()), json.loads(f.read_text())
        tol = 0.0 if rel.parts[1] == "popularity" else a.tol
        n += 1
        for m in METRICS:
            d = p[m] - r[m]
            flag = "OK " if abs(d) <= tol + 1e-9 else "DIFF"
            ok &= flag == "OK "
            print(f"{flag} {str(rel.parent):32s} {m:13s} ref={r[m]:.4f} repro={p[m]:.4f} diff={d:+.4f} (tol {tol})")
    print(f"\n{n} runs compared: {'REPRODUCED' if ok and n else 'NOT REPRODUCED'}")
    sys.exit(0 if ok and n else 1)


if __name__ == "__main__":
    main()