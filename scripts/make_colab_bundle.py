"""Build pcgate_colab_bundle.zip for the Colab notebooks.

    python scripts/make_colab_bundle.py            # code + configs + tests + committed results
                                                   # + data/processed/primekg if it exists locally

Upload the zip to the top level of Google Drive ("My Drive"). The raw kg.csv (~1 GB) is never
bundled; without data/processed the notebook downloads PrimeKG and runs `prepare` itself.
"""
from __future__ import annotations

import datetime as dt
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["pcgate", "configs", "tests", "colab", "results", "requirements.txt", "README.md", "LICENSE"]
OPTIONAL = ["data/processed/primekg", "data/raw/primekg/provenance.json"]
SKIP_SUFFIX = {".pt", ".pyc", ".zip", ".ipynb"}


def main(out: str = "pcgate_colab_bundle.zip"):
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        rev, branch = "unknown", "unknown"
    n = 0
    with zipfile.ZipFile(ROOT / out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE + [o for o in OPTIONAL if (ROOT / o).exists()]:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                if f.suffix in SKIP_SUFFIX or "__pycache__" in f.parts or "synthetic" in f.as_posix():
                    continue
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
        z.writestr("BUNDLE_INFO.txt", f"branch={branch} commit={rev} built={dt.datetime.now().isoformat(timespec='seconds')}\n"
                                      f"processed_data_included={(ROOT / OPTIONAL[0]).exists()}\n")
    print(f"wrote {ROOT / out}: {n} files, {(ROOT / out).stat().st_size / 2**20:.1f} MB "
          f"(processed data included: {(ROOT / OPTIONAL[0]).exists()})")


if __name__ == "__main__":
    main()