#!/usr/bin/env python
"""Pack the final single-ZIP submission (run after training, testing and make_report.py).

    python make_submission.py            ->  ../cross_modal_retrieval_submission.zip

Contains: all code, README, requirements, config, splits, test_dataset, checkpoints/best.pt,
plots, reports (report.pdf + JSON results) and tests. Excludes caches, virtual environments,
the full training data and checkpoints/last.pt (add --include_last to keep it).
"""
import argparse
import sys
import zipfile
from pathlib import Path

from utils import PROJECT_ROOT

REQUIRED = ["README.md", "requirements.txt", "train.py", "test.py", "model.py", "dataset.py", "losses.py",
            "retrieval.py", "utils.py", "configs/default.yaml", "splits/train.csv", "splits/val.csv",
            "splits/test.csv", "splits/classes.json", "checkpoints/best.pt", "plots/loss_curves.png",
            "reports/report.pdf", "reports/test_results.json"]
EXCLUDE_DIRS = {"__pycache__", ".venv", "venv", "cache", "data", ".git", ".pytest_cache", "smoke"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default=str(PROJECT_ROOT.parent / "cross_modal_retrieval_submission.zip"))
    p.add_argument("--include_last", action="store_true")
    p.add_argument("--allow_incomplete", action="store_true")
    a = p.parse_args(argv)

    missing = [r for r in REQUIRED if not (PROJECT_ROOT / r).exists()]
    for sub in ("RGB", "SAR"):
        if not any((PROJECT_ROOT / "test_dataset" / sub).rglob("*.*")):
            missing.append(f"test_dataset/{sub}/ (run prepare_test_dataset.py)")
    if missing and not a.allow_incomplete:
        sys.exit("[error] required items missing:\n  - " + "\n  - ".join(missing) +
                 "\nFinish the README workflow first (or use --allow_incomplete).")

    out = Path(a.out)
    n, size = 0, 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PROJECT_ROOT.rglob("*")):
            rel = f.relative_to(PROJECT_ROOT)
            if f.is_dir() or EXCLUDE_DIRS & set(rel.parts) or f.suffix == ".pyc":
                continue
            if f.name == "last.pt" and not a.include_last:
                continue
            z.write(f, Path(PROJECT_ROOT.name) / rel)
            n += 1; size += f.stat().st_size
    print(f"wrote {out} | {n} files | {size / 1024 ** 2:.0f} MB uncompressed | {out.stat().st_size / 1024 ** 2:.0f} MB zipped")
    if missing:
        print("[warn] incomplete package; missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
