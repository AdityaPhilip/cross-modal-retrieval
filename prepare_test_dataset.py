#!/usr/bin/env python
"""Copy ONLY the test-split image pairs into test_dataset/ so the ZIP is self-contained.

    python prepare_test_dataset.py --rgb_root "<path>/EuroSAT_RGB" --sar_root "<path>/EuroSAT-SAR"

Result: test_dataset/RGB/<class>/<stem>.jpg and test_dataset/SAR/<class>/<stem>.tif for every
row of splits/test.csv (same relative layout as the original datasets).
"""
import argparse
import os
import shutil
import sys

import dataset as ds
from utils import progress, resolve_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--rgb_root", default=os.environ.get("EUROSAT_RGB_ROOT", "data/EuroSAT_RGB"))
    p.add_argument("--sar_root", default=os.environ.get("EUROSAT_SAR_ROOT", "data/EuroSAT-SAR"))
    p.add_argument("--split_csv", default="splits/test.csv")
    p.add_argument("--out_dir", default="test_dataset")
    a = p.parse_args(argv)
    csv_path = resolve_path(a.split_csv)
    if not csv_path.exists():
        sys.exit(f"[error] {csv_path} not found. Run make_splits.py first.")
    records = ds.read_split(csv_path)
    rgb_idx = ds.index_folder(resolve_path(a.rgb_root), ds.RGB_EXTS)
    sar_idx = ds.index_folder(resolve_path(a.sar_root), ds.SAR_EXTS)
    out = resolve_path(a.out_dir)
    total = 0
    for c, s, _ in progress(records, desc="copying", total=len(records)):
        if (c, s) not in rgb_idx or (c, s) not in sar_idx:
            sys.exit(f"[error] missing file for {c}/{s}")
        for sub, idx in (("RGB", rgb_idx), ("SAR", sar_idx)):
            src = idx[(c, s)]
            dst = out / sub / c / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            total += src.stat().st_size
    print(f"copied {len(records)} pairs ({total / 1024 ** 2:.1f} MB) to {out}")


if __name__ == "__main__":
    main()
