#!/usr/bin/env python
"""Create the deterministic train / val / test split over RGB-SAR PAIRS.

    python make_splits.py --rgb_root "<path>/EuroSAT_RGB" --sar_root "<path>/EuroSAT-SAR"

Writes splits/{train,val,test}.csv and splits/classes.json. The split is stratified by class,
uses numpy's default_rng(seed) on a naturally-sorted file list, and keeps both modalities of a
pair together, so re-running it on the same data always gives identical files.
"""
import argparse
import os
import sys
from collections import Counter

import dataset as ds
from utils import apply_overrides, load_config, resolve_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--rgb_root"); p.add_argument("--sar_root")
    p.add_argument("--seed", type=int)
    p.add_argument("--out_dir")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args(argv)
    cfg = apply_overrides(load_config(resolve_path(a.config)), a.set)
    rgb = a.rgb_root or os.environ.get("EUROSAT_RGB_ROOT") or cfg["data"]["rgb_root"]
    sar = a.sar_root or os.environ.get("EUROSAT_SAR_ROOT") or cfg["data"]["sar_root"]
    seed = a.seed if a.seed is not None else cfg["seed"]
    out = resolve_path(a.out_dir or cfg["data"]["split_dir"])
    try:
        pairs, rep = ds.discover_pairs(resolve_path(rgb), resolve_path(sar), strict=True)
    except (FileNotFoundError, ds.PairingError) as e:
        sys.exit(f"[error] {e}")
    print(f"RGB files: {rep['n_rgb']} | SAR files: {rep['n_sar']} | matched pairs: {rep['n_matched']}")
    splits = ds.make_splits(pairs, seed=seed, ratios=tuple(cfg["data"]["ratios"]))
    ds.check_splits_disjoint(splits)
    names = ds.write_splits(splits, out)
    print(f"seed={seed} ratios={cfg['data']['ratios']} -> {out}")
    print(f"{'class':<24}" + "".join(f"{k:>8}" for k in splits))
    for c in names:
        print(f"{c:<24}" + "".join(f"{Counter(r[0] for r in v)[c]:>8}" for v in splits.values()))
    print(f"{'TOTAL':<24}" + "".join(f"{len(v):>8}" for v in splits.values()))


if __name__ == "__main__":
    main()
