#!/usr/bin/env python
"""Inspect the EuroSAT RGB / SAR folders and sanity-check the pairing.

    python inspect_data.py --rgb_root "<path>/EuroSAT_RGB" --sar_root "<path>/EuroSAT-SAR"

What this establishes
  * file-level pairing: every (class, filename stem) exists in both modalities (hard check);
  * array shapes / dtypes / value ranges of the GeoTIFFs;
  * a WEAK empirical check that true pairs look more alike than same-class shuffled pairs.
What it does NOT establish: pixel-level alignment or same-date acquisition. The EuroSAT-SAR
dataset description says each Sentinel-2 patch was matched with one Sentinel-1 patch via
geospatial coordinates, which is the (documented) basis of the pairing used in this project.
"""
import argparse
import os
import sys

import numpy as np

import dataset as ds
from utils import resolve_path, save_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--rgb_root", default=os.environ.get("EUROSAT_RGB_ROOT", "data/EuroSAT_RGB"))
    p.add_argument("--sar_root", default=os.environ.get("EUROSAT_SAR_ROOT", "data/EuroSAT-SAR"))
    p.add_argument("--n", type=int, default=600, help="number of random pairs to load")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="reports/data_inspection.json")
    a = p.parse_args(argv)
    try:
        pairs, rep = ds.discover_pairs(resolve_path(a.rgb_root), resolve_path(a.sar_root), strict=False)
    except (FileNotFoundError, ds.PairingError) as e:
        sys.exit(f"[error] {e}")
    print(f"RGB files {rep['n_rgb']} | SAR files {rep['n_sar']} | matched {rep['n_matched']} | "
          f"RGB-only {rep['n_rgb_only']} | SAR-only {rep['n_sar_only']}")
    print(f"classes ({len(rep['classes'])}): {rep['classes']}")
    rng = np.random.default_rng(a.seed)
    sel = sorted(rng.choice(len(pairs), size=min(a.n, len(pairs)), replace=False))
    classes = rep["classes"]
    recs = [(pairs[i][0], pairs[i][1], classes.index(pairs[i][0])) for i in sel]
    d = ds.load_arrays(recs, resolve_path(a.rgb_root), resolve_path(a.sar_root), desc="sampling")
    desc = ds.describe_sar(d["sar"])
    print(f"\nRGB sample array: {d['rgb'].shape} {d['rgb'].dtype}")
    print(f"SAR sample array: {desc['shape']} {desc['dtype']}")
    for i, ch in enumerate(desc["channels"]):
        print(f"  SAR channel {i}: " + ", ".join(f"{k}={v:.4g}" for k, v in ch.items()))
    sanity = ds.pairing_sanity(d["rgb"], d["sar"], d["labels"], n=len(recs), seed=a.seed)
    print(f"\nPairing sanity check on {sanity['n_pairs_checked']} pairs (mean |Pearson r|):")
    for kind in ("intensity", "gradient"):
        s = sanity[kind]
        print(f"  {kind:<10} true pairs {s['mean_abs_corr_true']:.4f} | same-class shuffled "
              f"{s['mean_abs_corr_shuffled']:.4f} | true > shuffled in {100 * s['frac_true_gt_shuffled']:.1f}%")
    print("\nInterpretation: a clear gap supports spatial correspondence; no gap is inconclusive "
          "(optical and radar imagery are not expected to correlate strongly).")
    save_json({"file_pairing": rep, "sar_stats_sample": desc, "pairing_sanity": sanity},
              resolve_path(a.output))
    print(f"[saved] {resolve_path(a.output)}")


if __name__ == "__main__":
    main()
