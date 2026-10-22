#!/usr/bin/env python
"""Evaluate a saved model on the held-out test pairs (RGB <-> SAR cross-modal retrieval).

One-line usage, from the project directory:

    python test.py

It reads  splits/test.csv  and the images in  test_dataset/RGB, test_dataset/SAR,
loads  checkpoints/best.pt , and prints mAP / P@K for RGB->SAR and SAR->RGB (plus
same-modality retrieval and a chance-level reference). No training data and no internet
access are needed (ImageNet weights are NOT downloaded; they are inside the checkpoint).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import dataset as ds
from utils import environment_info, get_device, resolve_path, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--split_csv", default="splits/test.csv")
    p.add_argument("--test_dir", default="test_dataset",
                   help="folder with RGB/<class>/<stem>.jpg and SAR/<class>/<stem>.tif")
    p.add_argument("--rgb_root", help="override: read RGB images from this folder instead")
    p.add_argument("--sar_root", help="override: read SAR images from this folder instead")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--l2norm", action="store_true", help="L2-normalise embeddings before ranking")
    p.add_argument("--output", default="reports/test_results.json")
    return p.parse_args(argv)


def main(argv=None):
    import torch
    from model import build_model
    from retrieval import evaluate_model, format_results_table

    args = parse_args(argv)
    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"[error] checkpoint not found: {ckpt_path}\n        Train first with the command in "
                 f"README.md, or pass --checkpoint.")
    csv_path = resolve_path(args.split_csv)
    if not csv_path.exists():
        sys.exit(f"[error] split file not found: {csv_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    class_names = ck["class_names"]
    set_seed(int(ck.get("seed", cfg.get("seed", 42))))
    device = get_device(args.device)

    records = ds.read_split(csv_path)
    rgb_root = args.rgb_root or resolve_path(args.test_dir) / "RGB"
    sar_root = args.sar_root or resolve_path(args.test_dir) / "SAR"
    for r in (rgb_root, sar_root):
        if not Path(r).exists():
            sys.exit(f"[error] test image folder not found: {r}\n        Create it with "
                     f"`python prepare_test_dataset.py --rgb_root ... --sar_root ...` or pass "
                     f"--rgb_root / --sar_root pointing at the full EuroSAT folders.")
    label_names = {c: l for l, c in enumerate(class_names)}
    for c, _, l in records:
        if label_names.get(c) != l:
            sys.exit(f"[error] label mismatch for class {c!r}: split file says {l}, "
                     f"checkpoint class list says {label_names.get(c)}. The split and checkpoint "
                     f"come from different runs.")

    print(f"[test] checkpoint: {ckpt_path} (epoch {ck.get('epoch')}, seed {ck.get('seed')})")
    print(f"[test] {len(records)} test pairs | RGB: {rgb_root} | SAR: {sar_root}")
    data = ds.load_arrays(records, rgb_root, sar_root,
                          num_threads=cfg["data"].get("io_threads", 8), desc="loading test data")

    model = build_model(cfg, num_classes=len(class_names), pretrained=False)  # no download
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()

    res, _ = evaluate_model(model, data, device, args.batch_size, amp=False, l2_normalize=args.l2norm)

    print("\nTest results (class-level relevance, Euclidean distance"
          f"{', L2-normalised' if args.l2norm else ''}); values are fractions in [0, 1]\n")
    print(format_results_table(res))
    print("\n'(strict)' removes the instance-paired counterpart from the gallery.")
    for k in ("RGB->SAR", "SAR->RGB"):
        print(f"\n{k} per-class AP:")
        print("  " + " | ".join(f"{class_names[c]}: {v:.3f}" for c, v in sorted(res[k]["per_class_mAP"].items())))
    print(f"\n>>> TEST mAP  RGB->SAR = {res['RGB->SAR']['mAP']:.4f}   SAR->RGB = {res['SAR->RGB']['mAP']:.4f}")

    out = {"checkpoint": str(ckpt_path), "checkpoint_epoch": ck.get("epoch"), "seed": ck.get("seed"),
           "n_test_pairs": len(records), "l2_normalize": bool(args.l2norm),
           "class_names": class_names, "results": res, "env": environment_info()}
    save_json(out, resolve_path(args.output))
    print(f"[saved] {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
