#!/usr/bin/env python
"""Train the attention-driven RGB <-> SAR cross-modal retrieval network.

One-line usage (Windows / Linux), from the project directory:

    python train.py --rgb_root "<path>/EuroSAT_RGB" --sar_root "<path>/EuroSAT-SAR"

(or put the paths in configs/default.yaml / set EUROSAT_RGB_ROOT and EUROSAT_SAR_ROOT).

Outputs: checkpoints/best.pt, checkpoints/last.pt, plots/*.png, plots/history.json,
reports/train_summary.json and (unless --no_final_test) reports/test_results_after_training.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import dataset as ds
from utils import (Timer, apply_overrides, environment_info, get_device, load_config, progress,
                   resolve_path, save_json, set_seed)

LOSS_KEYS = ["total", "l1", "l2", "l3", "l4", "aux", "acc_rgb", "acc_sar"]


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--rgb_root"); p.add_argument("--sar_root")
    p.add_argument("--epochs", type=int); p.add_argument("--batch_size", type=int)
    p.add_argument("--seed", type=int); p.add_argument("--device", default="auto")
    p.add_argument("--exp_name", help="write to checkpoints/<exp>/ and plots/<exp>/")
    p.add_argument("--max_train", type=int, default=0, help="smoke test: limit #train pairs")
    p.add_argument("--max_val", type=int, default=0, help="smoke test: limit #val pairs")
    p.add_argument("--no_final_test", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config entry, e.g. --set model.attention=false")
    return p.parse_args(argv)


def build_config(args) -> dict:
    cfg = load_config(resolve_path(args.config))
    cfg = apply_overrides(cfg, args.set)
    d = cfg["data"]
    d["rgb_root"] = args.rgb_root or os.environ.get("EUROSAT_RGB_ROOT") or d["rgb_root"]
    d["sar_root"] = args.sar_root or os.environ.get("EUROSAT_SAR_ROOT") or d["sar_root"]
    if args.epochs is not None: cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None: cfg["train"]["batch_size"] = args.batch_size
    if args.seed is not None: cfg["seed"] = args.seed
    if args.exp_name is not None: cfg["output"]["exp_name"] = args.exp_name
    return cfg


def output_dirs(cfg):
    exp = cfg["output"].get("exp_name") or ""
    dirs = {k: resolve_path(cfg["output"][k]) / exp for k in ("checkpoint_dir", "plot_dir", "report_dir")}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def prepare_splits(cfg):
    """Use existing splits/*.csv if present, otherwise create them deterministically."""
    split_dir = resolve_path(cfg["data"]["split_dir"])
    needed = [split_dir / f"{n}.csv" for n in ("train", "val", "test")] + [split_dir / "classes.json"]
    if all(p.exists() for p in needed):
        print(f"[data] using existing splits in {split_dir}")
    else:
        print(f"[data] creating splits (seed={cfg['seed']}) from the dataset folders ...")
        pairs, rep = ds.discover_pairs(cfg["data"]["rgb_root"], cfg["data"]["sar_root"], strict=True)
        print(f"[data] matched {rep['n_matched']} RGB/SAR pairs in {len(rep['classes'])} classes")
        splits = ds.make_splits(pairs, seed=cfg["seed"], ratios=tuple(cfg["data"]["ratios"]))
        ds.check_splits_disjoint(splits)
        ds.write_splits(splits, split_dir)
    return {n: ds.read_split(split_dir / f"{n}.csv") for n in ("train", "val", "test")}, \
        ds.read_class_names(split_dir)


def load_split_data(cfg, records, name, max_samples=0):
    cache = None
    if cfg["data"].get("use_cache", True):
        cache = resolve_path(cfg["data"]["cache_dir"]) / f"{name}_{ds._records_hash(records)}.npz"
    data = ds.load_arrays(records, cfg["data"]["rgb_root"], cfg["data"]["sar_root"],
                          num_threads=cfg["data"].get("io_threads", 8), cache_path=cache,
                          desc=f"loading {name}")
    return ds.subsample(data, max_samples, seed=cfg["seed"])


# --------------------------------------------------------------------------- #
# Batching / augmentation
# --------------------------------------------------------------------------- #
def random_dihedral(x):
    """Per-sample random h-flip, v-flip and transpose (dihedral group) on a (B,C,H,W) tensor."""
    import torch

    B = x.shape[0]

    def pick():
        return (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1, 1)

    x = torch.where(pick(), x.flip(3), x)
    x = torch.where(pick(), x.flip(2), x)
    if x.shape[2] == x.shape[3]:
        x = torch.where(pick(), x.transpose(2, 3), x)
    return x.contiguous()


def class_partners(labels, idx, class_index, rng):
    """For pairing='class': a random same-class SAR partner for every RGB index."""
    partners = np.empty_like(idx)
    lab = labels[idx]
    for c in np.unique(lab):
        m = lab == c
        partners[m] = rng.choice(class_index[int(c)], size=int(m.sum()), replace=True)
    return partners


def to_device(data, idx_rgb, idx_sar, device):
    import torch

    rgb = torch.from_numpy(data["rgb"][idx_rgb]).to(device, non_blocking=True)
    sar = torch.from_numpy(data["sar"][idx_sar]).to(device, non_blocking=True)
    y = torch.from_numpy(data["labels"][idx_rgb]).to(device, non_blocking=True)
    return rgb, sar, y


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def _new_acc():
    return {k: 0.0 for k in LOSS_KEYS}


def _accumulate(acc, losses, n):
    for k in LOSS_KEYS:
        acc[k] += float(losses[k].detach()) * n


def train_one_epoch(model, data, cfg, device, optimizer, scaler, rng, class_index):
    import torch
    from losses import total_loss

    model.train()
    n = len(data["labels"]); bs = cfg["train"]["batch_size"]
    perm = rng.permutation(n)
    weights = {k: cfg["loss"][f"w{i}"] for i, k in enumerate(["l1", "l2", "l3", "l4"], 1)}
    weights["aux"] = cfg["loss"]["aux"]
    use_amp = scaler is not None
    acc = _new_acc()
    starts = list(range(0, n, bs))
    for s in progress(starts, desc="train", total=len(starts)):
        idx = np.sort(perm[s:s + bs])
        sar_idx = class_partners(data["labels"], idx, class_index, rng) \
            if cfg["data"]["pairing"] == "class" else idx
        rgb, sar, y = to_device(data, idx, sar_idx, device)
        if cfg["train"].get("augment", True):
            rgb, sar = random_dihedral(rgb), random_dihedral(sar)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(rgb, sar)
        losses = total_loss(out, y, weights, cfg["loss"]["reduction"], cfg["model"]["attention"])
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("Loss became NaN/Inf. Try a lower learning rate or "
                                     "--set train.amp=false.")
        if use_amp:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if cfg["train"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer); scaler.update()
        else:
            losses["total"].backward()
            if cfg["train"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
        _accumulate(acc, losses, len(idx))
    return {k: v / n for k, v in acc.items()}


def validate(model, data, cfg, device):
    """Validation loss on TRUE pairs + cross-modal retrieval mAP in both directions."""
    import torch
    from losses import total_loss
    from retrieval import evaluate_retrieval, extract_embeddings

    model.eval()
    n = len(data["labels"]); bs = cfg["train"]["eval_batch_size"]
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    weights = {k: cfg["loss"][f"w{i}"] for i, k in enumerate(["l1", "l2", "l3", "l4"], 1)}
    weights["aux"] = cfg["loss"]["aux"]
    acc = _new_acc()
    with torch.no_grad():
        for s in range(0, n, bs):
            idx = np.arange(s, min(n, s + bs))
            rgb, sar, y = to_device(data, idx, idx, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(rgb, sar)
            losses = total_loss(out, y, weights, cfg["loss"]["reduction"], cfg["model"]["attention"])
            _accumulate(acc, losses, len(idx))
    res = {k: v / n for k, v in acc.items()}
    er, es = extract_embeddings(model, data, device, bs, use_amp)
    lab = data["labels"]
    res["map_rgb2sar"] = evaluate_retrieval(er, lab, es, lab)["mAP"]
    res["map_sar2rgb"] = evaluate_retrieval(es, lab, er, lab)["mAP"]
    res["map_mean"] = 0.5 * (res["map_rgb2sar"] + res["map_sar2rgb"])
    return res


# --------------------------------------------------------------------------- #
# Plots / checkpoints
# --------------------------------------------------------------------------- #
def plot_history(history, plot_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, [h["train_total"] for h in history], label="train")
    ax.plot(ep, [h["val_total"] for h in history], label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("total loss  L = L1+L2+L3+L4 (+aux)")
    ax.set_title("Training / validation loss"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "loss_curves.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, k in zip(axes.ravel(), ["l1", "l2", "l3", "l4", "aux"]):
        ax.plot(ep, [h[f"train_{k}"] for h in history], label="train")
        ax.plot(ep, [h[f"val_{k}"] for h in history], label="val")
        ax.set_title(k.upper()); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    axes[0, 0].legend()
    ax = axes[1, 2]
    ax.plot(ep, [h["train_acc_rgb"] for h in history], label="train acc (RGB)")
    ax.plot(ep, [h["val_acc_rgb"] for h in history], label="val acc (RGB)")
    ax.set_title("classifier accuracy (pair-free RGB)"); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "loss_components.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, [h["val_map_rgb2sar"] for h in history], label="RGB to SAR")
    ax.plot(ep, [h["val_map_sar2rgb"] for h in history], label="SAR to RGB")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation mAP"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Validation retrieval mAP")
    fig.tight_layout(); fig.savefig(plot_dir / "val_map.png", dpi=150); plt.close(fig)


def save_checkpoint(path, model, cfg, class_names, sar_stats, epoch, metrics, history):
    import torch

    torch.save({
        "format_version": 1,
        "model_state": model.state_dict(),
        "config": cfg,
        "class_names": class_names,
        "sar_stats": sar_stats,
        "seed": cfg["seed"],
        "epoch": epoch,
        "metrics": metrics,
        "history": history,
        "env": environment_info(),
    }, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    import torch
    from model import build_model
    from retrieval import evaluate_model, format_results_table

    args = parse_args(argv)
    cfg = build_config(args)
    set_seed(cfg["seed"])
    device = get_device(args.device)
    dirs = output_dirs(cfg)
    timer = Timer()

    for key in ("rgb_root", "sar_root"):
        root = resolve_path(cfg["data"][key])
        if not root.exists():
            sys.exit(f"[error] data.{key} = {cfg['data'][key]!r} does not exist. Training needs the "
                     f"full EuroSAT data; pass --rgb_root / --sar_root (see README).")
        cfg["data"][key] = str(root)
    splits, class_names = prepare_splits(cfg)
    print(f"[data] split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    train = load_split_data(cfg, splits["train"], "train", args.max_train)
    val = load_split_data(cfg, splits["val"], "val", args.max_val)
    if train["sar"].shape[1] != cfg["data"]["sar_channels"]:
        sys.exit(f"[error] SAR files have {train['sar'].shape[1]} channels but data.sar_channels="
                 f"{cfg['data']['sar_channels']}. Fix the config.")
    sar_stats = ds.fit_sar_stats(train["sar"], seed=cfg["seed"])
    print(f"[data] SAR stats (train only): lo={sar_stats['lo']} hi={sar_stats['hi']}")

    set_seed(cfg["seed"])  # identical initialisation regardless of data-loading time
    model = build_model(cfg, num_classes=len(class_names)).to(device)
    model.set_sar_stats(sar_stats)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] attention={cfg['model']['attention']} | params={n_params / 1e6:.1f}M | "
          f"input {cfg['model']['input_size']}px | emb {cfg['model']['emb_dim']}-d")

    tc = cfg["train"]
    optimizer = torch.optim.SGD(model.parameters(), lr=tc["lr"], momentum=tc["momentum"],
                                weight_decay=tc["weight_decay"])
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    class_index = {int(c): np.where(train["labels"] == c)[0] for c in np.unique(train["labels"])}
    rng = np.random.default_rng(cfg["seed"])

    history, best = [], {"score": -np.inf, "epoch": -1}
    best_path, last_path = dirs["checkpoint_dir"] / "best.pt", dirs["checkpoint_dir"] / "last.pt"
    print(f"[train] {tc['epochs']} epochs, batch {tc['batch_size']}, lr {tc['lr']}, AMP={use_amp}, "
          f"pairing={cfg['data']['pairing']}")
    try:
        for epoch in range(1, tc["epochs"] + 1):
            t0 = time.time()
            tr = train_one_epoch(model, train, cfg, device, optimizer, scaler, rng, class_index)
            va = validate(model, val, cfg, device)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                   **{f"val_{k}": va[k] for k in LOSS_KEYS},
                   "val_map_rgb2sar": va["map_rgb2sar"], "val_map_sar2rgb": va["map_sar2rgb"],
                   "val_map_mean": va["map_mean"], "seconds": time.time() - t0}
            history.append(row)
            score = va["map_mean"] if tc["select_by"] == "val_map" else -va["total"]
            is_best = score > best["score"]
            if is_best:
                best = {"score": score, "epoch": epoch, "val_map_mean": va["map_mean"],
                        "val_map_rgb2sar": va["map_rgb2sar"], "val_map_sar2rgb": va["map_sar2rgb"],
                        "val_total": va["total"]}
                save_checkpoint(best_path, model, cfg, class_names, sar_stats, epoch, best, history)
            save_checkpoint(last_path, model, cfg, class_names, sar_stats, epoch, row, history)
            save_json(history, dirs["plot_dir"] / "history.json")
            plot_history(history, dirs["plot_dir"])
            print(f"epoch {epoch:3d}/{tc['epochs']} | train {tr['total']:.4f} | val {va['total']:.4f} | "
                  f"val mAP RGB->SAR {va['map_rgb2sar']:.4f} SAR->RGB {va['map_sar2rgb']:.4f} | "
                  f"{row['seconds']:.0f}s{'  *best*' if is_best else ''}", flush=True)
    except torch.cuda.OutOfMemoryError:
        sys.exit("[error] CUDA out of memory. Re-run with a smaller batch, e.g. --batch_size 32 "
                 "(or --set model.input_size=96).")
    except KeyboardInterrupt:
        print("\n[interrupted] saving what we have so far.")

    # the best checkpoint is saved in the loop (and the last one is already on disk)
    # re-save the final history into best.pt so plots and checkpoint agree
    if best["epoch"] > 0:
        ck = torch.load(best_path, map_location="cpu", weights_only=True)
        ck["history"] = history
        torch.save(ck, best_path)

    summary = {"config": cfg, "seed": cfg["seed"], "best": best, "epochs_run": len(history),
               "split_sizes": {k: len(v) for k, v in splits.items()}, "sar_stats": sar_stats,
               "n_params": n_params, "train_seconds": timer.elapsed(), "env": environment_info(),
               "checkpoint": str(best_path), "plots": str(dirs["plot_dir"])}
    save_json(summary, dirs["report_dir"] / "train_summary.json")
    print(f"[done] best epoch {best['epoch']} (val mAP mean {best.get('val_map_mean', float('nan')):.4f}) "
          f"| total {timer.elapsed() / 60:.1f} min\n[saved] {best_path}\n[saved] {dirs['plot_dir']}")

    if not args.no_final_test and best["epoch"] > 0:
        print("\n[final test] evaluating best.pt on the held-out TEST split (selected on validation only)")
        ck = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model_state"])
        test = load_split_data(cfg, splits["test"], "test", 0)
        res, _ = evaluate_model(model, test, device, tc["eval_batch_size"], False,
                                cfg["test"]["l2_normalize"])  # fp32, same as test.py
        print(format_results_table(res))
        save_json({"checkpoint": str(best_path), "seed": cfg["seed"], "n_test_pairs": len(test["labels"]),
                   "results": res}, dirs["report_dir"] / "test_results_after_training.json")


if __name__ == "__main__":
    main()
