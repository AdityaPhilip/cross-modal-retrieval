"""EuroSAT RGB <-> EuroSAT-SAR paired data handling.

Everything here is NumPy-only (no torch) so it can be unit-tested anywhere.

Pairing
-------
EuroSAT-SAR was built by matching every EuroSAT Sentinel-2 patch with one
Sentinel-1 patch via geospatial coordinates, and it reuses EuroSAT's folder /
file naming. We therefore pair samples by (class folder, filename stem).
The dataset documentation does NOT state that the match is pixel-aligned or
acquired on the same date. ``pairing_sanity`` gives an empirical (weak) check.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from utils import progress

RGB_EXTS = (".jpg", ".jpeg", ".png")
SAR_EXTS = (".tif", ".tiff")


class PairingError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _visible_dirs(p: Path):
    return sorted(d for d in p.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))


def find_class_root(root: str | os.PathLike) -> Path:
    """Descend through single-child wrappers (e.g. EuroSAT_RGB/EuroSAT_RGB) until
    a directory containing >= 2 class sub-folders is found."""
    cur = Path(root)
    if not cur.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {cur}")
    for _ in range(4):
        subdirs = _visible_dirs(cur)
        if len(subdirs) >= 2:
            return cur
        if len(subdirs) == 1:
            cur = subdirs[0]
            continue
        break
    return cur


def index_folder(root: str | os.PathLike, exts: tuple[str, ...]) -> dict[tuple[str, str], Path]:
    """Map (class_name, stem) -> file path for every matching file under root/<class>/."""
    root = find_class_root(root)
    out: dict[tuple[str, str], Path] = {}
    for cdir in _visible_dirs(root):
        with os.scandir(cdir) as it:
            for e in it:
                if e.is_file() and Path(e.name).suffix.lower() in exts:
                    out[(cdir.name, Path(e.name).stem)] = Path(e.path)
    return out


def discover_pairs(rgb_root, sar_root, strict: bool = True):
    """Match RGB and SAR files by (class, stem).

    Returns (pairs, report). ``pairs`` is a sorted list of (class, stem).
    With ``strict`` any unmatched file raises PairingError.
    """
    rgb = index_folder(rgb_root, RGB_EXTS)
    sar = index_folder(sar_root, SAR_EXTS)
    matched = sorted(set(rgb) & set(sar), key=lambda p: (p[0], natural_key(p[1])))
    rgb_only = sorted(set(rgb) - set(sar))
    sar_only = sorted(set(sar) - set(rgb))
    report = {
        "n_rgb": len(rgb),
        "n_sar": len(sar),
        "n_matched": len(matched),
        "n_rgb_only": len(rgb_only),
        "n_sar_only": len(sar_only),
        "rgb_only_examples": rgb_only[:5],
        "sar_only_examples": sar_only[:5],
        "classes": sorted({c for c, _ in matched}),
    }
    if not matched:
        raise PairingError(
            f"No RGB/SAR files could be matched.\n  RGB root: {rgb_root}\n  SAR root: {sar_root}\n"
            f"  RGB files found: {len(rgb)}, SAR files found: {len(sar)}.\n"
            "Expected <root>/<ClassName>/<ClassName>_<n>.jpg|tif."
        )
    if strict and (rgb_only or sar_only):
        raise PairingError(
            f"Unpaired files found (RGB-only: {len(rgb_only)}, SAR-only: {len(sar_only)}). "
            f"Examples: {rgb_only[:3]} / {sar_only[:3]}. Fix the data or pass strict=False."
        )
    return matched, report


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_splits(pairs, seed: int = 42, ratios=(0.70, 0.15, 0.15)):
    """Stratified, deterministic split over *pairs* (an RGB/SAR pair never straddles splits)."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for c, s in pairs:
        by_class[c].append((c, s))
    splits = {"train": [], "val": [], "test": []}
    for c in sorted(by_class):
        items = sorted(by_class[c], key=lambda p: natural_key(p[1]))
        n = len(items)
        perm = rng.permutation(n)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        n_tr = min(n_tr, n - 2) if n >= 3 else n
        n_va = min(n_va, n - n_tr - 1) if n - n_tr >= 2 else 0
        splits["train"] += [items[i] for i in perm[:n_tr]]
        splits["val"] += [items[i] for i in perm[n_tr:n_tr + n_va]]
        splits["test"] += [items[i] for i in perm[n_tr + n_va:]]
    for k in splits:
        splits[k] = sorted(splits[k], key=lambda p: (p[0], natural_key(p[1])))
    return splits


def write_splits(splits, split_dir, class_names=None):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    if class_names is None:
        class_names = sorted({c for recs in splits.values() for c, _ in recs})
    label_of = {c: i for i, c in enumerate(class_names)}
    for name, recs in splits.items():
        with open(split_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class_name", "stem", "label"])
            for c, s in recs:
                w.writerow([c, s, label_of[c]])
    import json

    with open(split_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    return class_names


def read_split(csv_path):
    """Return list of (class_name, stem, label)."""
    recs = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            recs.append((row["class_name"], row["stem"], int(row["label"])))
    return recs


def read_class_names(split_dir):
    import json

    with open(Path(split_dir) / "classes.json", "r", encoding="utf-8") as f:
        return json.load(f)


def check_splits_disjoint(splits) -> None:
    seen = {}
    for name, recs in splits.items():
        for r in recs:
            key = (r[0], r[1])
            if key in seen:
                raise AssertionError(f"Leakage: {key} appears in both {seen[key]} and {name}")
            seen[key] = name


# --------------------------------------------------------------------------- #
# Reading images
# --------------------------------------------------------------------------- #
def read_rgb(path) -> np.ndarray:
    """-> uint8 array (3, H, W)."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def read_sar(path) -> np.ndarray:
    """-> float32 array (C, H, W). Non-finite values are replaced by the channel median."""
    import tifffile

    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3 and arr.shape[-1] <= 4 and arr.shape[0] > 4:
        arr = arr.transpose(2, 0, 1)  # H, W, C -> C, H, W
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected SAR array shape {arr.shape} in {path}")
    arr = arr.astype(np.float32)
    bad = ~np.isfinite(arr)
    if bad.any():
        for c in range(arr.shape[0]):
            ch = arr[c]
            m = ~np.isfinite(ch)
            if m.any():
                good = ch[~m]
                ch[m] = np.median(good) if good.size else 0.0
    return np.ascontiguousarray(arr)


def _records_hash(records, extra: str = "") -> str:
    h = hashlib.sha1(extra.encode())
    for c, s, _ in records:
        h.update(f"{c}/{s};".encode())
    return h.hexdigest()[:16]


def load_arrays(records, rgb_root, sar_root, num_threads: int = 8, cache_path=None,
                desc: str = "load"):
    """Load all pairs of ``records`` into memory.

    Returns dict(rgb=uint8 (N,3,H,W), sar=float32 (N,C,H,W), labels=int64 (N,), keys=list).
    If ``cache_path`` is given, a .npz cache keyed by the record list is used.
    """
    key = _records_hash(records)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                z = np.load(cache_path, allow_pickle=False)
                if str(z["key"]) == key:
                    return {
                        "rgb": z["rgb"], "sar": z["sar"], "labels": z["labels"],
                        "keys": [f"{c}/{s}" for c, s, _ in records],
                    }
            except Exception as e:  # corrupted cache -> rebuild
                print(f"[cache] ignoring unreadable cache {cache_path}: {e}")

    rgb_idx = index_folder(rgb_root, RGB_EXTS)
    sar_idx = index_folder(sar_root, SAR_EXTS)
    missing = [(c, s) for c, s, _ in records if (c, s) not in rgb_idx or (c, s) not in sar_idx]
    if missing:
        raise PairingError(
            f"{len(missing)} split records have no RGB and/or SAR file under the given roots, "
            f"e.g. {missing[:3]}.\n  RGB root: {rgb_root}\n  SAR root: {sar_root}"
        )

    def _load(rec):
        c, s, _ = rec
        return read_rgb(rgb_idx[(c, s)]), read_sar(sar_idx[(c, s)])

    with ThreadPoolExecutor(max_workers=max(1, num_threads)) as ex:
        results = list(progress(ex.map(_load, records), desc=desc, total=len(records)))
    try:
        rgb = np.stack([r[0] for r in results])
        sar = np.stack([r[1] for r in results])
    except ValueError as e:
        raise ValueError(f"Images in the dataset do not all have the same shape: {e}") from e
    labels = np.asarray([r[2] for r in records], dtype=np.int64)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, rgb=rgb, sar=sar, labels=labels, key=np.array(key))
    return {"rgb": rgb, "sar": sar, "labels": labels,
            "keys": [f"{c}/{s}" for c, s, _ in records]}


def subsample(data: dict, max_samples: int, seed: int = 0) -> dict:
    """Class-stratified-ish random subset (used for quick smoke runs only)."""
    n = len(data["labels"])
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return data
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_samples, replace=False))
    return {"rgb": data["rgb"][idx], "sar": data["sar"][idx], "labels": data["labels"][idx],
            "keys": [data["keys"][i] for i in idx]}


# --------------------------------------------------------------------------- #
# SAR normalisation statistics (fit on TRAIN only)
# --------------------------------------------------------------------------- #
def fit_sar_stats(sar: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0,
                  max_pixels: int = 2_000_000, seed: int = 0) -> dict:
    """Per-channel robust clip bounds + mean/std of the clipped values.

    Percentile clipping makes the pipeline robust to heavy-tailed SAR intensities
    whether the GeoTIFFs store linear power or dB values.
    """
    rng = np.random.default_rng(seed)
    n, c = sar.shape[0], sar.shape[1]
    lo, hi, mean, std = [], [], [], []
    for ch in range(c):
        flat = sar[:, ch].reshape(-1)
        if flat.size > max_pixels:
            flat = flat[rng.choice(flat.size, size=max_pixels, replace=False)]
        l, h = np.percentile(flat, [lo_pct, hi_pct])
        clipped = np.clip(flat, l, h)
        lo.append(float(l)); hi.append(float(h))
        mean.append(float(clipped.mean())); std.append(float(max(clipped.std(), 1e-6)))
    return {"lo": lo, "hi": hi, "mean": mean, "std": std}


def describe_sar(sar: np.ndarray) -> dict:
    out = {"shape": list(sar.shape), "dtype": str(sar.dtype), "channels": []}
    for ch in range(sar.shape[1]):
        x = sar[:, ch].reshape(-1)
        p = np.percentile(x[:: max(1, x.size // 1_000_000)], [0, 1, 50, 99, 100])
        out["channels"].append({"min": float(p[0]), "p1": float(p[1]), "median": float(p[2]),
                                "p99": float(p[3]), "max": float(p[4])})
    return out


# --------------------------------------------------------------------------- #
# Empirical pairing sanity check (weak evidence only)
# --------------------------------------------------------------------------- #
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) / d) if d > 1e-12 else 0.0


def _grad_mag(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img.astype(np.float64))
    return np.hypot(gx, gy)


def pairing_sanity(rgb: np.ndarray, sar: np.ndarray, labels: np.ndarray,
                   n: int = 500, seed: int = 0) -> dict:
    """Compare RGB-vs-SAR similarity of true pairs against same-class shuffled pairs.

    Uses |abs(r)| of two crude statistics: Pearson correlation of (a) grayscale vs SAR
    channel-0 intensity and (b) their gradient magnitudes. If true pairs are clearly more
    similar than shuffled same-class pairs, that supports spatial correspondence. A null result
    does NOT disprove pairing (optical and radar appearance differ a lot).
    """
    rng = np.random.default_rng(seed)
    N = len(labels)
    idx = rng.choice(N, size=min(n, N), replace=False)
    gray = rgb[idx].astype(np.float32).mean(axis=1)
    s0 = sar[idx, 0]
    lab = labels[idx]
    res = {"intensity": [], "intensity_shuffled": [], "gradient": [], "gradient_shuffled": []}
    for k in range(len(idx)):
        cands = np.where(lab == lab[k])[0]
        cands = cands[cands != k]
        if cands.size == 0:
            continue
        j = int(rng.choice(cands))
        res["intensity"].append(abs(_corr(gray[k], s0[k])))
        res["intensity_shuffled"].append(abs(_corr(gray[k], s0[j])))
        res["gradient"].append(abs(_corr(_grad_mag(gray[k]), _grad_mag(s0[k]))))
        res["gradient_shuffled"].append(abs(_corr(_grad_mag(gray[k]), _grad_mag(s0[j]))))
    out = {"n_pairs_checked": len(res["intensity"])}
    for kind in ("intensity", "gradient"):
        t = np.asarray(res[kind]); s = np.asarray(res[kind + "_shuffled"])
        out[kind] = {"mean_abs_corr_true": float(t.mean()), "mean_abs_corr_shuffled": float(s.mean()),
                     "frac_true_gt_shuffled": float((t > s).mean())}
    return out
