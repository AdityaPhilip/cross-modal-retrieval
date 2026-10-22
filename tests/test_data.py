"""Dataset pairing / split / loading tests (no torch needed). Uses synthetic data only."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dataset import (PairingError, check_splits_disjoint, discover_pairs, find_class_root,  # noqa: E402
                     fit_sar_stats, load_arrays, make_splits, pairing_sanity, read_class_names,
                     read_split, read_sar, write_splits)
from synth import make_synthetic_eurosat  # noqa: E402


def test_pairing_and_nested_root(tmp_path):
    rgb, sar = make_synthetic_eurosat(tmp_path)
    # emulate the user's nested layout  EuroSAT_RGB/EuroSAT_RGB/<class>
    import shutil
    nested = tmp_path / "nest" / "EuroSAT_RGB"
    shutil.copytree(rgb, nested / "EuroSAT_RGB")
    assert find_class_root(nested) == (nested / "EuroSAT_RGB")
    assert discover_pairs(nested, sar)[1]["n_matched"] == 72
    pairs, rep = discover_pairs(rgb, sar)
    assert rep["n_matched"] == 72 and rep["n_rgb_only"] == 0 and rep["n_sar_only"] == 0
    assert len(rep["classes"]) == 3
    # natural ordering: _2 before _10
    stems = [s for c, s in pairs if c == "Forest"]
    assert stems.index("Forest_2") < stems.index("Forest_10")


def test_unpaired_files_raise(tmp_path):
    rgb, sar = make_synthetic_eurosat(tmp_path)
    (sar / "Forest" / "Forest_3.tif").unlink()
    try:
        discover_pairs(rgb, sar)
    except PairingError as e:
        assert "RGB-only: 1" in str(e)
    else:
        raise AssertionError("expected PairingError")
    pairs, rep = discover_pairs(rgb, sar, strict=False)
    assert rep["n_matched"] == 71 and ("Forest", "Forest_3") not in pairs


def test_splits_deterministic_stratified_disjoint(tmp_path):
    rgb, sar = make_synthetic_eurosat(tmp_path, n_per_class=40)
    pairs, _ = discover_pairs(rgb, sar)
    a = make_splits(pairs, seed=42); b = make_splits(pairs, seed=42); c = make_splits(pairs, seed=7)
    assert a == b and a != c
    check_splits_disjoint(a)
    assert sum(len(v) for v in a.values()) == len(pairs)
    for cls in ("AnnualCrop", "Forest", "River"):
        n_tr = sum(1 for r in a["train"] if r[0] == cls)
        n_va = sum(1 for r in a["val"] if r[0] == cls)
        n_te = sum(1 for r in a["test"] if r[0] == cls)
        assert (n_tr, n_va, n_te) == (28, 6, 6)  # 70/15/15 of 40
    # round-trip through CSV
    names = write_splits(a, tmp_path / "splits")
    assert names == sorted(names) and read_class_names(tmp_path / "splits") == names
    tr = read_split(tmp_path / "splits" / "train.csv")
    assert [(c, s) for c, s, _ in tr] == a["train"]
    assert all(names[l] == c for c, _, l in tr)  # label <-> class consistency


def test_leakage_detector():
    bad = {"train": [("A", "A_1")], "val": [("A", "A_1")], "test": []}
    try:
        check_splits_disjoint(bad)
    except AssertionError:
        return
    raise AssertionError("leakage not detected")


def test_load_arrays_shapes_labels_pairing_and_cache(tmp_path):
    rgb, sar = make_synthetic_eurosat(tmp_path, n_per_class=10)
    pairs, _ = discover_pairs(rgb, sar)
    splits = make_splits(pairs, seed=1)
    names = write_splits(splits, tmp_path / "splits")
    recs = read_split(tmp_path / "splits" / "train.csv")
    cache = tmp_path / "cache" / "train.npz"
    d1 = load_arrays(recs, rgb, sar, num_threads=2, cache_path=cache)
    n = len(recs)
    assert d1["rgb"].shape == (n, 3, 64, 64) and d1["rgb"].dtype == np.uint8
    assert d1["sar"].shape == (n, 2, 64, 64) and d1["sar"].dtype == np.float32
    assert d1["labels"].shape == (n,) and cache.exists()
    # class-encoding in the synthetic SAR (mean ~ class index) confirms rgb[i] <-> sar[i] <-> label[i]
    for i in range(n):
        assert abs(d1["sar"][i, 1].mean() - d1["labels"][i]) < 0.3
    d2 = load_arrays(recs, rgb, sar, cache_path=cache)  # from cache
    assert np.array_equal(d1["rgb"], d2["rgb"]) and np.array_equal(d1["sar"], d2["sar"])


def test_sar_layouts_and_nan_handling(tmp_path):
    import tifffile

    a = np.random.default_rng(0).standard_normal((2, 64, 64)).astype(np.float32)
    tifffile.imwrite(tmp_path / "chw.tif", a)
    tifffile.imwrite(tmp_path / "hwc.tif", a.transpose(1, 2, 0))
    assert read_sar(tmp_path / "chw.tif").shape == (2, 64, 64)
    assert np.array_equal(read_sar(tmp_path / "hwc.tif"), a)
    b = a.copy(); b[0, 5, 5] = np.nan; b[1, 1, 1] = np.inf
    tifffile.imwrite(tmp_path / "bad.tif", b)
    r = read_sar(tmp_path / "bad.tif")
    assert np.isfinite(r).all()


def test_sar_stats_fit_on_train_only(tmp_path):
    rng = np.random.default_rng(0)
    sar = rng.standard_normal((50, 2, 16, 16)).astype(np.float32)
    sar[0, 0, 0, 0] = 1e6  # outlier must not blow up the stats
    st = fit_sar_stats(sar)
    assert len(st["lo"]) == 2 and st["hi"][0] < 10 and st["std"][0] > 0.5


def test_pairing_sanity_detects_correlated_pairs(tmp_path):
    rgb_r, sar_r = make_synthetic_eurosat(tmp_path / "a", n_per_class=20, correlated=True)
    pairs, _ = discover_pairs(rgb_r, sar_r)
    recs = [(c, s, ["AnnualCrop", "Forest", "River"].index(c)) for c, s in pairs]
    d = load_arrays(recs, rgb_r, sar_r, num_threads=2)
    res = pairing_sanity(d["rgb"], d["sar"], d["labels"], n=60)
    assert res["intensity"]["mean_abs_corr_true"] > res["intensity"]["mean_abs_corr_shuffled"] + 0.1
    assert res["intensity"]["frac_true_gt_shuffled"] > 0.8
