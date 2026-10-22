"""End-to-end smoke test: train.py -> best.pt -> test.py on synthetic data (needs PyTorch).

Uses random weights, 64px input and 2 epochs, so it finishes in about a minute on CPU.
It checks the plumbing (files written, finite metrics, train/test consistency, seed
reproducibility); it does NOT say anything about accuracy on real EuroSAT.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except ImportError:
    raise unittest.SkipTest("PyTorch / torchvision not installed")

import importlib.util  # noqa: E402


def _load(name, path):
    """Load by file path: a bare `import test` could resolve to Python's stdlib `test` package."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


test_script = _load("project_test_script", ROOT / "test.py")
train_script = _load("project_train_script", ROOT / "train.py")
from synth import make_synthetic_eurosat  # noqa: E402
from utils import load_json  # noqa: E402


def _run_train(tmp, tag, rgb, sar):
    out = tmp / tag
    args = ["--rgb_root", str(rgb), "--sar_root", str(sar), "--epochs", "2", "--batch_size", "16",
            "--device", "cpu", "--seed", "123",
            "--set", "model.pretrained=false", "--set", "model.input_size=64",
            "--set", "train.amp=false", "--set", "data.io_threads=2",
            "--set", f"data.split_dir={tmp / 'splits'}", "--set", f"data.cache_dir={tmp / 'cache'}",
            "--set", f"output.checkpoint_dir={out / 'ckpt'}", "--set", f"output.plot_dir={out / 'plots'}",
            "--set", f"output.report_dir={out / 'reports'}"]
    train_script.main(args)
    return out


def test_train_then_test_cli(tmp_path):
    rgb, sar = make_synthetic_eurosat(tmp_path / "data", n_per_class=30)
    a = _run_train(tmp_path, "run_a", rgb, sar)
    for f in ("ckpt/best.pt", "ckpt/last.pt", "plots/loss_curves.png", "plots/loss_components.png",
              "plots/history.json", "reports/train_summary.json", "reports/test_results_after_training.json"):
        assert (a / f).exists(), f
    hist = load_json(a / "plots/history.json")
    assert len(hist) == 2 and all(h["train_total"] == h["train_total"] for h in hist)  # not NaN

    out_json = tmp_path / "test_a.json"
    test_script.main(["--checkpoint", str(a / "ckpt/best.pt"), "--split_csv", str(tmp_path / "splits/test.csv"),
                      "--rgb_root", str(rgb), "--sar_root", str(sar), "--device", "cpu",
                      "--output", str(out_json)])
    res = load_json(out_json)["results"]
    for k in ("RGB->SAR", "SAR->RGB"):
        assert 0.0 <= res[k]["mAP"] <= 1.0 and res[k]["n_queries"] > 0
    # the mAP printed at the end of training must match a fresh test.py run on the same checkpoint
    after = load_json(a / "reports/test_results_after_training.json")["results"]
    assert abs(after["RGB->SAR"]["mAP"] - res["RGB->SAR"]["mAP"]) < 1e-4

    # same seed + same config -> (nearly) same result
    b = _run_train(tmp_path, "run_b", rgb, sar)
    out_b = tmp_path / "test_b.json"
    test_script.main(["--checkpoint", str(b / "ckpt/best.pt"), "--split_csv", str(tmp_path / "splits/test.csv"),
                      "--rgb_root", str(rgb), "--sar_root", str(sar), "--device", "cpu", "--output", str(out_b)])
    rb = load_json(out_b)["results"]
    assert abs(rb["RGB->SAR"]["mAP"] - res["RGB->SAR"]["mAP"]) < 0.05
