"""Shared utilities: seeding, config handling, device selection, JSON helpers.

This module deliberately avoids importing torch at module level so that the
data / retrieval unit tests can run in environments without PyTorch.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (if installed).

    Sources of residual non-determinism are documented in the README
    (mixed precision, some CUDA kernels, different GPU / driver / torch builds).
    """
    # Must be set before the first cuBLAS call to take effect.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # data-only environments
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - very old torch
            pass


def get_device(preference: str = "auto"):
    """Return a torch.device. 'auto' -> CUDA if available, else CPU."""
    import torch

    if preference == "auto":
        preference = "cuda" if torch.cuda.is_available() else "cpu"
    if preference.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        preference = "cpu"
    device = torch.device(preference)
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        print(f"[device] {props.name} | {props.total_memory / 1024 ** 3:.1f} GB VRAM")
    else:
        print("[device] CPU (training will be slow; see README for the CUDA setup)")
    return device


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` style overrides (values parsed as YAML scalars)."""
    import yaml

    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.subkey=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def resolve_path(p: str | os.PathLike) -> Path:
    """Relative paths are resolved against the project directory, not the CWD."""
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# --------------------------------------------------------------------------- #
# JSON / misc
# --------------------------------------------------------------------------- #
def _jsonable(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def progress(iterable, desc: str = "", total: int | None = None):
    """tqdm if available, otherwise a plain iterator."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total, leave=False)
    except ImportError:
        return iterable
