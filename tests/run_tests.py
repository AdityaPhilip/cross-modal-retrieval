"""Dependency-free test runner (works with or without pytest installed).

    python tests/run_tests.py            # run everything
    python tests/run_tests.py data       # only files whose name contains 'data'

Tests that need PyTorch are reported as SKIPPED when torch is missing.
"""
import importlib.util
import inspect
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(pattern: str = "") -> int:
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE))
    n_pass = n_fail = n_skip = 0
    for f in sorted(HERE.glob("test_*.py")):
        if pattern and pattern not in f.name:
            continue
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except unittest.SkipTest as e:
            print(f"SKIP  {f.name}: {e}"); n_skip += 1; continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                print(f"PASS  {f.name}::{name}"); n_pass += 1
            except unittest.SkipTest as e:
                print(f"SKIP  {f.name}::{name}: {e}"); n_skip += 1
            except Exception:
                print(f"FAIL  {f.name}::{name}"); traceback.print_exc(); n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
