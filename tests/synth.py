"""Create a tiny synthetic EuroSAT-like dataset (for tests only; NOT real data)."""
from pathlib import Path

import numpy as np


def make_synthetic_eurosat(root, classes=("AnnualCrop", "Forest", "River"), n_per_class=24,
                           size=64, sar_layout="chw", seed=0, correlated=True):
    """Writes <root>/EuroSAT_RGB/<cls>/<cls>_<i>.jpg and <root>/EuroSAT-SAR/<cls>/<cls>_<i>.tif.

    The SAR image is (optionally) a noisy function of the RGB brightness so that the pairing
    sanity check has something to detect. Class identity is encoded in the mean colour.
    """
    import tifffile
    from PIL import Image

    rng = np.random.default_rng(seed)
    root = Path(root)
    rgb_root, sar_root = root / "EuroSAT_RGB", root / "EuroSAT-SAR"
    for ci, c in enumerate(classes):
        (rgb_root / c).mkdir(parents=True, exist_ok=True)
        (sar_root / c).mkdir(parents=True, exist_ok=True)
        for i in range(1, n_per_class + 1):
            base = np.zeros((size, size, 3), np.float32)
            base[..., ci % 3] = 120
            tex = rng.random((size, size)).astype(np.float32)
            tex = (tex + np.roll(tex, 1, 0) + np.roll(tex, 1, 1)) / 3
            img = np.clip(base + 100 * tex[..., None], 0, 255).astype(np.uint8)
            Image.fromarray(img).save(rgb_root / c / f"{c}_{i}.jpg", quality=95)
            gray = img.astype(np.float32).mean(-1)
            if correlated:
                s0 = 0.05 * gray + 0.3 * rng.standard_normal((size, size)) + ci
            else:
                s0 = rng.standard_normal((size, size)) + ci
            s1 = rng.standard_normal((size, size)) + ci
            sar = np.stack([s0, s1]).astype(np.float32)  # (2,H,W)
            if sar_layout == "hwc":
                sar = sar.transpose(1, 2, 0)
            tifffile.imwrite(sar_root / c / f"{c}_{i}.tif", sar)
    return rgb_root, sar_root
