"""Model / loss tests (need PyTorch + torchvision; skipped otherwise).

Run:  python tests/run_tests.py model      (or: pytest tests/test_model_losses.py)
Uses randomly initialised weights (no download) and 64x64 inputs, so it is fast on CPU.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torchvision  # noqa: F401
except ImportError:
    raise unittest.SkipTest("PyTorch / torchvision not installed")

from losses import loss_l1, loss_l2, loss_l3, loss_l4, total_loss  # noqa: E402
from model import build_model  # noqa: E402
from utils import apply_overrides, load_config, set_seed  # noqa: E402

C = 10


def _cfg(**over):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    ov = ["model.pretrained=false", "model.input_size=64"] + [f"{k}={v}" for k, v in over.items()]
    return apply_overrides(cfg, ov)


def _batch(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.randint(0, 256, (b, 3, 64, 64), dtype=torch.uint8, generator=g)
    sar = torch.randn(b, 2, 64, 64, generator=g)
    y = torch.randint(0, C, (b,), generator=g)
    return rgb, sar, y


def test_forward_shapes_and_attention_range():
    set_seed(0)
    m = build_model(_cfg(), C, pretrained=False).eval()
    rgb, sar, _ = _batch()
    out = m(rgb, sar)
    assert out["v_rgb"].shape == (4, 128) and out["v_sar"].shape == (4, 128)
    assert out["v_rgb_plain"].shape == (4, 128)
    assert out["att"].shape == (4, 512) and float(out["att"].min()) > 0 and float(out["att"].max()) < 1
    assert out["logits_rgb"].shape == (4, C) and out["dec_a2b"].shape == (4, 128)
    assert m.embed_rgb(rgb).shape == (4, 128) and m.embed_sar(sar).shape == (4, 128)
    assert m.enc_rgb.features[0].in_channels == 3 and m.enc_sar.features[0].in_channels == 2


def test_attention_is_actually_used():
    set_seed(0)
    m = build_model(_cfg(), C, pretrained=False).eval()
    rgb, sar, _ = _batch()
    sar2 = torch.randn_like(sar) * 3
    o1, o2 = m(rgb, sar), m(rgb, sar2)
    assert not torch.allclose(o1["v_rgb"], o2["v_rgb"], atol=1e-6)          # SAR changes attended RGB
    assert torch.allclose(o1["v_rgb_plain"], o2["v_rgb_plain"], atol=1e-6)  # pair-free path unaffected
    m_no = build_model(_cfg(**{"model.attention": "false"}), C, pretrained=False).eval()
    p1, p2 = m_no(rgb, sar), m_no(rgb, sar2)
    assert torch.allclose(p1["v_rgb"], p2["v_rgb"], atol=1e-6) and p1["att"] is None


def test_loss_values_hand_checked():
    a, b = torch.ones(2, 3), torch.zeros(2, 3)
    assert abs(float(loss_l1(a, b, "mean")) - 1.0) < 1e-6
    assert abs(float(loss_l1(a, b, "sum")) - 3.0) < 1e-6          # per-sample ||.||_F^2, batch mean
    assert abs(float(loss_l3(a, b, "mean")) - 1.0) < 1e-6
    assert abs(float(loss_l3(a, b, "sum")) - 3.0) < 1e-6
    assert abs(float(loss_l4(b, b, a, b, "mean")) - 1.0) < 1e-6   # ||0-0||^2 + ||0-1||^2 (mean)
    logits = torch.zeros(2, 4)
    assert abs(float(loss_l2(logits, logits, torch.tensor([0, 1]))) - float(torch.log(torch.tensor(4.0)))) < 1e-6


def test_total_loss_finite_and_every_parameter_gets_gradient():
    set_seed(0)
    m = build_model(_cfg(), C, pretrained=False).train()
    rgb, sar, y = _batch()
    L = total_loss(m(rgb, sar), y, None, "mean", True)
    for k, v in L.items():
        assert torch.isfinite(v), k
    assert abs(float(L["total"] - (L["l1"] + L["l2"] + L["l3"] + L["l4"] + L["aux"]))) < 1e-5
    L["total"].backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for: {missing}"
    bad = [n for n, p in m.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradient for: {bad}"


def _grad_norm(m, name):
    g = dict(m.named_parameters())[name].grad
    return 0.0 if g is None else float(g.abs().sum())


def test_each_loss_term_contributes_independently():
    rgb, sar, y = _batch(8)
    plan = {  # term -> a parameter that must receive a non-zero gradient from that term alone
        "l1": "unified.2.weight", "l2": "classifier.weight", "l3": "unified.2.weight",
        "l4": "dec_ab.weight", "aux": "head_rgb.fc.weight",
    }
    for term, param in plan.items():
        set_seed(0)
        m = build_model(_cfg(), C, pretrained=False).train()
        w = {k: (1.0 if k == term else 0.0) for k in ("l1", "l2", "l3", "l4", "aux")}
        L = total_loss(m(rgb, sar), y, w, "mean", True)
        L["total"].backward()
        assert _grad_norm(m, param) > 0, f"{term} does not reach {param}"
    # attention distillation target is detached: aux must not push gradients into att_fc
    set_seed(0)
    m = build_model(_cfg(), C, pretrained=False).train()
    w = {k: (1.0 if k == "aux" else 0.0) for k in ("l1", "l2", "l3", "l4", "aux")}
    total_loss(m(rgb, sar), y, w, "mean", True)["total"].backward()
    assert _grad_norm(m, "att_fc.weight") == 0.0
    # while L1 (which uses the attended path) must train the attention FC
    set_seed(0)
    m = build_model(_cfg(), C, pretrained=False).train()
    w = {k: (1.0 if k == "l1" else 0.0) for k in ("l1", "l2", "l3", "l4", "aux")}
    total_loss(m(rgb, sar), y, w, "mean", True)["total"].backward()
    assert _grad_norm(m, "att_fc.weight") > 0.0


def test_no_attention_ablation_runs():
    set_seed(0)
    m = build_model(_cfg(**{"model.attention": "false"}), C, pretrained=False).train()
    rgb, sar, y = _batch()
    L = total_loss(m(rgb, sar), y, None, "mean", False)
    L["total"].backward()
    assert torch.isfinite(L["total"]) and float(L["aux"]) == 0.0


def test_checkpoint_roundtrip_and_sar_stats(tmp_path):
    set_seed(0)
    m = build_model(_cfg(), C, pretrained=False)
    stats = {"lo": [-3.0, -2.0], "hi": [3.0, 2.0], "mean": [0.1, 0.2], "std": [1.5, 0.7]}
    m.set_sar_stats(stats)
    m.eval()
    rgb, sar, _ = _batch()
    with torch.no_grad():
        e1 = m.embed_sar(sar)
    path = tmp_path / "ck.pt"
    torch.save({"model_state": m.state_dict(), "config": _cfg()}, path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m2 = build_model(ck["config"], C, pretrained=False)
    m2.load_state_dict(ck["model_state"]); m2.eval()
    with torch.no_grad():
        e2 = m2.embed_sar(sar)
    assert torch.allclose(e1, e2, atol=1e-6)
    assert torch.allclose(m2.sar_lo.flatten(), torch.tensor(stats["lo"]))


def test_tiny_overfit_loss_decreases():
    set_seed(0)
    m = build_model(_cfg(), C, pretrained=False).eval()  # eval: no dropout noise
    rgb, sar, y = _batch(8)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9)
    vals = []
    for _ in range(25):
        opt.zero_grad()
        L = total_loss(m(rgb, sar), y, None, "mean", True)["total"]
        L.backward(); opt.step(); vals.append(float(L))
    assert vals[-1] < vals[0], vals
