"""Loss functions of Att-CMIR-Net (Eqs. 1-4 of the paper) + one auxiliary term.

Paper (V_a = RGB/photo embedding, V_b = SAR/sketch embedding):
    L1 = ||V_a - V_b||_F^2                         alignment of paired embeddings
    L2 = CE(V_ab),  V_ab = [V_a, V_b]              classification of both modalities
    L3 = ||V_a||_F^2 + ||V_b||_F^2                 norm penalty
    L4 = ||w_ab V_a - V_b||_F^2 + ||w_ba V_b - V_a||_F^2   cross-decoder reconstruction
    L  = L1 + L2 + L3 + L4

Implementation choices (documented in README / report):
  * ``reduction='mean'`` (default) averages squared errors over batch AND embedding
    dimensions, so the magnitude of L1/L3/L4 does not scale with the 128-d embedding.
    ``reduction='sum'`` gives the literal per-sample Frobenius norm (averaged over the batch).
  * V_ab is realised as the concatenation of the two modalities' samples along the batch axis,
    with a shared linear classifier and the same labels for both halves.
  * AUXILIARY (not in the paper): because inference embeds each modality *without* its pair,
    the pair-free RGB path is (i) trained with CE and (ii) pulled towards the attended RGB
    embedding (stop-gradient), so the attention benefit transfers to the pair-free path.
  * Taken literally L1 and L3 admit the trivial solution V = 0; only L2 resists it. The
    per-term weights (default 1.0, i.e. the plain sum of the paper) are configurable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _sq(a: torch.Tensor, b: torch.Tensor, reduction: str) -> torch.Tensor:
    d = (a.float() - b.float()) ** 2
    return d.mean() if reduction == "mean" else d.sum(dim=1).mean()


def _sqnorm(a: torch.Tensor, reduction: str) -> torch.Tensor:
    d = a.float() ** 2
    return d.mean() if reduction == "mean" else d.sum(dim=1).mean()


def loss_l1(v_a, v_b, reduction="mean"):
    return _sq(v_a, v_b, reduction)


def loss_l2(logits_a, logits_b, labels):
    logits = torch.cat([logits_a.float(), logits_b.float()], dim=0)
    return F.cross_entropy(logits, torch.cat([labels, labels], dim=0))


def loss_l3(v_a, v_b, reduction="mean"):
    return _sqnorm(v_a, reduction) + _sqnorm(v_b, reduction)


def loss_l4(dec_a2b, dec_b2a, v_a, v_b, reduction="mean"):
    return _sq(dec_a2b, v_b, reduction) + _sq(dec_b2a, v_a, reduction)


def loss_aux(v_plain, v_att, logits_plain, labels, reduction="mean"):
    """Pair-free RGB path: CE + distillation towards the (detached) attended embedding."""
    return F.cross_entropy(logits_plain.float(), labels) + _sq(v_plain, v_att.detach(), reduction)


def total_loss(out: dict, labels: torch.Tensor, weights: dict | None = None,
               reduction: str = "mean", use_attention: bool = True) -> dict:
    """Compute all terms. Returns a dict of scalar tensors (``total`` is differentiable)."""
    w = {"l1": 1.0, "l2": 1.0, "l3": 1.0, "l4": 1.0, "aux": 1.0}
    if weights:
        w.update(weights)
    l1 = loss_l1(out["v_rgb"], out["v_sar"], reduction)
    l2 = loss_l2(out["logits_rgb"], out["logits_sar"], labels)
    l3 = loss_l3(out["v_rgb"], out["v_sar"], reduction)
    l4 = loss_l4(out["dec_a2b"], out["dec_b2a"], out["v_rgb"], out["v_sar"], reduction)
    total = w["l1"] * l1 + w["l2"] * l2 + w["l3"] * l3 + w["l4"] * l4
    if use_attention:
        aux = loss_aux(out["v_rgb_plain"], out["v_rgb"], out["logits_rgb_plain"], labels, reduction)
        total = total + w["aux"] * aux
    else:
        aux = torch.zeros((), device=total.device)
    with torch.no_grad():
        acc_rgb = (out["logits_rgb_plain"].argmax(1) == labels).float().mean()
        acc_sar = (out["logits_sar"].argmax(1) == labels).float().mean()
    return {"total": total, "l1": l1, "l2": l2, "l3": l3, "l4": l4, "aux": aux,
            "acc_rgb": acc_rgb, "acc_sar": acc_sar}
