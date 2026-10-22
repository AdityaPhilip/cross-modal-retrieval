"""Retrieval evaluation (NumPy only, except ``extract_embeddings``).

Protocol (applies to every experiment in this project)
------------------------------------------------------
* Relevance: gallery item g is relevant to query q  <=>  label(g) == label(q)  (class-level).
* Ranking: ascending Euclidean distance between embeddings; stable sort for ties.
* AP(q) = (1 / R) * sum_{k: rank k relevant} Precision@k, over the FULL ranked gallery,
  R = number of relevant gallery items. mAP = mean over evaluated queries.
* P@K = fraction of relevant items among the top K.
* Queries with R == 0 are skipped and counted (``n_skipped``).
* Same-modality retrieval excludes the query itself from the gallery (``exclude_idx``).
* Cross-modal retrieval reports two variants:
    - standard : the instance-paired counterpart (same index, other modality) stays in the gallery;
    - strict   : the paired counterpart is removed from the gallery, so no instance-level
                 correspondence can be exploited.
"""
from __future__ import annotations

import numpy as np

DEFAULT_KS = (10, 20, 50)


def pairwise_sq_euclidean(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64); g = g.astype(np.float64)
    d = (q ** 2).sum(1)[:, None] + (g ** 2).sum(1)[None, :] - 2.0 * q @ g.T
    return np.maximum(d, 0.0)


def evaluate_retrieval(q_emb, q_lab, g_emb, g_lab, exclude_idx=None, ks=DEFAULT_KS,
                       chunk: int = 512, l2_normalize: bool = False) -> dict:
    """Compute mAP and P@K for one retrieval direction.

    exclude_idx: optional int array (len = n_queries); gallery index to drop for each query
                 (use arange for same-modality retrieval, or the paired index for 'strict').
    """
    q_emb = np.asarray(q_emb, dtype=np.float64); g_emb = np.asarray(g_emb, dtype=np.float64)
    q_lab = np.asarray(q_lab); g_lab = np.asarray(g_lab)
    if q_emb.shape[1] != g_emb.shape[1]:
        raise ValueError("Query and gallery embeddings must have the same dimensionality")
    if l2_normalize:
        q_emb = q_emb / np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12)
        g_emb = g_emb / np.maximum(np.linalg.norm(g_emb, axis=1, keepdims=True), 1e-12)
    nq, ng = len(q_lab), len(g_lab)
    ap_all = np.full(nq, np.nan)
    pk_all = {k: np.full(nq, np.nan) for k in ks}

    for s in range(0, nq, chunk):
        e = min(nq, s + chunk)
        D = pairwise_sq_euclidean(q_emb[s:e], g_emb)
        if exclude_idx is not None:
            D[np.arange(e - s), np.asarray(exclude_idx)[s:e]] = np.inf
        order = np.argsort(D, axis=1, kind="stable")
        valid = np.isfinite(np.take_along_axis(D, order, axis=1))
        rel = (g_lab[order] == q_lab[s:e, None]) & valid
        n_rel = rel.sum(axis=1)
        csum = np.cumsum(rel, axis=1)
        ranks = np.arange(1, ng + 1)[None, :]
        prec = csum / ranks
        with np.errstate(invalid="ignore", divide="ignore"):
            ap = (prec * rel).sum(axis=1) / n_rel
        ap[n_rel == 0] = np.nan
        ap_all[s:e] = ap
        for k in ks:
            kk = min(k, ng)
            pk = rel[:, :kk].sum(axis=1) / float(kk)
            pk[n_rel == 0] = np.nan
            pk_all[k][s:e] = pk

    ok = ~np.isnan(ap_all)
    per_class = {int(c): float(np.mean(ap_all[ok & (q_lab == c)])) for c in np.unique(q_lab[ok])}
    res = {"mAP": float(np.mean(ap_all[ok])) if ok.any() else float("nan"),
           "n_queries": int(ok.sum()), "n_skipped": int((~ok).sum()),
           "per_class_mAP": per_class}
    for k in ks:
        res[f"P@{k}"] = float(np.mean(pk_all[k][ok])) if ok.any() else float("nan")
    return res


def evaluate_all_directions(emb_rgb, emb_sar, labels, ks=DEFAULT_KS, l2_normalize=False,
                            chance_seed: int = 0) -> dict:
    """RGB->SAR, SAR->RGB (headline), plus RGB->RGB, SAR->SAR, and a random-embedding baseline.

    emb_rgb[i] and emb_sar[i] must belong to the same instance pair i.
    """
    n = len(labels)
    idx = np.arange(n)
    kw = dict(ks=ks, l2_normalize=l2_normalize)
    res = {
        "RGB->SAR": evaluate_retrieval(emb_rgb, labels, emb_sar, labels, **kw),
        "SAR->RGB": evaluate_retrieval(emb_sar, labels, emb_rgb, labels, **kw),
        "RGB->SAR (strict)": evaluate_retrieval(emb_rgb, labels, emb_sar, labels,
                                                exclude_idx=idx, **kw),
        "SAR->RGB (strict)": evaluate_retrieval(emb_sar, labels, emb_rgb, labels,
                                                exclude_idx=idx, **kw),
        "RGB->RGB": evaluate_retrieval(emb_rgb, labels, emb_rgb, labels, exclude_idx=idx, **kw),
        "SAR->SAR": evaluate_retrieval(emb_sar, labels, emb_sar, labels, exclude_idx=idx, **kw),
    }
    rng = np.random.default_rng(chance_seed)
    d = np.asarray(emb_rgb).shape[1]
    ra, rb = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    res["chance (random embeddings, RGB->SAR)"] = evaluate_retrieval(ra, labels, rb, labels, **kw)
    return res


def format_results_table(res: dict, ks=DEFAULT_KS) -> str:
    cols = ["mAP"] + [f"P@{k}" for k in ks]
    head = f"{'Direction':<40}" + "".join(f"{c:>9}" for c in cols) + f"{'#queries':>10}"
    lines = [head, "-" * len(head)]
    for name, r in res.items():
        lines.append(f"{name:<40}" + "".join(f"{r[c]:>9.4f}" for c in cols) + f"{r['n_queries']:>10d}")
    return "\n".join(lines)


def extract_embeddings(model, data: dict, device, batch_size: int = 256, amp: bool = False):
    """Pair-free embeddings for all RGB and SAR items -> (emb_rgb, emb_sar) float32 arrays."""
    import torch

    model.eval()
    use_amp = bool(amp) and device.type == "cuda"
    out_r, out_s = [], []
    n = len(data["labels"])
    with torch.no_grad():
        for s in range(0, n, batch_size):
            rgb = torch.from_numpy(data["rgb"][s:s + batch_size]).to(device)
            sar = torch.from_numpy(data["sar"][s:s + batch_size]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                er = model.embed_rgb(rgb)
                es = model.embed_sar(sar)
            out_r.append(er.float().cpu().numpy())
            out_s.append(es.float().cpu().numpy())
    return np.concatenate(out_r), np.concatenate(out_s)


def evaluate_model(model, data: dict, device, batch_size: int = 256, amp: bool = False,
                   l2_normalize: bool = False, ks=DEFAULT_KS):
    """Embed the whole (paired) split with ``model`` and evaluate every direction."""
    emb_rgb, emb_sar = extract_embeddings(model, data, device, batch_size, amp)
    res = evaluate_all_directions(emb_rgb, emb_sar, data["labels"], ks=ks, l2_normalize=l2_normalize)
    return res, (emb_rgb, emb_sar)
