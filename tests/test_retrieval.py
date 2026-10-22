"""Retrieval / mAP tests. NumPy only. Expected values are computed by hand or by an
independent brute-force reference implementation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import evaluate_all_directions, evaluate_retrieval  # noqa: E402


def brute_force(q, ql, g, gl, exclude=None, ks=(10,)):
    """Slow, obviously-correct reference."""
    aps, p10 = [], []
    for i in range(len(q)):
        items = []
        for j in range(len(g)):
            if exclude is not None and j == exclude[i]:
                continue
            items.append((float(np.sum((q[i] - g[j]) ** 2)), j))
        items.sort()
        rel = [int(gl[j] == ql[i]) for _, j in items]
        if sum(rel) == 0:
            continue
        hits, ap = 0, 0.0
        for r, x in enumerate(rel, 1):
            if x:
                hits += 1
                ap += hits / r
        aps.append(ap / sum(rel))
        p10.append(sum(rel[:10]) / min(10, len(rel)))
    return float(np.mean(aps)), float(np.mean(p10))


def test_hand_computed_ap():
    # query at 0 (class 0); gallery points at 1,2,3,4 with classes 0,1,0,1
    q = np.array([[0.0]]); g = np.array([[1.0], [2.0], [3.0], [4.0]])
    r = evaluate_retrieval(q, [0], g, [0, 1, 0, 1], ks=(2,))
    # ranking: rel, non, rel, non -> AP = (1/1 + 2/3)/2
    assert abs(r["mAP"] - (1 + 2 / 3) / 2) < 1e-12
    assert abs(r["P@2"] - 0.5) < 1e-12
    assert r["n_queries"] == 1 and r["n_skipped"] == 0


def test_perfect_and_worst_ranking():
    q = np.array([[0.0]]); g = np.array([[1.0], [2.0], [9.0]])
    assert evaluate_retrieval(q, [0], g, [0, 0, 1])["mAP"] == 1.0
    worst = evaluate_retrieval(q, [0], g, [1, 1, 0])["mAP"]
    assert abs(worst - 1 / 3) < 1e-12


def test_self_exclusion_same_modality():
    emb = np.array([[0.0], [1.0], [3.0], [7.0]]); lab = [0, 0, 1, 1]
    idx = np.arange(4)
    r = evaluate_retrieval(emb, lab, emb, lab, exclude_idx=idx, ks=(1,))
    # q0: [g1 rel, g2, g3] -> AP 1 ; q1: [g0 rel,...] -> 1 ; q2: [g3 rel (d=4)? g1 d=2 non, g0 d=3 non, g3 d=4 rel]
    # q2 ranking: g1(2,non) g0(3,non) g3(4,rel) -> AP = 1/3 ; q3: g2(4,rel) first -> 1
    assert abs(r["mAP"] - (1 + 1 + 1 / 3 + 1) / 4) < 1e-12
    # without exclusion every query would retrieve itself first -> mAP strictly larger
    r2 = evaluate_retrieval(emb, lab, emb, lab, ks=(1,))
    assert r2["mAP"] > r["mAP"]


def test_matches_bruteforce_random():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((37, 8)); g = rng.standard_normal((53, 8))
    ql = rng.integers(0, 4, 37); gl = rng.integers(0, 4, 53)
    ex = rng.integers(0, 53, 37)
    for e in (None, ex):
        ref_map, ref_p = brute_force(q, ql, g, gl, e)
        r = evaluate_retrieval(q, ql, g, gl, exclude_idx=e, ks=(10,), chunk=8)
        assert abs(r["mAP"] - ref_map) < 1e-10
        assert abs(r["P@10"] - ref_p) < 1e-10


def test_queries_without_relevant_items_are_skipped():
    q = np.array([[0.0], [5.0]]); g = np.array([[1.0], [2.0]])
    r = evaluate_retrieval(q, [0, 7], g, [0, 0])
    assert r["n_queries"] == 1 and r["n_skipped"] == 1 and r["mAP"] == 1.0


def test_all_directions_perfect_and_chance():
    n_per, C = 40, 10
    labels = np.repeat(np.arange(C), n_per)
    onehot = np.eye(C)[labels] * 5.0
    rng = np.random.default_rng(1)
    emb_r = onehot + 0.01 * rng.standard_normal(onehot.shape)
    emb_s = onehot + 0.01 * rng.standard_normal(onehot.shape)
    res = evaluate_all_directions(emb_r, emb_s, labels)
    for k in ("RGB->SAR", "SAR->RGB", "RGB->SAR (strict)", "SAR->RGB (strict)", "RGB->RGB", "SAR->SAR"):
        assert res[k]["mAP"] > 0.999, k
    chance = res["chance (random embeddings, RGB->SAR)"]["mAP"]
    assert abs(chance - 0.1) < 0.03  # ~ class prior for 10 balanced classes


def test_strict_mode_removes_paired_counterpart():
    # If embeddings of the two modalities are identical, the paired item is always rank 1
    # in standard mode. Strict mode must not give it that free hit.
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 5, 100)
    emb = rng.standard_normal((100, 16))
    res = evaluate_all_directions(emb, emb, labels)
    assert res["RGB->SAR"]["mAP"] > res["RGB->SAR (strict)"]["mAP"]
    assert res["RGB->SAR"]["P@10"] > res["RGB->SAR (strict)"]["P@10"]


def test_both_directions_differ_when_modalities_differ():
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(4), 25)
    emb_r = np.eye(4)[labels] * 3 + 0.1 * rng.standard_normal((100, 4))
    emb_s = rng.standard_normal((100, 4))  # SAR embeddings carry no class information
    res = evaluate_all_directions(emb_r, emb_s, labels)
    assert res["RGB->RGB"]["mAP"] > 0.9
    assert res["SAR->SAR"]["mAP"] < 0.5
