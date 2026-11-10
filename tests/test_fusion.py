import numpy as np
import pytest

from hybrid_ir import fusion
from hybrid_ir.fusion import fuse_weighted, minmax, positive_topk, rank_documents, topk_indices


def test_topk_order_and_ties():
    s = np.array([0.1, 0.9, 0.9, 0.5])
    assert list(topk_indices(s, 3)) == [1, 2, 3]       # seri 0.9 -> indeks kecil dulu
    assert list(topk_indices(s, 10)) == [1, 2, 3, 0]


def test_positive_topk_ignores_zeros():
    assert list(positive_topk(np.array([0, 0.2, 0, 0.5]), 5)) == [3, 1]
    assert positive_topk(np.zeros(4), 3).size == 0


def test_minmax():
    assert list(minmax(np.array([2.0, 4.0, 3.0]))) == [0.0, 1.0, 0.5]
    assert not minmax(np.array([1.0, 1.0])).any()


def test_alpha_extremes_match_single_signals():
    sem = np.array([0.9, 0.5, 0.1, 0.4])
    kp = np.array([0.0, 0.1, 0.9, 0.5])
    idx1, _ = fuse_weighted(sem, kp, 1.0, 4)
    idx0, _ = fuse_weighted(sem, kp, 0.0, 4)
    assert idx1[0] == 0 and idx0[0] == 2


def test_weighted_mixes_signals():
    sem = np.array([1.0, 0.0, 0.8])
    kp = np.array([0.0, 1.0, 0.9])
    idx, vals = fuse_weighted(sem, kp, 0.5, 3)
    assert idx[0] == 2 and vals[0] > vals[1]


def test_keyphrase_mode_returns_only_matches():
    sem = np.array([0.5, 0.4, 0.3])
    kp = np.array([0.0, 0.0, 0.2])
    idx, _ = rank_documents(sem, kp, "keyphrase", 0.5, 10, 5)
    assert list(idx) == [2]


def test_all_zero_keyphrase_falls_back_to_semantic_order():
    sem = np.array([0.2, 0.9, 0.5])
    idx, _ = rank_documents(sem, np.zeros(3), "hybrid", 0.6, 10, 3)
    assert list(idx) == [1, 2, 0]


def test_unknown_mode():
    with pytest.raises(ValueError):
        rank_documents(np.zeros(2), np.zeros(2), "x", 0.5, 5, 2)
    with pytest.raises(ValueError, match="tidak dikenal"):
        rank_documents(np.zeros(2), np.zeros(2), "hybrid_bm25", 0.5, 5, 2, bm=np.zeros(2))


def test_bm25_mode_ranks_only_bm25_matches():
    sem = np.array([0.9, 0.5, 0.1])
    bm = np.array([0.0, 2.0, 5.0])
    idx, vals = rank_documents(sem, np.zeros(3), "bm25", 0.5, 10, 5, bm=bm)
    assert list(idx) == [2, 1] and list(vals) == [5.0, 2.0]
    with pytest.raises(ValueError, match="BM25"):
        rank_documents(sem, np.zeros(3), "bm25", 0.5, 10, 5)


def test_hybrid_mode_ignores_bm25():
    rng = np.random.default_rng(0)
    sem, kp, bm = rng.random(30), rng.random(30), rng.random(30)
    a = rank_documents(sem, kp, "hybrid", 0.6, 10, 10)
    b = rank_documents(sem, kp, "hybrid", 0.6, 10, 10, bm=bm)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_removed_fusion_functions_are_gone():
    for name in ("fuse_rrf", "fuse_rrf3", "candidate_set3", "fuse_weighted3", "weight_grid"):
        assert not hasattr(fusion, name)
