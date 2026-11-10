"""Penggabungan (fusi) skor semantik dan skor keyphrase, plus peringkat satu sinyal.

Fusi hanya satu strategi, kombinasi linear: skor tiap sinyal dinormalkan min-max pada himpunan
kandidat, lalu  s = alpha * s_sem + (1 - alpha) * s_kp   (terinspirasi kombinasi linear Relwt di A1).
BM25 tidak ikut difusikan; ia hanya diperingkat sendiri sebagai pembanding (mode `bm25`).

Kandidat = gabungan top-C tiap sinyal. Dokumen di luar kandidat tidak ikut diperingkat.
Semua fungsi menerima larik skor untuk SATU query (panjang N dokumen) dan deterministik:
skor seri diurutkan menurut indeks dokumen.
"""
from __future__ import annotations

import numpy as np


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indeks k skor terbesar, terurut menurun; seri diputus oleh indeks terkecil."""
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    order = np.lexsort((part, -scores[part]))
    return part[order].astype(np.int64)


def positive_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k hanya di antara dokumen yang skornya > 0 (agar dokumen tanpa kecocokan tidak jadi kandidat)."""
    pos = np.flatnonzero(scores > 0)
    if pos.size == 0:
        return np.empty(0, dtype=np.int64)
    sub = topk_indices(scores[pos], k)
    return pos[sub]


def minmax(values: np.ndarray) -> np.ndarray:
    """Normalisasi min-max ke [0, 1]. Bila semua nilai sama (tak ada informasi), hasilnya nol."""
    if values.size == 0:
        return values.astype(np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def candidate_set(sem: np.ndarray, kp: np.ndarray, candidate_k: int) -> np.ndarray:
    parts = [topk_indices(sem, candidate_k), positive_topk(kp, candidate_k)]
    return np.unique(np.concatenate(parts))


def fuse_weighted(sem: np.ndarray, kp: np.ndarray, alpha: float, candidate_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan (indeks_kandidat, skor_fusi) terurut menurun."""
    cand = candidate_set(sem, kp, candidate_k)
    fused = alpha * minmax(sem[cand]) + (1.0 - alpha) * minmax(kp[cand])
    order = np.lexsort((cand, -fused))
    return cand[order], fused[order]


def rank_single(scores: np.ndarray, k: int, positive_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    idx = positive_topk(scores, k) if positive_only else topk_indices(scores, k)
    return idx, scores[idx]


def rank_documents(sem: np.ndarray, kp: np.ndarray, mode: str, alpha: float, candidate_k: int, top_k: int,
                   bm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Titik masuk tunggal: kembalikan (indeks_dokumen, skor) teratas untuk mode yang diminta.

    Mode `bm25` memerlukan `bm` (skor BM25 query itu).
    """
    if mode == "bm25" and bm is None:
        raise ValueError("mode bm25 memerlukan skor BM25")
    if mode == "semantic":
        idx, vals = rank_single(sem, top_k)
    elif mode == "keyphrase":
        idx, vals = rank_single(kp, top_k, positive_only=True)
    elif mode == "bm25":
        idx, vals = rank_single(bm, top_k, positive_only=True)
    elif mode == "hybrid":
        idx, vals = fuse_weighted(sem, kp, alpha, max(candidate_k, top_k))
    else:
        raise ValueError(f"mode tidak dikenal: {mode}")
    return idx[:top_k], vals[:top_k]
