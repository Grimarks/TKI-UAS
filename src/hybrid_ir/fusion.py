"""Normalisasi dan fusi skor semantik + keyphrase, serta peringkat satu sinyal.

Peran di pipeline (Gambar 1)
    Fase 2 — kotak "Normalisasi + Fusi" (mode hybrid) dan "Ranking top-k" (semua mode).

Masukan  : larik skor untuk SATU query (panjang N dokumen): sem (Jalur A), kp (Jalur B), dan untuk
           mode pembanding, bm25.
Keluaran : (indeks dokumen, skor) terurut menurun.

Rumus (nomor persamaan sesuai laporan)
    Kandidat  R = top-C(sem) ∪ top-C(kp, hanya skor > 0)                       C = candidate_k = 100
    (6) s̃(d) = (s(d) − s_min) / (s_max − s_min)       min-max di dalam R, per jalur
    (7) s_hyb(q, d) = α·s̃_sem(q, d) + (1 − α)·s̃_kp(q, d),   0 ≤ α ≤ 1            (α final = 0.9)

Hanya ada satu strategi fusi, yaitu kombinasi linear (konveks); RRF tidak dipakai (alasan: Bruch dkk. [18]).
BM25 TIDAK ikut difusikan; ia hanya diperingkat sendiri sebagai pembanding (mode `bm25`).
Semua fungsi deterministik: skor seri diurutkan menurut indeks dokumen (indeks kecil dulu).
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ pemilihan top-k
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


# ------------------------------------------------------------------ Normalisasi + Fusi (mode hybrid)
def minmax(values: np.ndarray) -> np.ndarray:
    """Persamaan (6): normalisasi min-max ke [0, 1]. Bila semua nilai sama (tak ada informasi), hasilnya nol."""
    if values.size == 0:
        return values.astype(np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def candidate_set(sem: np.ndarray, kp: np.ndarray, candidate_k: int) -> np.ndarray:
    """Kandidat R = gabungan top-C semantik dan top-C keyphrase yang berskor positif (terurut menurut indeks)."""
    parts = [topk_indices(sem, candidate_k), positive_topk(kp, candidate_k)]
    return np.unique(np.concatenate(parts))


def fuse_weighted(sem: np.ndarray, kp: np.ndarray, alpha: float, candidate_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Persamaan (7) pada kandidat R. Kembalikan (indeks_kandidat, skor_fusi) terurut menurun."""
    cand = candidate_set(sem, kp, candidate_k)
    fused = alpha * minmax(sem[cand]) + (1.0 - alpha) * minmax(kp[cand])
    order = np.lexsort((cand, -fused))
    return cand[order], fused[order]


# ------------------------------------------------------------------ Ranking top-k (semua mode)
def rank_single(scores: np.ndarray, k: int, positive_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Peringkat satu sinyal saja (mode semantic, keyphrase, atau bm25)."""
    idx = positive_topk(scores, k) if positive_only else topk_indices(scores, k)
    return idx, scores[idx]


def rank_documents(sem: np.ndarray, kp: np.ndarray, mode: str, alpha: float, candidate_k: int, top_k: int,
                   bm25: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Titik masuk tunggal: kembalikan (indeks_dokumen, skor) teratas untuk mode yang diminta.

    semantic  : urut menurut s_sem
    keyphrase : urut menurut s_kp, hanya dokumen dengan s_kp > 0
    bm25      : urut menurut s_bm25, hanya dokumen dengan s_bm25 > 0 (memerlukan argumen `bm25`)
    hybrid    : fusi linear (7) pada kandidat R
    """
    if mode == "bm25" and bm25 is None:
        raise ValueError("mode bm25 memerlukan skor BM25")
    if mode == "semantic":
        idx, vals = rank_single(sem, top_k)
    elif mode == "keyphrase":
        idx, vals = rank_single(kp, top_k, positive_only=True)
    elif mode == "bm25":
        idx, vals = rank_single(bm25, top_k, positive_only=True)
    elif mode == "hybrid":
        idx, vals = fuse_weighted(sem, kp, alpha, max(candidate_k, top_k))
    else:
        raise ValueError(f"mode tidak dikenal: {mode}")
    return idx[:top_k], vals[:top_k]
