"""BM25 sebagai pembanding leksikal (baseline). TIDAK ikut difusikan ke skor hybrid.

Peran di pipeline (Gambar 1)
    Pembanding — Fase 1: "Token + stemming" → "BM25 index" (dibangun di memori saat pertama dipakai).
    Pembanding — Fase 2: "Token query" → "Skor BM25" → langsung ke "Skenario" BM25 di Fase 3.

Masukan  : teks dokumen ('judul. abstrak') dan teks query.
Keluaran : matriks skor (Q, N).

Rumus (persamaan (5) di laporan; k1 = 1.2, b = 0.75)
    s_bm25(q, d) = Σ_{t ∈ q} idf(t) · f(t,d)·(k1 + 1) / (f(t,d) + k1·(1 − b + b·|d|/avgdl))
    idf(t)       = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

Implementasinya kecil dan transparan: bobot setiap pasangan (dokumen, term) dihitung sekali di awal
(`fit`), lalu skor query cukup berupa jumlah bobot term query (`scores`). Tokenisasi, stopword, dan
stemming sama dengan unit token keyphrase (`text.content_tokens`), supaya perbandingannya adil.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from .text import content_tokens


def bm25_idf(doc_freq: np.ndarray, n_docs: int) -> np.ndarray:
    """idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5)) — versi yang selalu positif (seperti Lucene)."""
    return np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


class BM25:
    """Indeks BM25 di memori: matriks jarang (N dokumen × V term) berisi bobot per pasangan."""

    def __init__(self, language: str = "en", k1: float = 1.2, b: float = 0.75) -> None:
        self.language = language
        self.k1 = k1
        self.b = b
        self.term_index: dict[str, int] = {}
        self.matrix: sp.csr_matrix | None = None   # (N, V) bobot BM25 tiap dokumen-term

    def fit(self, texts: Sequence[str]) -> "BM25":
        """Hitung bobot BM25 untuk setiap (dokumen, term) dari seluruh korpus."""
        docs = [Counter(content_tokens(t, self.language)) for t in texts]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        vocab = sorted(df)
        self.term_index = {t: i for i, t in enumerate(vocab)}
        n = len(docs)
        lengths = np.array([sum(d.values()) for d in docs], dtype=np.float64)   # |d|
        avgdl = max(float(lengths.mean()) if n else 1.0, 1e-9)
        idf = bm25_idf(np.array([df[t] for t in vocab]), n)
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            # penyebut bagian kanan (5) yang hanya bergantung pada panjang dokumen
            norm = self.k1 * (1.0 - self.b + self.b * lengths[r] / avgdl)
            for term, tf in d.items():
                c = self.term_index[term]
                rows.append(r)
                cols.append(c)
                vals.append(float(idf[c]) * tf * (self.k1 + 1.0) / (tf + norm))
        self.matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
        return self

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        """Skor (Q, N) = jumlah kolom term query. Term query yang tidak ada di korpus diabaikan.

        Term yang muncul dua kali di query dihitung sekali (himpunan term), sesuai Σ_{t ∈ q}.
        """
        if self.matrix is None:
            raise RuntimeError("panggil fit() lebih dulu")
        out = np.zeros((len(queries), self.matrix.shape[0]), dtype=np.float32)
        for qi, q in enumerate(queries):
            cols = sorted({self.term_index[t] for t in content_tokens(q, self.language) if t in self.term_index})
            if cols:
                out[qi] = np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()
        return out
