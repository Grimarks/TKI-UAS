"""BM25 sebagai pembanding leksikal (baseline) pada evaluasi.

Implementasi kecil dan transparan: bobot BM25 tiap pasangan (dokumen, term) dihitung sekali di
awal, lalu skor query = jumlah bobot term query. Memakai tokenisasi + stemming yang sama dengan
bagian lain proyek agar perbandingannya adil.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from .text import content_tokens


class BM25:
    def __init__(self, language: str = "en", k1: float = 1.2, b: float = 0.75) -> None:
        self.language = language
        self.k1 = k1
        self.b = b
        self.term_index: dict[str, int] = {}
        self.matrix: sp.csr_matrix | None = None   # (N, V) bobot BM25 tiap dokumen-term

    def fit(self, texts: Sequence[str]) -> "BM25":
        docs = [Counter(content_tokens(t, self.language)) for t in texts]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        vocab = sorted(df)
        self.term_index = {t: i for i, t in enumerate(vocab)}
        n = len(docs)
        lengths = np.array([sum(d.values()) for d in docs], dtype=np.float64)
        avgdl = max(float(lengths.mean()) if n else 1.0, 1e-9)
        idf = np.log(1.0 + (n - np.array([df[t] for t in vocab]) + 0.5) / (np.array([df[t] for t in vocab]) + 0.5))
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            norm = self.k1 * (1.0 - self.b + self.b * lengths[r] / avgdl)
            for term, tf in d.items():
                c = self.term_index[term]
                rows.append(r)
                cols.append(c)
                vals.append(float(idf[c]) * tf * (self.k1 + 1.0) / (tf + norm))
        self.matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
        return self

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        """Skor (Q, N). Term query yang tidak ada di korpus diabaikan."""
        if self.matrix is None:
            raise RuntimeError("panggil fit() lebih dulu")
        out = np.zeros((len(queries), self.matrix.shape[0]), dtype=np.float32)
        for qi, q in enumerate(queries):
            cols = sorted({self.term_index[t] for t in content_tokens(q, self.language) if t in self.term_index})
            if cols:
                out[qi] = np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()
        return out
