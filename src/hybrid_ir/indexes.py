"""Dua indeks yang dipakai sistem hibrida.

DenseIndex     : matriks (N dokumen x d dimensi) vektor ternormalisasi. Skor = cosine = dot product.
KeyphraseIndex : matriks jarang (N dokumen x V unit) berbobot (bobot keyphrase x IDF), tiap baris
                 dinormalkan L2. Skor = cosine atas vektor bobot keyphrase (A2: "Cosine"), atau
                 Jaccard atas himpunan unit (A2: "Jaccard").

Penyimpanan memakai format .npy/.npz/.json saja. Kita TIDAK memakai pickle, karena memuat pickle
dari berkas yang tidak tepercaya dapat mengeksekusi kode sembarang.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from .text import phrase_units

Keyphrase = tuple[str, float]


class DenseIndex:
    def __init__(self, ids: list[str], matrix: np.ndarray) -> None:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("bentuk matriks dense tidak cocok dengan jumlah id")
        self.ids = ids
        self.matrix = matrix

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def scores(self, query_vecs: np.ndarray) -> np.ndarray:
        """Skor cosine tiap query terhadap semua dokumen: bentuk (Q, N)."""
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"dimensi query {q.shape} tidak cocok dengan indeks (dim={self.dim})")
        return q @ self.matrix.T

    def save(self, directory: Path) -> None:
        np.save(directory / "dense.npy", self.matrix, allow_pickle=False)
        (directory / "dense_ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        matrix = np.load(directory / "dense.npy", allow_pickle=False)
        ids = json.loads((directory / "dense_ids.json").read_text(encoding="utf-8"))
        return cls(ids, matrix)


class KeyphraseIndex:
    def __init__(self, vocab: list[str], idf: np.ndarray, matrix: sp.csr_matrix,
                 language: str, unit: str, similarity: str) -> None:
        self.vocab = vocab
        self.term_index = {u: i for i, u in enumerate(vocab)}
        self.idf = np.asarray(idf, dtype=np.float32)
        self.matrix = matrix.tocsr().astype(np.float32)
        self.language = language
        self.unit = unit
        self.similarity = similarity
        binary = self.matrix.copy()
        binary.data[:] = 1.0
        self._binary = binary
        self._doc_sizes = np.asarray(binary.sum(axis=1)).ravel().astype(np.float32)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, doc_keyphrases: Sequence[Sequence[Keyphrase]], language: str,
              unit: str, similarity: str) -> "KeyphraseIndex":
        per_doc: list[dict[str, float]] = []
        for kps in doc_keyphrases:
            weights: dict[str, float] = {}
            for phrase, w in kps:
                for u in phrase_units(phrase, language, unit):
                    if w > weights.get(u, 0.0):
                        weights[u] = float(w)
            per_doc.append(weights)

        df: Counter[str] = Counter()
        for d in per_doc:
            df.update(d.keys())
        vocab = sorted(df)
        term_index = {u: i for i, u in enumerate(vocab)}
        n_docs = len(per_doc)
        idf = np.log((n_docs + 1) / (np.array([df[u] for u in vocab], dtype=np.float64) + 1)) + 1.0

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for r, d in enumerate(per_doc):
            for u, w in d.items():
                c = term_index[u]
                rows.append(r)
                cols.append(c)
                vals.append(w * float(idf[c]))
        mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_docs, len(vocab)), dtype=np.float32)
        mat = _row_normalize(mat)
        return cls(vocab, idf.astype(np.float32), mat, language, unit, similarity)

    # ------------------------------------------------------------------ query side
    def query_units(self, keyphrases: Sequence[Keyphrase]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for phrase, w in keyphrases:
            for u in phrase_units(phrase, self.language, self.unit):
                if w > weights.get(u, 0.0):
                    weights[u] = float(w)
        return weights

    def scores(self, query_keyphrases: Sequence[Sequence[Keyphrase]]) -> np.ndarray:
        """Skor tiap query terhadap semua dokumen: bentuk (Q, N). Query tanpa unit dikenal -> skor 0."""
        n_q = len(query_keyphrases)
        n_docs = self.matrix.shape[0]
        out = np.zeros((n_q, n_docs), dtype=np.float32)
        for qi, kps in enumerate(query_keyphrases):
            units = self.query_units(kps)
            known = [(self.term_index[u], w) for u, w in units.items() if u in self.term_index]
            if not known:
                continue
            cols = np.array([c for c, _ in known], dtype=np.int64)
            if self.similarity == "cosine":
                vals = np.array([w * float(self.idf[c]) for c, w in known], dtype=np.float32)
                norm = float(np.linalg.norm(vals))
                if norm == 0.0:
                    continue
                qvec = sp.csr_matrix((vals / norm, (np.zeros_like(cols), cols)),
                                     shape=(1, self.matrix.shape[1]), dtype=np.float32)
                out[qi] = (self.matrix @ qvec.T).toarray().ravel()
            else:  # jaccard atas himpunan unit; unit tak dikenal tetap dihitung di ukuran query
                qbin = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (np.zeros_like(cols), cols)),
                                     shape=(1, self.matrix.shape[1]), dtype=np.float32)
                inter = (self._binary @ qbin.T).toarray().ravel()
                union = self._doc_sizes + float(len(units)) - inter
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[qi] = np.where(union > 0, inter / union, 0.0)
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        sp.save_npz(directory / "kp_matrix.npz", self.matrix, compressed=True)
        np.save(directory / "kp_idf.npy", self.idf, allow_pickle=False)
        (directory / "kp_vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, language: str, unit: str, similarity: str) -> "KeyphraseIndex":
        matrix = sp.load_npz(directory / "kp_matrix.npz")
        idf = np.load(directory / "kp_idf.npy", allow_pickle=False)
        vocab = json.loads((directory / "kp_vocab.json").read_text(encoding="utf-8"))
        if len(vocab) != matrix.shape[1] or len(idf) != matrix.shape[1]:
            raise ValueError("indeks keyphrase rusak: ukuran vocab/idf tidak cocok dengan matriks")
        return cls(vocab, idf, matrix, language, unit, similarity)


def _row_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms).dot(mat).tocsr()
