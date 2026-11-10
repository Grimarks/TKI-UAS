"""Dua indeks yang dipakai sistem hibrida: vector index (Jalur A) dan keyphrase index (Jalur B).

Peran di pipeline (Gambar 1)
    Fase 1 — Jalur A: "Vector index" (`DenseIndex`).        Fase 2 — Jalur A: "Skor semantik".
    Fase 1 — Jalur B: "Keyphrase index" (`KeyphraseIndex`). Fase 2 — Jalur B: "Skor keyphrase".

Masukan  : vektor dokumen dari encoder (Jalur A); daftar (keyphrase, bobot) per dokumen (Jalur B).
Keluaran : matriks skor berbentuk (Q query, N dokumen) untuk tiap jalur.

Rumus (nomor persamaan sesuai laporan)
    (1) s_sem(q, d) = cos(e_q, e_d) = e_q · e_d            vektor sudah bernorma L2 = 1
    (2) idf(u) = ln((N + 1) / (df(u) + 1)) + 1              u = unit pencocokan (`text.phrase_units`)
    (3) w_{d,u} = b_{d,u} · idf(u)                          b_{d,u} = bobot tertinggi keyphrase d yang memuat u
    (4) s_kp(q, d) = (w_q · w_d) / (‖w_q‖ ‖w_d‖)            baris w_d sudah dinormalkan L2 saat build

`KeyphraseIndex` juga menyediakan kemiripan Jaccard atas himpunan unit (alternatif dari penelitian acuan
Sarwar dkk. [7]); konfigurasi final memakai cosine.

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


# ====================================================================== Jalur A: Vector index
class DenseIndex:
    """Matriks (N dokumen × d dimensi) vektor ternormalisasi; pencarian eksak dengan perkalian matriks."""

    def __init__(self, ids: list[str], matrix: np.ndarray) -> None:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("bentuk matriks dense tidak cocok dengan jumlah id")
        self.ids = ids
        self.matrix = matrix

    @property
    def dim(self) -> int:
        """Dimensi embedding (384 untuk all-MiniLM-L6-v2)."""
        return int(self.matrix.shape[1])

    def scores(self, query_vecs: np.ndarray) -> np.ndarray:
        """Persamaan (1) untuk semua pasangan sekaligus: (Q × d) @ (d × N) = skor cosine (Q, N)."""
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"dimensi query {q.shape} tidak cocok dengan indeks (dim={self.dim})")
        return q @ self.matrix.T

    def save(self, directory: Path) -> None:
        """Tulis dense.npy (matriks) dan dense_ids.json (urutan id dokumen)."""
        np.save(directory / "dense.npy", self.matrix, allow_pickle=False)
        (directory / "dense_ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        """Baca indeks yang ditulis `save`."""
        matrix = np.load(directory / "dense.npy", allow_pickle=False)
        ids = json.loads((directory / "dense_ids.json").read_text(encoding="utf-8"))
        return cls(ids, matrix)


# ====================================================================== Jalur B: Keyphrase index
def unit_weights(keyphrases: Sequence[Keyphrase], language: str, unit: str) -> dict[str, float]:
    """Pecah keyphrase menjadi unit pencocokan; tiap unit diberi bobot keyphrase TERTINGGI yang memuatnya.

    Ini b_{d,u} pada persamaan (3). Dipakai sama persis untuk dokumen (saat build) dan query (saat search).
    Contoh: [('heart attack', 1.0), ('attack rate', 0.8)] -> t:attack mendapat 1.0, bukan 0.8.
    """
    weights: dict[str, float] = {}
    for phrase, w in keyphrases:
        for u in phrase_units(phrase, language, unit):
            if w > weights.get(u, 0.0):
                weights[u] = float(w)
    return weights


def keyphrase_idf(doc_freq: np.ndarray, n_docs: int) -> np.ndarray:
    """Persamaan (2): idf(u) = ln((N + 1) / (df(u) + 1)) + 1 (versi 'smooth', selalu > 0)."""
    return np.log((n_docs + 1) / (doc_freq + 1)) + 1.0


class KeyphraseIndex:
    """Matriks jarang (N dokumen × V unit) berisi w_{d,u} = b_{d,u}·idf(u), tiap baris dinormalkan L2."""

    def __init__(self, vocab: list[str], idf: np.ndarray, matrix: sp.csr_matrix,
                 language: str, unit: str, similarity: str) -> None:
        self.vocab = vocab
        self.term_index = {u: i for i, u in enumerate(vocab)}
        self.idf = np.asarray(idf, dtype=np.float32)
        self.matrix = matrix.tocsr().astype(np.float32)
        self.language = language
        self.unit = unit
        self.similarity = similarity
        # Versi biner (unit ada/tidak) hanya dipakai kemiripan Jaccard.
        binary = self.matrix.copy()
        binary.data[:] = 1.0
        self._binary = binary
        self._doc_sizes = np.asarray(binary.sum(axis=1)).ravel().astype(np.float32)

    # ------------------------------------------------------------------ build (Fase 1)
    @classmethod
    def build(cls, doc_keyphrases: Sequence[Sequence[Keyphrase]], language: str,
              unit: str, similarity: str) -> "KeyphraseIndex":
        """Bangun indeks dari keyphrase semua dokumen: unit → df → idf (2) → w_{d,u} (3) → normalisasi L2."""
        per_doc = [unit_weights(kps, language, unit) for kps in doc_keyphrases]

        df: Counter[str] = Counter()
        for d in per_doc:
            df.update(d.keys())
        vocab = sorted(df)
        term_index = {u: i for i, u in enumerate(vocab)}
        n_docs = len(per_doc)
        idf = keyphrase_idf(np.array([df[u] for u in vocab], dtype=np.float64), n_docs)

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

    # ------------------------------------------------------------------ query side (Fase 2)
    def query_units(self, keyphrases: Sequence[Keyphrase]) -> dict[str, float]:
        """Unit pencocokan query beserta bobotnya (aturan sama dengan dokumen)."""
        return unit_weights(keyphrases, self.language, self.unit)

    def scores(self, query_keyphrases: Sequence[Sequence[Keyphrase]]) -> np.ndarray:
        """Skor tiap query terhadap semua dokumen: bentuk (Q, N). Query tanpa unit dikenal -> skor 0."""
        n_q = len(query_keyphrases)
        n_docs = self.matrix.shape[0]
        out = np.zeros((n_q, n_docs), dtype=np.float32)
        for qi, kps in enumerate(query_keyphrases):
            units = self.query_units(kps)
            # unit query yang tidak ada di kosakata indeks diabaikan (tidak punya idf)
            known = [(self.term_index[u], w) for u, w in units.items() if u in self.term_index]
            if not known:
                continue
            if self.similarity == "cosine":
                row = self._cosine_row(known)
                if row is not None:
                    out[qi] = row
            else:
                out[qi] = self._jaccard_row(known, len(units))
        return out

    def _cosine_row(self, known: list[tuple[int, float]]) -> np.ndarray | None:
        """Persamaan (4) untuk satu query: w_q = b_{q,u}·idf(u), dinormalkan L2, lalu M @ w_q."""
        cols = np.array([c for c, _ in known], dtype=np.int64)
        vals = np.array([w * float(self.idf[c]) for c, w in known], dtype=np.float32)
        norm = float(np.linalg.norm(vals))
        if norm == 0.0:
            return None
        qvec = sp.csr_matrix((vals / norm, (np.zeros_like(cols), cols)),
                             shape=(1, self.matrix.shape[1]), dtype=np.float32)
        return (self.matrix @ qvec.T).toarray().ravel()

    def _jaccard_row(self, known: list[tuple[int, float]], n_query_units: int) -> np.ndarray:
        """Jaccard |U_q ∩ U_d| / |U_q ∪ U_d|; unit query tak dikenal tetap dihitung di ukuran query."""
        cols = np.array([c for c, _ in known], dtype=np.int64)
        qbin = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (np.zeros_like(cols), cols)),
                             shape=(1, self.matrix.shape[1]), dtype=np.float32)
        inter = (self._binary @ qbin.T).toarray().ravel()
        union = self._doc_sizes + float(n_query_units) - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, inter / union, 0.0)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        """Tulis kp_matrix.npz, kp_idf.npy, dan kp_vocab.json."""
        sp.save_npz(directory / "kp_matrix.npz", self.matrix, compressed=True)
        np.save(directory / "kp_idf.npy", self.idf, allow_pickle=False)
        (directory / "kp_vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, language: str, unit: str, similarity: str) -> "KeyphraseIndex":
        """Baca indeks yang ditulis `save` dan periksa ukurannya konsisten."""
        matrix = sp.load_npz(directory / "kp_matrix.npz")
        idf = np.load(directory / "kp_idf.npy", allow_pickle=False)
        vocab = json.loads((directory / "kp_vocab.json").read_text(encoding="utf-8"))
        if len(vocab) != matrix.shape[1] or len(idf) != matrix.shape[1]:
            raise ValueError("indeks keyphrase rusak: ukuran vocab/idf tidak cocok dengan matriks")
        return cls(vocab, idf, matrix, language, unit, similarity)


def _row_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
    """Bagi tiap baris dengan norma L2-nya (baris nol dibiarkan nol)."""
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms).dot(mat).tocsr()
