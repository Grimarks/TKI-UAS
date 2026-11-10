"""Encoder embedding: mengubah teks menjadi vektor ternormalisasi (norma L2 = 1).

Peran di pipeline (Gambar 1)
    Fase 1 — Jalur A: "A. Encoder SBERT" (all-MiniLM-L6-v2, 1 vektor per dokumen) → "Vector index".
    Fase 2 — Jalur A: "A. Encode query" (encoder yang sama) → "Skor semantik".

Masukan  : daftar teks (dokumen 'judul. abstrak' atau query).
Keluaran : matriks float32 berbentuk (jumlah teks, d), tiap baris bernorma L2 = 1 (d = 384 untuk MiniLM).

Rumus
    Normalisasi L2: e ← e / ‖e‖. Karena itu cosine pada persamaan (1),
        s_sem(q, d) = cos(e_q, e_d) = (e_q · e_d) / (‖e_q‖ ‖e_d‖),
    cukup dihitung sebagai perkalian dalam e_q · e_d (lihat `indexes.DenseIndex.scores`).

Dua backend:
  * SentenceTransformerEncoder : model sungguhan (bawaan all-MiniLM-L6-v2). Dipakai untuk hasil nyata.
  * HashingEncoder             : pengganti deterministik tanpa unduhan/GPU. BUKAN model semantik;
                                 hanya untuk tes otomatis dan demo offline.
"""
from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .config import EncoderConfig
from .text import content_tokens


class EncoderUnavailable(RuntimeError):
    """Backend encoder tidak dapat dipakai (paket belum terpasang, dsb)."""


class Encoder(Protocol):
    """Antarmuka bersama semua encoder."""

    dim: int

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Vektor dokumen bernorma L2 = 1, bentuk (N, dim)."""
        ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Vektor query bernorma L2 = 1, bentuk (Q, dim)."""
        ...

    def fingerprint(self) -> dict:
        """Identitas encoder untuk meta.json."""
        ...


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Bagi tiap baris dengan normanya (batas bawah 1e-12 agar vektor nol tidak membuat pembagian nol)."""
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def resolve_device(requested: str) -> str:
    """'auto' memilih cuda, lalu mps (Apple Silicon), lalu cpu."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class HashingEncoder:
    """Bag-of-words + bigram yang di-hash ke dimensi tetap. Deterministik antar-proses (crc32)."""

    def __init__(self, cfg: EncoderConfig, language: str = "en") -> None:
        self.cfg = cfg
        self.dim = cfg.hashing_dim
        self.language = language

    def _vector(self, text: str) -> np.ndarray:
        toks = content_tokens(text, self.language)
        feats = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, tf in feats.items():
            h = zlib.crc32(feat.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(tf))
        return vec

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.stack([self._vector(t) for t in texts]))

    encode_documents = _encode
    encode_queries = _encode

    def fingerprint(self) -> dict:
        """Identitas encoder hashing (untuk pemeriksaan kecocokan indeks)."""
        return {"backend": "hashing", "dim": self.dim, "language": self.language}


class SentenceTransformerEncoder:
    """Encoder SBERT (bi-encoder): dokumen dan query dienkode terpisah dengan model yang sama."""

    def __init__(self, cfg: EncoderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - bergantung lingkungan
            raise EncoderUnavailable(
                "sentence-transformers belum terpasang. Jalankan: pip install 'hybrid-ir[models]'"
            ) from exc
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        kwargs: dict = {"device": self.device, "trust_remote_code": False}  # tidak pernah menjalankan kode dari hub
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        self.model = SentenceTransformer(cfg.model_name, **kwargs)
        self.model.max_seq_length = cfg.max_seq_length
        if self.device == "cuda" and cfg.fp16:
            self.model.half()
        # Dimensi diambil dari keluaran nyata agar tidak bergantung pada nama metode antar-versi.
        self.dim = int(self._encode(["dimensi"], "").shape[1])

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, getattr(self, "dim", 1)), dtype=np.float32)
        batch = [prefix + t for t in texts]
        emb = self.model.encode(
            batch,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(batch) > 512,
        )
        return _l2_normalize(emb)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Vektor dokumen (Fase 1). Bentuk (N, dim)."""
        return self._encode(texts, self.cfg.doc_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Vektor query (Fase 2). Bentuk (Q, dim)."""
        return self._encode(texts, self.cfg.query_prefix)

    def fingerprint(self) -> dict:
        """Identitas encoder; disimpan di meta.json supaya indeks tidak dipakai dengan encoder lain."""
        return {
            "backend": "sentence-transformers",
            "model_name": self.cfg.model_name,
            "revision": self.cfg.revision,
            "query_prefix": self.cfg.query_prefix,
            "doc_prefix": self.cfg.doc_prefix,
            "max_seq_length": self.cfg.max_seq_length,
            "dim": self.dim,
        }


def build_encoder(cfg: EncoderConfig, language: str = "en") -> Encoder:
    """Buat encoder sesuai `encoder.backend` di konfigurasi."""
    if cfg.backend == "hashing":
        return HashingEncoder(cfg, language)
    return SentenceTransformerEncoder(cfg)
