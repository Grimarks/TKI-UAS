"""Pembangkitan keyphrase dengan BART-KPG, langsung (seq2seq) atau dari hasil Kaggle (precomputed).

Peran di pipeline (Gambar 1)
    Fase 1 — Jalur B: "B. Keyphrase + bobot (BART-KPG)" untuk tiap dokumen (paling banyak 10 frasa).
    Fase 2 — Jalur B: "B. Keyphrase query" dengan metode yang sama (paling banyak 8 frasa).

Masukan  : daftar `TextItem` (judul + isi; query hanya punya judul).
Keluaran : untuk tiap item, daftar (keyphrase, bobot) dengan bobot di (0, 1]; lebih besar = lebih penting.

Rumus
    Bobot menurut urutan keluaran generator (i dihitung dari 0):  b_i = 1 / (1 + 0.25·i).
    Bobot ini menjadi b_{d,u} pada persamaan (3) di `indexes.KeyphraseIndex`.

Dua metode (keduanya dari generator BART-KPG yang sama):
  seq2seq     : generatif. Bisa menghasilkan keyphrase yang TIDAK muncul di teks ("absent keyphrase").
                Berat; di CPU lambat, sebaiknya dijalankan di GPU.
  precomputed : membaca hasil generasi yang sudah dihitung di Kaggle (artifacts/keyphrases_*.jsonl).
                Konfigurasi final memakai ini, ditambah fallback seq2seq untuk query bebas di UI/CLI.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .config import KeyphraseConfig
from .encoders import resolve_device
from .text import make_full_text, normalize_text, text_fingerprint

log = logging.getLogger(__name__)

Keyphrase = tuple[str, float]


class ExtractorUnavailable(RuntimeError):
    """Backend tidak dapat dipakai (paket belum terpasang)."""


class MissingPrecomputed(KeyError):
    """Ada teks yang tidak punya keyphrase di berkas precomputed."""


@dataclass(frozen=True)
class TextItem:
    """Teks yang akan dibangkitkan keyphrase-nya: judul + isi (query: hanya `title`)."""

    title: str
    body: str = ""

    @property
    def full(self) -> str:
        """Teks gabungan; SHA-1-nya menjadi kunci cache precomputed."""
        return make_full_text(self.title, self.body)


class KeyphraseExtractor(Protocol):
    """Antarmuka bersama semua sumber keyphrase."""

    name: str

    def extract(self, items: Sequence[TextItem], top_n: int) -> list[list[Keyphrase]]:
        """Paling banyak `top_n` (keyphrase, bobot) per item, urutan sama dengan `items`."""
        ...

    def fingerprint(self) -> dict:
        """Identitas sumber keyphrase untuk meta.json."""
        ...


# --------------------------------------------------------------------------------------
# Fungsi murni (juga ditiru oleh kaggle/generate_keyphrases.py; ada tes paritasnya)
# --------------------------------------------------------------------------------------
def rank_weights(n: int) -> list[float]:
    """Bobot berdasarkan urutan keluaran generator: 1.0, 0.8, 0.67, 0.57, ... (1 / (1 + 0.25 i))."""
    return [1.0 / (1.0 + 0.25 * i) for i in range(n)]


def parse_generated_keyphrases(text: str, max_phrases: int, sep: str = ";",
                               max_words: int = 8, max_chars: int = 80) -> list[Keyphrase]:
    """Ubah keluaran generator 'a;b;c' menjadi [(a, w1), (b, w2), ...].

    Frasa kosong, terlalu panjang, atau duplikat (tanpa memandang huruf besar/kecil) dibuang.
    """
    seen: set[str] = set()
    phrases: list[str] = []
    for part in text.split(sep):
        phrase = normalize_text(part)
        if not phrase or len(phrase) > max_chars or len(phrase.split()) > max_words:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return list(zip(phrases, rank_weights(len(phrases))))


# --------------------------------------------------------------------------------------
# Generator seq2seq (BART-KPG)
# --------------------------------------------------------------------------------------
class Seq2SeqKeyphraseGenerator:
    """Generator keyphrase One2Seq: satu urutan 'kp1;kp2;kp3' per dokumen.

    Format masukan mengikuti model taln-ls2n/bart-base-kp20k: "judul<s>abstrak".
    """

    name = "seq2seq"

    def __init__(self, cfg: KeyphraseConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ExtractorUnavailable(
                "transformers/torch belum terpasang. Jalankan: pip install 'hybrid-ir[models]'"
            ) from exc
        self.cfg = cfg
        self._torch = torch
        self.device = resolve_device(cfg.device)
        common: dict = {"trust_remote_code": False}
        if cfg.revision:
            common["revision"] = cfg.revision
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, **common)
        model_kwargs = dict(common)
        if cfg.require_safetensors:
            model_kwargs["use_safetensors"] = True   # menolak bobot berformat pickle (.bin)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.commit_hash = getattr(self.model.config, "_commit_hash", None)

    def extract(self, items: Sequence[TextItem], top_n: int) -> list[list[Keyphrase]]:
        """Bangkitkan keyphrase per batch dengan beam search; urutan keluaran sama dengan `items`."""
        n = len(items)
        out: list[list[Keyphrase]] = [[] for _ in range(n)]
        order = sorted(range(n), key=lambda i: -len(items[i].full))   # panjang dulu: kurangi padding
        bs = self.cfg.generation_batch_size
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            sources = [f"{items[i].title}<s>{items[i].body}" for i in idx]
            enc = self.tokenizer(sources, return_tensors="pt", padding=True, truncation=True,
                                 max_length=self.cfg.max_source_tokens).to(self.device)
            with self._torch.inference_mode():
                generated = self.model.generate(**enc, num_beams=self.cfg.num_beams,
                                                max_new_tokens=self.cfg.max_new_tokens, early_stopping=True)
            for i, text in zip(idx, self.tokenizer.batch_decode(generated, skip_special_tokens=True)):
                out[i] = parse_generated_keyphrases(text, top_n)
        return out

    def fingerprint(self) -> dict:
        """Identitas generator (model, revisi, beam, top_n) untuk meta.json indeks."""
        c = self.cfg
        return {"method": "seq2seq", "model": c.model_name_or_path, "revision": c.revision,
                "commit": self.commit_hash, "num_beams": c.num_beams, "top_n": c.top_n}


# --------------------------------------------------------------------------------------
# Precomputed (hasil Kaggle)
# --------------------------------------------------------------------------------------
def _parse_precomputed_line(line: str, where: str) -> tuple[str, list[Keyphrase]]:
    """Validasi satu baris JSONL precomputed: kunci SHA-1 40 heks, bobot positif dan berhingga."""
    try:
        obj = json.loads(line)
        key = obj["key"]
        kps = [(str(k), float(w)) for k, w in obj["keyphrases"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: format tidak valid") from exc
    if not (isinstance(key, str) and len(key) == 40):
        raise ValueError(f"{where}: 'key' harus SHA-1 (40 heks)")
    if any(not math.isfinite(w) or w <= 0 for _, w in kps):
        raise ValueError(f"{where}: bobot harus bilangan positif")
    return key, kps


def _read_precomputed_file(path: Path) -> dict[str, list[Keyphrase]]:
    """Baca satu berkas keyphrase hasil Kaggle menjadi {sha1 teks: [(frasa, bobot), ...]}."""
    if not path.is_file():
        raise FileNotFoundError(f"berkas keyphrase precomputed tidak ditemukan: {path}")
    table: dict[str, list[Keyphrase]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                key, kps = _parse_precomputed_line(line, f"{path.name} baris {lineno}")
                table[key] = kps
    return table


class PrecomputedKeyphrases:
    """Membaca JSONL: {"key": <sha1 teks>, "keyphrases": [["frasa", bobot], ...]} per baris."""

    name = "precomputed"

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self.paths = [str(p) for p in paths]
        self.table: dict[str, list[Keyphrase]] = {}
        for p in paths:
            self.table.update(_read_precomputed_file(Path(p)))

    def lookup(self, item: TextItem) -> list[Keyphrase] | None:
        """Keyphrase tersimpan untuk teks ini, atau None bila tidak ada."""
        return self.table.get(text_fingerprint(item.full))

    def extract(self, items: Sequence[TextItem], top_n: int) -> list[list[Keyphrase]]:
        """Ambil keyphrase tersimpan; bila ada teks yang tidak ditemukan, hentikan dengan `MissingPrecomputed`."""
        out: list[list[Keyphrase]] = []
        missing = 0
        for item in items:
            kps = self.lookup(item)
            if kps is None:
                missing += 1
                out.append([])
            else:
                out.append(kps[:top_n])
        if missing:
            raise MissingPrecomputed(
                f"{missing} dari {len(items)} teks tidak ada di berkas precomputed {self.paths}. "
                "Jalankan ulang kaggle/generate_keyphrases.py untuk korpus/query ini, "
                "atau pakai keyphrase.query_method/query_fallback: seq2seq untuk query bebas."
            )
        return out

    def fingerprint(self) -> dict:
        """Identitas sumber precomputed (jumlah entri)."""
        return {"method": "precomputed", "entries": len(self.table)}


class FallbackExtractor:
    """Pakai hasil precomputed bila ada; teks yang tidak ada (mis. query bebas di UI) diserahkan ke extractor cadangan.

    Extractor cadangan baru dimuat saat pertama kali dibutuhkan (model besar tidak dimuat sia-sia).
    """

    name = "precomputed+fallback"

    def __init__(self, primary: PrecomputedKeyphrases, make_fallback: Callable[[], KeyphraseExtractor]) -> None:
        self.primary = primary
        self._make_fallback = make_fallback
        self._fallback: KeyphraseExtractor | None = None

    def extract(self, items: Sequence[TextItem], top_n: int) -> list[list[Keyphrase]]:
        """Precomputed lebih dulu; sisanya dibangkitkan oleh extractor cadangan (BART lokal)."""
        out: list[list[Keyphrase] | None] = [None] * len(items)
        missing: list[int] = []
        for i, item in enumerate(items):
            kps = self.primary.lookup(item)
            if kps is None:
                missing.append(i)
            else:
                out[i] = kps[:top_n]
        if missing:
            if self._fallback is None:
                self._fallback = self._make_fallback()
            for i, got in zip(missing, self._fallback.extract([items[i] for i in missing], top_n)):
                out[i] = got
        return [o or [] for o in out]

    def fingerprint(self) -> dict:
        """Identitas extractor gabungan."""
        return {"method": "precomputed+fallback", "primary": self.primary.fingerprint()}


def build_extractor(cfg: KeyphraseConfig, role: str) -> KeyphraseExtractor:
    """role: 'doc' atau 'query'. Sisi dokumen dengan 'precomputed' bersifat ketat (harus lengkap)."""
    method = cfg.method if role == "doc" else cfg.effective_query_method
    if method == "seq2seq":
        return Seq2SeqKeyphraseGenerator(cfg)
    if method == "precomputed":
        primary = PrecomputedKeyphrases(cfg.precomputed_files)
        if role == "query" and cfg.query_fallback == "seq2seq":
            return FallbackExtractor(primary, lambda: Seq2SeqKeyphraseGenerator(cfg))
        return primary
    raise ValueError(f"metode keyphrase tidak dikenal: {method}")
