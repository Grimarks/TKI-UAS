"""Konfigurasi proyek (dibaca dari YAML) dengan validasi ketat.

Peran di pipeline (Gambar 1)
    Bukan kotak tersendiri: berkas YAML di `configs/` menentukan pilihan di SEMUA kotak, misalnya model
    "A. Encoder SBERT", berkas keyphrase "B. Keyphrase + bobot (BART-KPG)", nilai α dan C pada
    "Normalisasi + Fusi", dan split untuk "Tuning α".

Masukan  : berkas YAML (lima bagian: data, encoder, keyphrase, retrieval, index).
Keluaran : objek `AppConfig` yang sudah divalidasi.

Prinsip: kunci yang tidak dikenal ditolak (bukan diabaikan), supaya salah ketik seperti `alfa:` tidak
diam-diam membuat sistem berjalan dengan nilai bawaan. YAML dibaca dengan `yaml.safe_load`, yang tidak
mengeksekusi objek Python apa pun.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Konfigurasi tidak valid."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


# Empat skenario yang dilaporkan. bm25 hanya pembanding; hybrid = semantik + keyphrase (BM25 tidak difusikan).
MODES = ("semantic", "keyphrase", "bm25", "hybrid")

# ------------------------------------------------------------------ penolakan desain lama
# Desain lama memakai YAKE, fusi RRF, dan mode tiga jalur (hybrid_bm25). Konfigurasi lama di configs/_arsip/
# sengaja ditolak dengan pesan khusus (bukan sekadar "kunci tidak dikenal") supaya jelas sebabnya.
YAKE_REMOVED = "YAKE sudah tidak didukung pada pipeline ini; gunakan seq2seq atau precomputed"
LEGACY_KEYS = {
    "keyphrase": {"max_ngram": "keyphrase.max_ngram hanya dipakai YAKE; " + YAKE_REMOVED},
    "retrieval": {
        "fusion": "retrieval.fusion sudah dihapus: fusi RRF tidak dipakai lagi, hybrid selalu kombinasi linear (alpha)",
        "rrf_k": "retrieval.rrf_k sudah dihapus: fusi RRF tidak dipakai lagi",
        "bm25_weights": "retrieval.bm25_weights sudah dihapus: BM25 hanya pembanding dan tidak ikut difusikan",
    },
}
HYBRID_BM25_REMOVED = "retrieval.mode hybrid_bm25 sudah dihapus: BM25 hanya pembanding dan tidak ikut difusikan"


# ------------------------------------------------------------------ bagian-bagian konfigurasi
@dataclass
class DataConfig:
    """Bagian `data`: lokasi koleksi BEIR dan split untuk evaluasi/penalaan."""

    path: str = "data/demo"          # folder format BEIR (corpus.jsonl, queries.jsonl, qrels/)
    eval_split: str = "test"         # split qrels untuk evaluasi akhir
    tune_split: str | None = None    # split qrels untuk menala alpha; None = bagi query eval (dev/test)
    dev_fraction: float = 0.3        # dipakai bila tune_split None
    max_docs: int | None = None      # batasi jumlah dokumen (uji cepat)

    def __post_init__(self) -> None:
        _check(0.0 < self.dev_fraction < 1.0, "data.dev_fraction harus di antara 0 dan 1")
        _check(self.max_docs is None or self.max_docs > 0, "data.max_docs harus > 0")


@dataclass
class EncoderConfig:
    """Bagian `encoder` (Jalur A): model embedding dan cara menjalankannya."""

    backend: str = "sentence-transformers"   # sentence-transformers | hashing
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str | None = None              # commit HF (disarankan dipin untuk reproduksibilitas)
    query_prefix: str = ""
    doc_prefix: str = ""
    max_seq_length: int = 256
    batch_size: int = 64
    device: str = "auto"                     # auto | cpu | cuda | mps
    fp16: bool = True                        # hanya berlaku di CUDA
    hashing_dim: int = 512                   # hanya untuk backend hashing

    def __post_init__(self) -> None:
        _check(self.backend in ("sentence-transformers", "hashing"), "encoder.backend tidak dikenal")
        _check(self.device in ("auto", "cpu", "cuda", "mps"), "encoder.device tidak dikenal")
        _check(8 <= self.max_seq_length <= 8192, "encoder.max_seq_length di luar rentang 8..8192")
        _check(1 <= self.batch_size <= 4096, "encoder.batch_size di luar rentang 1..4096")
        _check(16 <= self.hashing_dim <= 8192, "encoder.hashing_dim di luar rentang 16..8192")


@dataclass
class KeyphraseConfig:
    """Bagian `keyphrase` (Jalur B): sumber keyphrase dan cara mencocokkannya.

    Satu-satunya sumber keyphrase adalah generator BART-KPG: dihitung langsung (seq2seq) atau dibaca dari
    hasil Kaggle (precomputed). Bawaan seq2seq karena precomputed wajib menyebut berkas hasilnya.
    """

    method: str = "seq2seq"          # seq2seq | precomputed
    query_method: str = "same"       # same | seq2seq | precomputed
    query_fallback: str = "seq2seq"  # dipakai bila query_method=precomputed dan query tidak ada di berkas: none | seq2seq
    top_n: int = 10                  # jumlah keyphrase per dokumen
    query_top_n: int = 8             # jumlah keyphrase per query
    language: str = "en"             # en | id | none (untuk stopword dan stemmer)
    unit: str = "both"               # phrase | token | both  (satuan pencocokan di indeks keyphrase)
    similarity: str = "cosine"       # cosine (final, persamaan (4)) | jaccard (alternatif dari acuan [7])
    # --- seq2seq (generator, mis. BART-KPG) ---
    model_name_or_path: str = "taln-ls2n/bart-base-kp20k"
    revision: str | None = None
    require_safetensors: bool = True   # tolak bobot .bin (pickle) kecuali sengaja dimatikan
    generation_batch_size: int = 16
    num_beams: int = 4
    max_source_tokens: int = 512
    max_new_tokens: int = 64
    device: str = "auto"
    # --- precomputed (hasil dari Kaggle) ---
    precomputed_files: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for key in ("method", "query_method", "query_fallback"):
            _check(getattr(self, key) != "yake", f"keyphrase.{key}: {YAKE_REMOVED}")
        _check(self.method in ("seq2seq", "precomputed"), "keyphrase.method tidak dikenal")
        _check(self.query_method in ("same", "seq2seq", "precomputed"), "keyphrase.query_method tidak dikenal")
        _check(self.query_fallback in ("none", "seq2seq"), "keyphrase.query_fallback tidak dikenal")
        _check(self.language in ("en", "id", "none"), "keyphrase.language harus en, id, atau none")
        _check(self.unit in ("phrase", "token", "both"), "keyphrase.unit tidak dikenal")
        _check(self.similarity in ("cosine", "jaccard"), "keyphrase.similarity tidak dikenal")
        _check(1 <= self.top_n <= 50 and 1 <= self.query_top_n <= 50, "keyphrase.top_n/query_top_n harus 1..50")
        _check(self.device in ("auto", "cpu", "cuda", "mps"), "keyphrase.device tidak dikenal")
        _check(1 <= self.generation_batch_size <= 1024, "keyphrase.generation_batch_size harus 1..1024")
        _check(1 <= self.num_beams <= 16, "keyphrase.num_beams harus 1..16")
        _check(16 <= self.max_source_tokens <= 4096, "keyphrase.max_source_tokens harus 16..4096")
        _check(8 <= self.max_new_tokens <= 512, "keyphrase.max_new_tokens harus 8..512")
        _check(isinstance(self.precomputed_files, list), "keyphrase.precomputed_files harus berupa daftar")
        if "precomputed" in (self.method, self.query_method):
            _check(bool(self.precomputed_files), "method 'precomputed' memerlukan keyphrase.precomputed_files")

    @property
    def effective_query_method(self) -> str:
        """Metode keyphrase untuk query setelah 'same' diganti dengan metode dokumen."""
        return self.method if self.query_method == "same" else self.query_method


@dataclass
class RetrievalConfig:
    """Bagian `retrieval`: mode bawaan, α dan C pada fusi (persamaan (7)), serta jumlah hasil."""

    mode: str = "hybrid"             # semantic | keyphrase | bm25 | hybrid
    alpha: float = 0.6               # bobot semantik pada fusi (1 - alpha untuk keyphrase)
    candidate_k: int = 100           # C: ukuran kandidat per jalur sebelum fusi
    top_k: int = 10
    max_query_chars: int = 1000

    def __post_init__(self) -> None:
        _check(self.mode != "hybrid_bm25", HYBRID_BM25_REMOVED)
        _check(self.mode in MODES, "retrieval.mode tidak dikenal")
        _check(0.0 <= self.alpha <= 1.0, "retrieval.alpha harus di antara 0 dan 1")
        _check(1 <= self.candidate_k <= 10_000, "retrieval.candidate_k harus 1..10000")
        _check(1 <= self.top_k <= 1000, "retrieval.top_k harus 1..1000")
        _check(1 <= self.max_query_chars <= 20_000, "retrieval.max_query_chars harus 1..20000")


@dataclass
class IndexConfig:
    """Bagian `index`: folder tempat indeks disimpan oleh `hybrid-ir build`."""

    dir: str = "indexes/demo"


@dataclass
class AppConfig:
    """Seluruh konfigurasi satu eksperimen."""

    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    keyphrase: KeyphraseConfig = field(default_factory=KeyphraseConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    index: IndexConfig = field(default_factory=IndexConfig)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "encoder": EncoderConfig,
    "keyphrase": KeyphraseConfig,
    "retrieval": RetrievalConfig,
    "index": IndexConfig,
}


# ------------------------------------------------------------------ pembacaan
def _build_section(cls: type, name: str, raw: Any):
    if raw is None:
        raw = {}
    _check(isinstance(raw, dict), f"bagian '{name}' harus berupa pemetaan (key: value)")
    for key, msg in LEGACY_KEYS.get(name, {}).items():
        _check(key not in raw, msg)
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    _check(not unknown, f"kunci tidak dikenal di '{name}': {unknown}. Kunci yang valid: {sorted(allowed)}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any]) -> AppConfig:
    """Bangun `AppConfig` dari dict hasil YAML; bagian yang tidak ada memakai nilai bawaan."""
    _check(isinstance(raw, dict), "isi konfigurasi harus berupa pemetaan (key: value)")
    unknown = sorted(set(raw) - set(_SECTIONS))
    _check(not unknown, f"bagian tidak dikenal: {unknown}. Bagian yang valid: {sorted(_SECTIONS)}")
    return AppConfig(**{name: _build_section(cls, name, raw.get(name)) for name, cls in _SECTIONS.items()})


def load_config(path: str | Path) -> AppConfig:
    """Baca dan validasi berkas YAML konfigurasi."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"berkas konfigurasi tidak ditemukan: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw or {})
