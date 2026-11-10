"""HybridRetriever: perakit yang menyatukan encoder, keyphrase, kedua indeks, BM25 pembanding, dan fusi.

Peran di pipeline (Gambar 1)
    Fase 1 (`build`, perintah `hybrid-ir build`):
        Korpus dokumen → Pra-pemrosesan → A. Encoder SBERT → Vector index
                                        → B. Keyphrase + bobot (BART-KPG) → Keyphrase index
    Fase 2 (`search`, perintah `hybrid-ir search` / `hybrid-ir ui`):
        Query pengguna → Pra-pemrosesan → A. Encode query  → Skor semantik  ┐
                                        → B. Keyphrase query → Skor keyphrase ┴→ Normalisasi + Fusi → Ranking top-k
        Pembanding: Token query → Skor BM25 → Ranking top-k (hanya mode bm25; tidak difusikan)
    Fase 3: `score_matrices` dan `bm25_scores` dipakai `evaluation.run_experiment` untuk semua skenario.

Masukan  : konfigurasi + dokumen (build) atau konfigurasi + folder indeks (load); teks query (search).
Keluaran : folder indeks (save) atau `SearchResult` berisi daftar `Hit`.

Indeks disimpan di satu folder (lihat `save`). `load` memeriksa bahwa encoder dan pengaturan indeks
keyphrase yang aktif SAMA dengan yang dipakai saat membangun indeks; bila beda, vektor tidak
sebanding dan hasilnya tidak bermakna, jadi kita berhenti dengan pesan yang jelas.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import scipy

from . import __version__
from .bm25 import BM25
from .config import MODES, AppConfig, KeyphraseConfig
from .data import Document
from .encoders import Encoder, build_encoder
from .fusion import rank_documents
from .indexes import DenseIndex, KeyphraseIndex
from .keyphrases import Keyphrase, KeyphraseExtractor, TextItem, build_extractor
from .text import normalize_text, phrase_units

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SNIPPET_CHARS = 400


class IndexMismatchError(RuntimeError):
    """Indeks tidak cocok dengan konfigurasi yang aktif."""


class QueryError(ValueError):
    """Query atau parameter pencarian tidak valid (kosong, terlalu panjang, mode/alpha/top_k salah)."""


@dataclass
class Hit:
    """Satu dokumen di daftar hasil, lengkap dengan skor tiap jalur untuk penjelasan."""

    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float                 # skor yang dipakai mengurutkan (sesuai mode)
    semantic_score: float        # s_sem mentah (0 bila jalur A tidak dihitung pada mode ini)
    keyphrase_score: float       # s_kp mentah (0 bila jalur B tidak dihitung pada mode ini)
    bm25_score: float = 0.0      # s_bm25 mentah (hanya dihitung pada mode bm25)
    doc_keyphrases: list[str] = field(default_factory=list)
    matched_keyphrases: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Hasil satu query."""

    query: str
    mode: str
    query_keyphrases: list[Keyphrase]
    hits: list[Hit]


def _progress(iterable, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except ImportError:  # pragma: no cover
        return iterable


def _extract_in_chunks(extractor: KeyphraseExtractor, items: Sequence[TextItem], top_n: int,
                       chunk: int = 256) -> list[list[Keyphrase]]:
    out: list[list[Keyphrase]] = []
    starts = range(0, len(items), chunk)
    for s in _progress(starts, "keyphrase", total=len(starts)):
        out.extend(extractor.extract(items[s:s + chunk], top_n))
    return out


def _keyphrase_index_settings(kcfg: KeyphraseConfig) -> dict:
    """Pengaturan yang menentukan isi indeks keyphrase; harus sama saat build dan saat load."""
    return {"language": kcfg.language, "unit": kcfg.unit, "similarity": kcfg.similarity}


class HybridRetriever:
    """Sistem temu kembali lengkap untuk satu koleksi (lihat diagram di kepala modul)."""

    def __init__(self, cfg: AppConfig, encoder: Encoder, doc_extractor: KeyphraseExtractor,
                 query_extractor: KeyphraseExtractor, documents: list[Document], dense: DenseIndex,
                 kp_index: KeyphraseIndex, doc_keyphrases: list[list[Keyphrase]], meta: dict) -> None:
        if not (len(documents) == len(dense.ids) == len(doc_keyphrases) == kp_index.matrix.shape[0]):
            raise IndexMismatchError("jumlah dokumen, vektor, dan keyphrase tidak konsisten")
        self.cfg = cfg
        self.encoder = encoder
        self.doc_extractor = doc_extractor
        self.query_extractor = query_extractor
        self.documents = documents
        self.dense = dense
        self.kp_index = kp_index
        self.doc_keyphrases = doc_keyphrases
        self.meta = meta
        self.doc_ids = dense.ids
        self._bm25: BM25 | None = None

    # ================================================================== Fase 1: build
    @classmethod
    def build(cls, cfg: AppConfig, documents: list[Document], encoder: Encoder | None = None,
              doc_extractor: KeyphraseExtractor | None = None,
              query_extractor: KeyphraseExtractor | None = None) -> "HybridRetriever":
        """Bangun kedua indeks dari dokumen: Jalur A (vector index) dan Jalur B (keyphrase index)."""
        kcfg = cfg.keyphrase
        encoder = encoder or build_encoder(cfg.encoder, kcfg.language)
        doc_extractor = doc_extractor or build_extractor(kcfg, "doc")
        if query_extractor is None:
            query_extractor = cls._default_query_extractor(kcfg, doc_extractor)

        # Jalur A: A. Encoder SBERT -> Vector index
        log.info("Meng-encode %d dokumen ...", len(documents))
        dense = DenseIndex([d.id for d in documents], encoder.encode_documents([d.full_text for d in documents]))

        # Jalur B: B. Keyphrase + bobot (BART-KPG) -> Keyphrase index
        log.info("Mengekstrak keyphrase (%s) ...", doc_extractor.name)
        doc_kps = _extract_in_chunks(doc_extractor, [TextItem(d.title, d.text) for d in documents], kcfg.top_n)
        kp_index = KeyphraseIndex.build(doc_kps, kcfg.language, kcfg.unit, kcfg.similarity)

        meta = cls._index_meta(cfg, len(documents), encoder, doc_extractor)
        return cls(cfg, encoder, doc_extractor, query_extractor, documents, dense, kp_index, doc_kps, meta)

    @staticmethod
    def _default_query_extractor(kcfg: KeyphraseConfig, doc_extractor: KeyphraseExtractor) -> KeyphraseExtractor:
        """Pakai ulang extractor dokumen bila metodenya sama (hemat memuat model dua kali)."""
        same = kcfg.effective_query_method == kcfg.method and kcfg.method != "precomputed"
        return doc_extractor if same else build_extractor(kcfg, "query")

    @staticmethod
    def _index_meta(cfg: AppConfig, n_docs: int, encoder: Encoder, doc_extractor: KeyphraseExtractor) -> dict:
        """Isi meta.json: identitas encoder/extractor dan pengaturan indeks, untuk pemeriksaan saat load."""
        return {
            "schema": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_docs": n_docs,
            "encoder": encoder.fingerprint(),
            "doc_extractor": doc_extractor.fingerprint(),
            "keyphrase_index": _keyphrase_index_settings(cfg.keyphrase),
            "versions": {"hybrid_ir": __version__, "numpy": np.__version__, "scipy": scipy.__version__},
        }

    # ================================================================== simpan / muat indeks
    def save(self, directory: str | Path | None = None) -> Path:
        """Tulis indeks ke folder sementara lalu ganti folder lama sekaligus (tidak pernah setengah jadi)."""
        target = Path(directory or self.cfg.index.dir)
        _guard_replaceable(target)
        tmp = target.parent / (target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        self.dense.save(tmp)
        self.kp_index.save(tmp)
        self._write_documents(tmp / "docs.jsonl")
        self._write_doc_keyphrases(tmp / "doc_keyphrases.jsonl")
        (tmp / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
        return target

    def _write_documents(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for d in self.documents:
                fh.write(json.dumps({"id": d.id, "title": d.title, "text": d.text}, ensure_ascii=False) + "\n")

    def _write_doc_keyphrases(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for d, kps in zip(self.documents, self.doc_keyphrases):
                fh.write(json.dumps({"id": d.id, "keyphrases": [[k, round(w, 6)] for k, w in kps]},
                                    ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, cfg: AppConfig, directory: str | Path | None = None, encoder: Encoder | None = None,
             query_extractor: KeyphraseExtractor | None = None) -> "HybridRetriever":
        """Muat indeks hasil `build`; tolak bila encoder atau pengaturan keyphrase berbeda dari konfigurasi."""
        root = Path(directory or cfg.index.dir)
        meta = _read_meta(root)
        kcfg = cfg.keyphrase
        _check_keyphrase_settings(meta, _keyphrase_index_settings(kcfg))
        encoder = encoder or build_encoder(cfg.encoder, kcfg.language)
        _check_encoder(meta, encoder)

        dense = DenseIndex.load(root)
        kp_index = KeyphraseIndex.load(root, kcfg.language, kcfg.unit, kcfg.similarity)
        documents = [Document(o["id"], o["title"], o["text"]) for o in _read_jsonl(root / "docs.jsonl")]
        kp_rows = {o["id"]: [(k, float(w)) for k, w in o["keyphrases"]] for o in _read_jsonl(root / "doc_keyphrases.jsonl")}
        doc_kps = [kp_rows.get(d.id, []) for d in documents]
        if [d.id for d in documents] != dense.ids:
            raise IndexMismatchError("urutan dokumen pada docs.jsonl tidak sama dengan indeks vektor")
        if query_extractor is None:
            query_extractor = build_extractor(kcfg, "query")
        return cls(cfg, encoder, query_extractor, query_extractor, documents, dense, kp_index, doc_kps, meta)

    # ================================================================== Fase 2: skor per jalur
    def validate_query(self, query: str) -> str:
        """Pra-pemrosesan query (normalisasi) dan tolak query kosong atau terlalu panjang."""
        q = normalize_text(query)
        if not q:
            raise QueryError("query kosong")
        if len(q) > self.cfg.retrieval.max_query_chars:
            raise QueryError(f"query terlalu panjang (maks {self.cfg.retrieval.max_query_chars} karakter)")
        return q

    def score_matrices(self, queries: Sequence[str], need_semantic: bool = True,
                       need_keyphrase: bool = True) -> tuple[np.ndarray, np.ndarray, list[list[Keyphrase]]]:
        """Skor mentah semua dokumen untuk tiap query: (semantik Q × N, keyphrase Q × N, keyphrase query).

        Jalur A: A. Encode query → Skor semantik (persamaan (1)).
        Jalur B: B. Keyphrase query → Skor keyphrase (persamaan (4)).
        Dipakai langsung oleh evaluasi supaya query cukup di-encode/di-ekstrak SEKALI untuk semua
        skenario dan nilai alpha. Jalur yang tidak diminta diisi nol.
        """
        qs = [self.validate_query(q) for q in queries]
        n = len(self.doc_ids)
        if need_semantic:
            sem = self.dense.scores(self.encoder.encode_queries(qs))
        else:
            sem = np.zeros((len(qs), n), dtype=np.float32)
        if need_keyphrase:
            qkps = self.query_extractor.extract([TextItem(q) for q in qs], self.cfg.keyphrase.query_top_n)
            kp = self.kp_index.scores(qkps)
        else:
            qkps = [[] for _ in qs]
            kp = np.zeros((len(qs), n), dtype=np.float32)
        return sem, kp, qkps

    @property
    def bm25(self) -> BM25:
        """Pembanding: BM25 index atas teks dokumen; dibangun di memori saat pertama dipakai (tidak disimpan)."""
        if self._bm25 is None:
            self._bm25 = BM25(self.cfg.keyphrase.language).fit([d.full_text for d in self.documents])
        return self._bm25

    def bm25_scores(self, queries: Sequence[str]) -> np.ndarray:
        """Pembanding: Token query → Skor BM25 (persamaan (5)), bentuk (Q, N)."""
        return self.bm25.scores([self.validate_query(q) for q in queries])

    # ================================================================== Fase 2: search
    def search_many(self, queries: Sequence[str], top_k: int | None = None, mode: str | None = None,
                    alpha: float | None = None) -> list[SearchResult]:
        """Cari banyak query sekaligus. Nilai None berarti pakai nilai di konfigurasi (`retrieval.*`)."""
        r = self.cfg.retrieval
        top_k = r.top_k if top_k is None else top_k
        mode = r.mode if mode is None else mode
        alpha = r.alpha if alpha is None else alpha
        _check_search_args(top_k, mode, alpha)

        sem, kp, bm25, qkps = self._scores_for_mode(queries, mode)
        return [self._hits_for_query(query, mode, alpha, top_k, sem[qi], kp[qi], bm25[qi], qkps[qi])
                for qi, query in enumerate(queries)]

    def search(self, query: str, **kwargs) -> SearchResult:
        """Cari satu query; argumen sama dengan `search_many` (top_k, mode, alpha)."""
        return self.search_many([query], **kwargs)[0]

    def _scores_for_mode(self, queries: Sequence[str], mode: str
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[Keyphrase]]]:
        """Hitung hanya jalur yang dibutuhkan mode ini; jalur lain diisi nol."""
        sem, kp, qkps = self.score_matrices(queries, need_semantic=mode not in ("keyphrase", "bm25"),
                                            need_keyphrase=mode not in ("semantic", "bm25"))
        bm25 = self.bm25_scores(queries) if mode == "bm25" else np.zeros_like(sem)
        return sem, kp, bm25, qkps

    def _hits_for_query(self, query: str, mode: str, alpha: float, top_k: int, sem: np.ndarray,
                        kp: np.ndarray, bm25: np.ndarray, qkps: list[Keyphrase]) -> SearchResult:
        """Normalisasi + Fusi → Ranking top-k untuk satu query, lalu susun `Hit` beserta penjelasannya."""
        idx, vals = rank_documents(sem, kp, mode, alpha, self.cfg.retrieval.candidate_k, top_k,
                                   bm25=bm25 if mode == "bm25" else None)
        q_units = set(self.kp_index.query_units(qkps))
        hits = [self._make_hit(rank, int(di), float(v), float(sem[di]), float(kp[di]), q_units, float(bm25[di]))
                for rank, (di, v) in enumerate(zip(idx, vals), start=1)]
        return SearchResult(normalize_text(query), mode, qkps, hits)

    def _make_hit(self, rank: int, di: int, score: float, sem: float, kp: float, q_units: set[str],
                  bm25: float = 0.0) -> Hit:
        """Satu baris hasil. `matched_keyphrases` = keyphrase dokumen yang berbagi unit dengan query."""
        doc = self.documents[di]
        kcfg = self.cfg.keyphrase
        doc_kps = self.doc_keyphrases[di]
        matched = [p for p, _ in doc_kps if q_units & phrase_units(p, kcfg.language, kcfg.unit)]
        snippet = doc.text[:SNIPPET_CHARS] + ("..." if len(doc.text) > SNIPPET_CHARS else "")
        return Hit(rank, doc.id, doc.title, snippet, score, sem, kp, bm25, [p for p, _ in doc_kps[:10]], matched[:8])


# ====================================================================== fungsi bantu
def _check_search_args(top_k: int, mode: str, alpha: float) -> None:
    """Tolak parameter pencarian di luar rentang (berasal dari pengguna CLI/UI)."""
    if not (1 <= top_k <= 1000):
        raise QueryError("top_k harus di antara 1 dan 1000")
    if not (0.0 <= alpha <= 1.0):
        raise QueryError("alpha harus di antara 0 dan 1")
    if mode not in MODES:
        raise QueryError(f"mode tidak dikenal; pilih salah satu dari {', '.join(MODES)}")


def _read_meta(root: Path) -> dict:
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"indeks tidak ditemukan di {root}. Jalankan `hybrid-ir build` lebih dulu.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA_VERSION:
        raise IndexMismatchError(f"versi skema indeks {meta.get('schema')} != {SCHEMA_VERSION}; build ulang indeks")
    return meta


def _check_keyphrase_settings(meta: dict, expected: dict) -> None:
    if meta["keyphrase_index"] != expected:
        raise IndexMismatchError(
            f"pengaturan indeks keyphrase berbeda. Indeks: {meta['keyphrase_index']}, konfigurasi: {expected}. "
            "Samakan konfigurasi atau build ulang indeks."
        )


def _check_encoder(meta: dict, encoder: Encoder) -> None:
    if encoder.fingerprint() != meta["encoder"]:
        raise IndexMismatchError(
            f"encoder berbeda. Indeks dibangun dengan {meta['encoder']}, konfigurasi sekarang {encoder.fingerprint()}. "
            "Vektor dari encoder berbeda tidak sebanding; build ulang indeks."
        )


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _guard_replaceable(target: Path) -> None:
    """Hanya timpa folder yang jelas-jelas indeks lama (punya meta.json) atau kosong.

    Mencegah `rmtree` menghapus folder penting bila `index.dir` salah diisi (mis. '.' atau '/').
    """
    if not target.exists():
        return
    if not target.is_dir():
        raise IndexMismatchError(f"{target} bukan folder")
    entries = list(target.iterdir())
    if entries and not (target / "meta.json").is_file():
        raise IndexMismatchError(
            f"{target} berisi berkas lain dan bukan indeks hybrid-ir; menolak menimpa. Pilih index.dir yang lain."
        )
