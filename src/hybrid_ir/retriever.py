"""HybridRetriever: menyatukan encoder, ekstraktor keyphrase, kedua indeks, dan fusi.

Alur `build` (offline)            Alur `search` (online)
  dokumen -> encoder -> DenseIndex     query -> encoder  -> skor semantik  \
  dokumen -> extractor -> KeyphraseIndex   query -> extractor -> skor keyphrase  -> fusi -> top-k
  (dokumen -> BM25, dibangun saat pertama dibutuhkan)  query -> BM25 -> skor leksikal (hanya mode bm25, pembanding)

Indeks disimpan di satu folder (lihat `save`). `load` memeriksa bahwa encoder dan pengaturan indeks
keyphrase yang aktif SAMA dengan yang dipakai saat membangun indeks; bila beda, vektor tidak akan
sebanding dan hasilnya sampah, jadi kita berhenti dengan pesan yang jelas.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy

from . import __version__
from .bm25 import BM25
from .config import MODES, AppConfig
from .data import Document
from .encoders import Encoder, build_encoder
from .fusion import rank_documents
from .indexes import DenseIndex, KeyphraseIndex
from .keyphrases import Keyphrase, KeyphraseExtractor, TextItem, build_extractor
from .text import normalize_text, phrase_units

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class IndexMismatchError(RuntimeError):
    """Indeks tidak cocok dengan konfigurasi yang aktif."""


class QueryError(ValueError):
    """Query tidak valid (kosong atau terlalu panjang)."""


@dataclass
class Hit:
    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float
    semantic_score: float
    keyphrase_score: float
    bm25_score: float = 0.0
    doc_keyphrases: list[str] = field(default_factory=list)
    matched_keyphrases: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
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


class HybridRetriever:
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

    @property
    def bm25(self) -> BM25:
        """Indeks BM25 atas teks dokumen; dibangun di memori saat pertama dipakai (cepat; tidak disimpan ke disk)."""
        if self._bm25 is None:
            self._bm25 = BM25(self.cfg.keyphrase.language).fit([d.full_text for d in self.documents])
        return self._bm25

    def bm25_scores(self, queries: Sequence[str]) -> np.ndarray:
        return self.bm25.scores([self.validate_query(q) for q in queries])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, cfg: AppConfig, documents: list[Document], encoder: Encoder | None = None,
              doc_extractor: KeyphraseExtractor | None = None,
              query_extractor: KeyphraseExtractor | None = None) -> "HybridRetriever":
        kcfg = cfg.keyphrase
        encoder = encoder or build_encoder(cfg.encoder, kcfg.language)
        doc_extractor = doc_extractor or build_extractor(kcfg, "doc")
        if query_extractor is None:
            same = kcfg.effective_query_method == kcfg.method and kcfg.method != "precomputed"
            query_extractor = doc_extractor if same else build_extractor(kcfg, "query")

        log.info("Meng-encode %d dokumen ...", len(documents))
        doc_vecs = encoder.encode_documents([d.full_text for d in documents])
        log.info("Mengekstrak keyphrase (%s) ...", doc_extractor.name)
        items = [TextItem(d.title, d.text) for d in documents]
        doc_kps = _extract_in_chunks(doc_extractor, items, kcfg.top_n)

        dense = DenseIndex([d.id for d in documents], doc_vecs)
        kp_index = KeyphraseIndex.build(doc_kps, kcfg.language, kcfg.unit, kcfg.similarity)
        meta = {
            "schema": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_docs": len(documents),
            "encoder": encoder.fingerprint(),
            "doc_extractor": doc_extractor.fingerprint(),
            "keyphrase_index": {"language": kcfg.language, "unit": kcfg.unit, "similarity": kcfg.similarity},
            "versions": {"hybrid_ir": __version__, "numpy": np.__version__, "scipy": scipy.__version__},
        }
        return cls(cfg, encoder, doc_extractor, query_extractor, documents, dense, kp_index, doc_kps, meta)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path | None = None) -> Path:
        target = Path(directory or self.cfg.index.dir)
        _guard_replaceable(target)
        tmp = target.parent / (target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        self.dense.save(tmp)
        self.kp_index.save(tmp)
        with (tmp / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for d in self.documents:
                fh.write(json.dumps({"id": d.id, "title": d.title, "text": d.text}, ensure_ascii=False) + "\n")
        with (tmp / "doc_keyphrases.jsonl").open("w", encoding="utf-8") as fh:
            for d, kps in zip(self.documents, self.doc_keyphrases):
                fh.write(json.dumps({"id": d.id, "keyphrases": [[k, round(w, 6)] for k, w in kps]},
                                    ensure_ascii=False) + "\n")
        (tmp / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
        return target

    @classmethod
    def load(cls, cfg: AppConfig, directory: str | Path | None = None, encoder: Encoder | None = None,
             query_extractor: KeyphraseExtractor | None = None) -> "HybridRetriever":
        root = Path(directory or cfg.index.dir)
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"indeks tidak ditemukan di {root}. Jalankan `hybrid-ir build` lebih dulu.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema") != SCHEMA_VERSION:
            raise IndexMismatchError(f"versi skema indeks {meta.get('schema')} != {SCHEMA_VERSION}; build ulang indeks")

        kcfg = cfg.keyphrase
        expected_kp = {"language": kcfg.language, "unit": kcfg.unit, "similarity": kcfg.similarity}
        if meta["keyphrase_index"] != expected_kp:
            raise IndexMismatchError(
                f"pengaturan indeks keyphrase berbeda. Indeks: {meta['keyphrase_index']}, konfigurasi: {expected_kp}. "
                "Samakan konfigurasi atau build ulang indeks."
            )
        encoder = encoder or build_encoder(cfg.encoder, kcfg.language)
        if encoder.fingerprint() != meta["encoder"]:
            raise IndexMismatchError(
                f"encoder berbeda. Indeks dibangun dengan {meta['encoder']}, konfigurasi sekarang {encoder.fingerprint()}. "
                "Vektor dari encoder berbeda tidak sebanding; build ulang indeks."
            )

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

    # ------------------------------------------------------------------ query handling
    def validate_query(self, query: str) -> str:
        q = normalize_text(query)
        if not q:
            raise QueryError("query kosong")
        if len(q) > self.cfg.retrieval.max_query_chars:
            raise QueryError(f"query terlalu panjang (maks {self.cfg.retrieval.max_query_chars} karakter)")
        return q

    def score_matrices(self, queries: Sequence[str], need_semantic: bool = True,
                       need_keyphrase: bool = True) -> tuple[np.ndarray, np.ndarray, list[list[Keyphrase]]]:
        """Skor mentah semua dokumen untuk tiap query: (semantik Q x N, keyphrase Q x N, keyphrase query).

        Dipakai langsung oleh evaluasi supaya query cukup di-encode/di-ekstrak SEKALI untuk semua
        skenario dan nilai alpha.
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

    # ------------------------------------------------------------------ search
    def search_many(self, queries: Sequence[str], top_k: int | None = None, mode: str | None = None,
                    alpha: float | None = None) -> list[SearchResult]:
        r = self.cfg.retrieval
        top_k = r.top_k if top_k is None else top_k
        mode = r.mode if mode is None else mode
        alpha = r.alpha if alpha is None else alpha
        if not (1 <= top_k <= 1000):
            raise QueryError("top_k harus di antara 1 dan 1000")
        if not (0.0 <= alpha <= 1.0):
            raise QueryError("alpha harus di antara 0 dan 1")
        if mode not in MODES:
            raise QueryError(f"mode tidak dikenal; pilih salah satu dari {', '.join(MODES)}")

        sem, kp, qkps = self.score_matrices(queries, need_semantic=mode not in ("keyphrase", "bm25"),
                                            need_keyphrase=mode not in ("semantic", "bm25"))
        use_bm = mode == "bm25"
        bm = self.bm25_scores(queries) if use_bm else np.zeros_like(sem)
        results: list[SearchResult] = []
        for qi, query in enumerate(queries):
            idx, vals = rank_documents(sem[qi], kp[qi], mode, alpha, r.candidate_k, top_k,
                                       bm=bm[qi] if use_bm else None)
            q_units = set(self.kp_index.query_units(qkps[qi]))
            hits = [
                self._make_hit(rank, int(di), float(v), float(sem[qi, di]), float(kp[qi, di]), q_units,
                               float(bm[qi, di]))
                for rank, (di, v) in enumerate(zip(idx, vals), start=1)
            ]
            results.append(SearchResult(normalize_text(query), mode, qkps[qi], hits))
        return results

    def search(self, query: str, **kwargs) -> SearchResult:
        return self.search_many([query], **kwargs)[0]

    def _make_hit(self, rank: int, di: int, score: float, sem: float, kp: float, q_units: set[str],
                  bm: float = 0.0) -> Hit:
        doc = self.documents[di]
        kcfg = self.cfg.keyphrase
        doc_kps = self.doc_keyphrases[di]
        matched = [p for p, _ in doc_kps if q_units & phrase_units(p, kcfg.language, kcfg.unit)]
        snippet = doc.text[:400] + ("..." if len(doc.text) > 400 else "")
        return Hit(rank, doc.id, doc.title, snippet, score, sem, kp, bm, [p for p, _ in doc_kps[:10]], matched[:8])


def _read_jsonl(path: Path):
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
