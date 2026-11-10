"""Pemuat dataset format BEIR.

Struktur folder yang diharapkan:

    <folder>/corpus.jsonl      {"_id": "...", "title": "...", "text": "..."}   satu objek per baris
    <folder>/queries.jsonl     {"_id": "...", "text": "..."}
    <folder>/qrels/<split>.tsv kolom: query-id <TAB> corpus-id <TAB> score   (baris pertama = header)

Format ini dipakai SciFact, NFCorpus, SCIDOCS (BEIR), dan juga mudah dibuat untuk korpus sendiri
(misalnya korpus berbahasa Indonesia).
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .text import make_full_text, normalize_text


class DataError(ValueError):
    """Dataset tidak valid atau tidak ditemukan."""


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return make_full_text(self.title, self.text)


@dataclass
class Dataset:
    name: str
    documents: list[Document]
    queries: dict[str, str]                       # qid -> teks query
    qrels: dict[str, dict[str, dict[str, int]]]   # split -> qid -> {doc_id: relevansi}
    splits: list[str] = field(default_factory=list)

    def qrels_for(self, split: str) -> dict[str, dict[str, int]]:
        if split not in self.qrels:
            raise DataError(f"split qrels '{split}' tidak ada. Split yang tersedia: {sorted(self.qrels)}")
        return self.qrels[split]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path.name} baris {lineno}: JSON tidak valid ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise DataError(f"{path.name} baris {lineno}: setiap baris harus berupa objek JSON")
            yield obj


def load_corpus(path: Path, max_docs: int | None = None) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set()
    for obj in iter_jsonl(path):
        if "_id" not in obj:
            raise DataError(f"{path.name}: setiap dokumen wajib punya '_id'")
        doc_id = str(obj["_id"])
        if doc_id in seen:
            raise DataError(f"{path.name}: _id ganda '{doc_id}'")
        seen.add(doc_id)
        title = normalize_text(str(obj.get("title") or ""))
        text = normalize_text(str(obj.get("text") or ""))
        if not (title or text):
            continue  # dokumen kosong tidak berguna
        docs.append(Document(doc_id, title, text))
        if max_docs and len(docs) >= max_docs:
            break
    if not docs:
        raise DataError(f"{path}: korpus kosong")
    return docs


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for obj in iter_jsonl(path):
        if "_id" not in obj or "text" not in obj:
            raise DataError(f"{path.name}: setiap query wajib punya '_id' dan 'text'")
        text = normalize_text(str(obj["text"]))
        if text:
            queries[str(obj["_id"])] = text
    if not queries:
        raise DataError(f"{path}: daftar query kosong")
    return queries


def load_qrels_file(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0].strip().lower() in ("query-id", "query_id", "qid"):
                continue
            if len(row) < 3:
                raise DataError(f"{path.name}: baris qrels harus punya 3 kolom, ditemukan {row}")
            qid, did, score = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                rel = int(float(score))
            except ValueError as exc:
                raise DataError(f"{path.name}: skor relevansi bukan angka: {score!r}") from exc
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_beir_dir(path: str | Path, max_docs: int | None = None) -> Dataset:
    root = Path(path)
    if not root.is_dir():
        raise DataError(
            f"folder dataset tidak ditemukan: {root}. "
            "Unduh dengan `hybrid-ir download scifact` atau siapkan korpus sendiri (lihat README)."
        )
    for name in ("corpus.jsonl", "queries.jsonl"):
        if not (root / name).is_file():
            raise DataError(f"berkas wajib tidak ada: {root / name}")
    documents = load_corpus(root / "corpus.jsonl", max_docs)
    queries = load_queries(root / "queries.jsonl")
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    qdir = root / "qrels"
    if qdir.is_dir():
        for f in sorted(qdir.glob("*.tsv")):
            qrels[f.stem] = load_qrels_file(f)
    return Dataset(name=root.name, documents=documents, queries=queries, qrels=qrels, splits=sorted(qrels))


def usable_queries(dataset: Dataset, split: str) -> dict[str, str]:
    """Query pada `split` yang punya minimal satu dokumen relevan yang ada di korpus."""
    doc_ids = {d.id for d in dataset.documents}
    out: dict[str, str] = {}
    for qid, rels in dataset.qrels_for(split).items():
        if qid in dataset.queries and any(rel > 0 and did in doc_ids for did, rel in rels.items()):
            out[qid] = dataset.queries[qid]
    return out


def split_query_ids(qids: list[str], dev_fraction: float, seed: int = 13) -> tuple[list[str], list[str]]:
    """Bagi id query secara deterministik menjadi (dev, eval). Dipakai bila tidak ada split train."""
    ordered = sorted(qids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_dev = max(1, int(round(len(ordered) * dev_fraction)))
    n_dev = min(n_dev, len(ordered) - 1) if len(ordered) > 1 else 0
    return sorted(ordered[:n_dev]), sorted(ordered[n_dev:])
