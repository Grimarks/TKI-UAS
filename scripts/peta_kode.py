#!/usr/bin/env python3
"""Bangkitkan tabel "kotak di Gambar 1 → berkas → fungsi → baris" di docs/PETA_KODE.md.

Nomor baris diambil langsung dari kode (modul `ast`), jadi tabel selalu cocok dengan kode.

    ../.venv/bin/python scripts/peta_kode.py          # tulis ulang tabel di docs/PETA_KODE.md
    ../.venv/bin/python scripts/peta_kode.py --cek    # hanya periksa; keluar dengan kode 1 bila tabel basi

Tabel ditulis di antara penanda <!-- PETA:MULAI --> dan <!-- PETA:SELESAI -->; teks lain di berkas itu
ditulis tangan dan tidak disentuh.
"""
from __future__ import annotations

import argparse
import ast
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "PETA_KODE.md"
START, END = "<!-- PETA:MULAI -->", "<!-- PETA:SELESAI -->"
P = "src/hybrid_ir/"

# (judul fase, [(kotak di Gambar 1, berkas, [nama fungsi/metode ...]), ...])
PETA: list[tuple[str, list[tuple[str, str, list[str]]]]] = [
    ("Fase 1 — Indexing dokumen (offline), `hybrid-ir build`", [
        ("Korpus dokumen", P + "data.py", ["load_beir_dir", "load_corpus", "Document.full_text"]),
        ("Pra-pemrosesan", P + "text.py", ["normalize_text", "make_full_text"]),
        ("A. Encoder SBERT", P + "encoders.py", ["SentenceTransformerEncoder.encode_documents",
                                                 "SentenceTransformerEncoder._encode", "_l2_normalize"]),
        ("Vector index", P + "indexes.py", ["DenseIndex.__init__", "DenseIndex.save"]),
        ("B. Keyphrase + bobot (BART-KPG) — di Kaggle", "kaggle/generate_keyphrases.py",
         ["load_generator", "run_worker", "parse_generated_keyphrases", "rank_weights"]),
        ("B. Keyphrase + bobot (BART-KPG) — dibaca lokal", P + "keyphrases.py",
         ["PrecomputedKeyphrases.extract", "Seq2SeqKeyphraseGenerator.extract", "rank_weights"]),
        ("Keyphrase index", P + "text.py", ["phrase_units", "normalize_phrase"]),
        ("Keyphrase index", P + "indexes.py", ["unit_weights", "keyphrase_idf", "KeyphraseIndex.build"]),
        ("Pembanding: Token + stemming → BM25 index", P + "text.py", ["content_tokens"]),
        ("Pembanding: Token + stemming → BM25 index", P + "bm25.py", ["bm25_idf", "BM25.fit"]),
        ("(perakit Fase 1)", P + "retriever.py", ["HybridRetriever.build", "HybridRetriever.save"]),
        ("(perintah)", P + "cli.py", ["_cmd_build"]),
    ]),
    ("Fase 2 — Query & retrieval (online), `hybrid-ir search` / `hybrid-ir ui`", [
        ("Query pengguna", P + "cli.py", ["_cmd_search"]),
        ("Query pengguna", P + "ui_app.py", ["main"]),
        ("Pra-pemrosesan", P + "retriever.py", ["HybridRetriever.validate_query"]),
        ("A. Encode query", P + "encoders.py", ["SentenceTransformerEncoder.encode_queries"]),
        ("Skor semantik", P + "indexes.py", ["DenseIndex.scores"]),
        ("B. Keyphrase query", P + "keyphrases.py", ["build_extractor", "FallbackExtractor.extract"]),
        ("Skor keyphrase", P + "indexes.py", ["KeyphraseIndex.query_units", "KeyphraseIndex.scores",
                                              "KeyphraseIndex._cosine_row"]),
        ("Pembanding: Token query → Skor BM25", P + "retriever.py", ["HybridRetriever.bm25",
                                                                     "HybridRetriever.bm25_scores"]),
        ("Pembanding: Token query → Skor BM25", P + "bm25.py", ["BM25.scores"]),
        ("(perakit Fase 2)", P + "retriever.py", ["HybridRetriever.search_many", "HybridRetriever.score_matrices",
                                                  "HybridRetriever._scores_for_mode"]),
        ("Normalisasi + Fusi", P + "fusion.py", ["candidate_set", "minmax", "fuse_weighted"]),
        ("Ranking top-k", P + "fusion.py", ["rank_documents", "topk_indices", "positive_topk"]),
        ("Ranking top-k", P + "retriever.py", ["HybridRetriever._hits_for_query", "HybridRetriever._make_hit"]),
        ("Ranking top-k (tampilan)", P + "cli.py", ["_print_result"]),
        ("Ranking top-k (tampilan)", P + "ui_app.py", ["render_hit"]),
    ]),
    ("Fase 3 — Evaluasi, `hybrid-ir evaluate`", [
        ("Data uji BEIR", P + "data.py", ["usable_queries", "load_qrels_file"]),
        ("Data uji BEIR", P + "evaluation.py", ["_split_queries"]),
        ("Skenario", P + "evaluation.py", ["ScoreTable.compute", "ScoreTable.single_signal_rankers",
                                           "ScoreTable.hybrid_ranker", "_evaluate_scenarios"]),
        ("Tuning α", P + "evaluation.py", ["_alpha_sweep", "_select_alpha"]),
        ("Metrik", P + "evaluation.py", ["_dcg", "ndcg_at_k", "average_precision", "recall_at_k",
                                         "precision_at_k", "mrr_at_k", "evaluate_run"]),
        ("Analisis", P + "evaluation.py", ["paired_bootstrap", "_significance_tests", "write_outputs"]),
        ("(perakit Fase 3)", P + "evaluation.py", ["run_experiment"]),
        ("(perintah)", P + "cli.py", ["_cmd_evaluate"]),
    ]),
]

# Persamaan di laporan UAS → kode yang menghitungnya.
PERSAMAAN: list[tuple[str, str, str, list[str]]] = [
    ("(1)", "skor semantik, cosine = perkalian dalam", P + "indexes.py", ["DenseIndex.scores"]),
    ("(2)", "idf unit keyphrase", P + "indexes.py", ["keyphrase_idf"]),
    ("(3)", "bobot unit w = b·idf", P + "indexes.py", ["unit_weights", "KeyphraseIndex.build"]),
    ("(4)", "skor keyphrase, cosine", P + "indexes.py", ["KeyphraseIndex._cosine_row"]),
    ("(5)", "BM25 (pembanding)", P + "bm25.py", ["bm25_idf", "BM25.fit", "BM25.scores"]),
    ("(6)", "normalisasi min-max", P + "fusion.py", ["minmax"]),
    ("(7)", "fusi linear α", P + "fusion.py", ["fuse_weighted"]),
    ("(8)", "DCG@k", P + "evaluation.py", ["_dcg"]),
    ("(9)", "nDCG@k", P + "evaluation.py", ["ndcg_at_k"]),
]


@lru_cache(maxsize=None)
def _definitions(path: str) -> dict[str, tuple[int, int]]:
    """{nama atau Kelas.metode: (baris awal, baris akhir)} untuk semua def/class tingkat atas dan metode."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = (node.lineno, node.end_lineno)
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    # dekorator (@property, @classmethod) ikut dihitung sebagai awal definisi
                    start = min([sub.lineno] + [d.lineno for d in sub.decorator_list])
                    out[f"{node.name}.{sub.name}"] = (start, sub.end_lineno)
    return out


def _span(path: str, name: str) -> str:
    try:
        a, b = _definitions(path)[name]
    except KeyError:
        raise SystemExit(f"GALAT: {name} tidak ditemukan di {path}; perbarui PETA di scripts/peta_kode.py")
    return f"{a}–{b}"


def render() -> str:
    """Isi Markdown di antara penanda."""
    lines: list[str] = []
    for fase, rows in PETA:
        lines += [f"### {fase}", "", "| Kotak di Gambar 1 | Berkas | Fungsi | Baris |", "|---|---|---|---|"]
        last_box = None
        for box, path, names in rows:
            for name in names:
                shown = box if box != last_box else ""
                last_box = box
                lines.append(f"| {shown} | `{path}` | `{name}` | {_span(path, name)} |")
        lines.append("")
    lines += ["### Persamaan di laporan → kode", "", "| Persamaan | Isi | Berkas | Fungsi | Baris |",
              "|---|---|---|---|---|"]
    for eq, isi, path, names in PERSAMAAN:
        for i, name in enumerate(names):
            lines.append(f"| {eq if i == 0 else ''} | {isi if i == 0 else ''} | `{path}` | `{name}` | "
                         f"{_span(path, name)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cek", action="store_true", help="hanya periksa apakah tabel masih cocok dengan kode")
    args = ap.parse_args(argv)
    text = DOC.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"GALAT: penanda {START} / {END} tidak ada di {DOC}")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render()}{END}{tail}"
    if args.cek:
        if new != text:
            print("Tabel di docs/PETA_KODE.md BASI. Jalankan: python scripts/peta_kode.py")
            return 1
        print("docs/PETA_KODE.md cocok dengan kode.")
        return 0
    DOC.write_text(new, encoding="utf-8")
    print(f"Tabel ditulis ke {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
