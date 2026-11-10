#!/usr/bin/env python3
"""Analisis UAS: tabel dan grafik dari hasil yang SUDAH ADA (tanpa pelatihan ulang, tanpa Kaggle).

Sumber angka:
  * results/<dataset>_bart/metrics.json dan per_query_ndcg10.csv  -> tabel utama, signifikansi, α, menang/seri/kalah
  * indexes/<dataset>_bart/ + artifacts/keyphrases_*.jsonl + data/  -> statistik keyphrase dan studi kasus,
    dihitung ulang dengan fungsi kode yang sama (ScoreTable, normalize_phrase, unit_weights, content_tokens).
    Sebagai pengaman, nDCG@10 hasil hitung ulang harus SAMA dengan per_query_ndcg10.csv; bila tidak, skrip berhenti.

Keluaran:
  analisis/*.csv, analisis/*.md, analisis/ringkasan.json, analisis/README.md (semua tabel dalam satu halaman)
  analisis/gambar/*.png (300 dpi)

Jalankan dari folder hybrid-ir/:
    ../.venv/bin/python scripts/analisis_uas.py                  # semua, termasuk pengukuran waktu (~2 menit)
    ../.venv/bin/python scripts/analisis_uas.py --tanpa-waktu    # lewati pengukuran waktu (pakai tabel_waktu.csv lama)
    ../.venv/bin/python scripts/analisis_uas.py --tanpa-bart     # ukur waktu tanpa memuat BART untuk query bebas
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # model sudah ada di cache; jangan menghubungi Hugging Face
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import json
import logging
import platform
import random
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from hybrid_ir.config import load_config
from hybrid_ir.data import load_beir_dir, usable_queries
from hybrid_ir.encoders import resolve_device
from hybrid_ir.evaluation import (DEPTH, METRIC_COLUMNS, SCENARIOS, ScoreTable, _split_queries, ndcg_at_k)
from hybrid_ir.indexes import unit_weights
from hybrid_ir.keyphrases import TextItem
from hybrid_ir.retriever import HybridRetriever
from hybrid_ir.text import content_tokens, normalize_phrase, stem, tokenize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analisis"
FIG = OUT / "gambar"
DATASETS = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}
SEED = 13
ROUNDS = 3   # putaran pengukuran waktu per skenario

NAMA = {"bm25": "BM25 (pembanding)", "semantic": "Semantic-only", "keyphrase": "Keyphrase-only",
        "hybrid_weighted": "Hybrid"}
NAMA_PENDEK = {"bm25": "BM25", "semantic": "Semantic", "keyphrase": "Keyphrase", "hybrid_weighted": "Hybrid"}
MODE_SEARCH = {"bm25": "bm25", "semantic": "semantic", "keyphrase": "keyphrase", "hybrid_weighted": "hybrid"}
LABEL_METRIK = {"ndcg@10": "nDCG@10", "map@100": "MAP@100", "recall@10": "Recall@10", "recall@100": "Recall@100",
                "precision@10": "P@10", "mrr@10": "MRR@10"}

# Warna (palet referensi skill dataviz, mode terang; divalidasi: biru↔oranye ΔE 24.7, biru↔merah ΔE 21.6)
C = {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
     "axis": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834", "red": "#e34948", "neutral": "#c3c2b7",
     "neutral_bar": "#b4b2a9", "tie_bg": "#e9e8e3"}

log = logging.getLogger("analisis")


# ====================================================================== utilitas keluaran
def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_md(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def load_metrics(ds: str) -> dict:
    return json.loads((ROOT / "results" / f"{ds}_bart" / "metrics.json").read_text(encoding="utf-8"))


def load_per_query(ds: str) -> dict[str, dict[str, float]]:
    """{skenario: {qid: nDCG@10}} dari per_query_ndcg10.csv (6 desimal, sesuai berkas)."""
    with (ROOT / "results" / f"{ds}_bart" / "per_query_ndcg10.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {s: {r["query_id"]: float(r[s]) for r in rows} for s in SCENARIOS}


# ====================================================================== 1. tabel utama + signifikansi
def tabel_utama(metrics: dict[str, dict]) -> tuple[str, dict]:
    header = ["Dataset", "Skenario"] + [LABEL_METRIK[m] for m in METRIC_COLUMNS]
    rows, summary = [], {}
    for ds, m in metrics.items():
        for s in SCENARIOS:
            vals = m["scenarios"][s]["metrics"]
            name = NAMA[s] + (f" (α = {m['alpha_selected']})" if s == "hybrid_weighted" else "")
            rows.append([DATASETS[ds], name] + [fmt(vals[c]) for c in METRIC_COLUMNS])
            summary.setdefault(ds, {})[s] = {c: vals[c] for c in METRIC_COLUMNS}
    write_csv("tabel_utama.csv", ["dataset", "skenario"] + METRIC_COLUMNS,
              [[ds, s] + [metrics[ds]["scenarios"][s]["metrics"][c] for c in METRIC_COLUMNS]
               for ds in metrics for s in SCENARIOS])
    md = "## Tabel utama (split test)\n\n" + md_table(header, rows)
    write_md("tabel_utama.md", md)
    return md, summary


def tabel_signifikansi(metrics: dict[str, dict]) -> tuple[str, list]:
    header = ["Dataset", "Perbandingan", "Selisih rata-rata nDCG@10", "CI 95%", "p", "n", "Signifikan (p < 0.05)"]
    rows, raw = [], []
    for ds, m in metrics.items():
        for s in m["significance"]:
            sig = s["p_value"] < 0.05
            rows.append([DATASETS[ds], f"Hybrid − {NAMA_PENDEK[s['b']]}", f"{s['mean_diff']:+.4f}",
                         f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]", fmt(s["p_value"]), s["n"],
                         "ya" if sig else "tidak"])
            raw.append({"dataset": ds, "a": s["a"], "b": s["b"], "mean_diff": s["mean_diff"], "ci95": s["ci95"],
                        "p_value": s["p_value"], "n": s["n"], "signifikan": sig})
    write_csv("tabel_signifikansi.csv", ["dataset", "a", "b", "mean_diff", "ci95_low", "ci95_high", "p_value", "n"],
              [[r["dataset"], r["a"], r["b"], r["mean_diff"], r["ci95"][0], r["ci95"][1], r["p_value"], r["n"]]
               for r in raw])
    md = ("## Uji signifikansi (paired bootstrap 10.000×, nDCG@10, Hybrid dikurangi pembanding)\n\n"
          + md_table(header, rows)
          + "\n\nCatatan: p = 0.0001 adalah batas bawah (1/10.000), artinya tidak ada satu pun sampel bootstrap "
            "yang selisihnya ≤ 0.")
    write_md("tabel_signifikansi.md", md)
    return md, raw


def tabel_alpha(metrics: dict[str, dict]) -> tuple[str, dict]:
    rows, summary = [], {}
    for ds, m in metrics.items():
        tune = {r["alpha"]: r for r in m["alpha_sweep_tune"]}
        ev = {r["alpha"]: r for r in m["alpha_sweep_eval_posthoc"]}
        for a in tune:
            rows.append([ds, a, tune[a]["ndcg@10"], ev[a]["ndcg@10"], tune[a]["recall@100"], ev[a]["recall@100"]])
        best_eval = max(m["alpha_sweep_eval_posthoc"], key=lambda r: r["ndcg@10"])
        summary[ds] = {"alpha_selected": m["alpha_selected"], "tune_split": m["tune_source"],
                       "n_tune": m["n_tune_queries"], "n_eval": m["n_eval_queries"],
                       "ndcg_tune_at_selected": tune[m["alpha_selected"]]["ndcg@10"],
                       "alpha_terbaik_posthoc_test": best_eval["alpha"], "ndcg_test_terbaik_posthoc": best_eval["ndcg@10"]}
    write_csv("tabel_alpha.csv", ["dataset", "alpha", "ndcg10_tune", "ndcg10_test_posthoc", "recall100_tune",
                                  "recall100_test_posthoc"], rows)
    header = ["α"] + [f"{DATASETS[ds]} {k}" for ds in metrics for k in ("tune", "test*")]
    alphas = [r["alpha"] for r in next(iter(metrics.values()))["alpha_sweep_tune"]]
    body = []
    for a in alphas:
        line = [a]
        for ds in metrics:
            line += [fmt(r[2]) for r in rows if r[0] == ds and r[1] == a]
            line += [fmt(r[3]) for r in rows if r[0] == ds and r[1] == a]
        body.append(line)
    md = ("## Pengaruh α terhadap nDCG@10\n\n" + md_table(header, body)
          + "\n\n*test = sapuan pada query uji, **hanya untuk analisis**; α dipilih dari kolom tune "
            "(nilai tertinggi; seri → paling dekat 0.5).")
    write_md("tabel_alpha.md", md)
    return md, summary


# ====================================================================== 2. menang/seri/kalah per query
def menang_seri_kalah(per_query: dict[str, dict]) -> tuple[str, dict]:
    rows, summary = [], {}
    for ds, pq in per_query.items():
        for other in ("semantic", "bm25", "keyphrase"):
            qids = sorted(pq["hybrid_weighted"])
            d = np.array([pq["hybrid_weighted"][q] - pq[other][q] for q in qids])
            d = np.round(d, 6)                                   # CSV 6 desimal: selisih 0 berarti seri
            win, tie, loss = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
            both_zero = int(sum(1 for q in qids if pq["hybrid_weighted"][q] == 0 and pq[other][q] == 0))
            rows.append([DATASETS[ds], f"Hybrid vs {NAMA_PENDEK[other]}", len(qids), win, tie, loss, both_zero,
                         f"{d.mean():+.4f}"])
            summary.setdefault(ds, {})[other] = {"n": len(qids), "menang": win, "seri": tie, "kalah": loss,
                                                 "seri_keduanya_nol": both_zero, "rata_selisih": float(d.mean())}
    header = ["Dataset", "Perbandingan", "Query", "Menang", "Seri", "Kalah", "Seri, keduanya nDCG@10 = 0",
              "Rata-rata selisih"]
    write_csv("tabel_menang_seri_kalah.csv", ["dataset", "perbandingan", "n", "menang", "seri", "kalah",
                                              "seri_keduanya_nol", "rata_selisih"], rows)
    md = ("## Menang / seri / kalah per query (nDCG@10, split test)\n\n" + md_table(header, rows)
          + "\n\nMenang = nDCG@10 Hybrid lebih tinggi pada query itu; seri = sama persis (6 desimal).")
    write_md("tabel_menang_seri_kalah.md", md)
    return md, summary


# ====================================================================== 3. data
def tabel_dataset(datasets: dict) -> tuple[str, dict]:
    rows, summary = [], {}
    roles = {"scifact": {"train": "Penalaan α", "test": "Evaluasi akhir"},
             "nfcorpus": {"train": "Tidak dipakai", "dev": "Penalaan α", "test": "Evaluasi akhir"}}
    for ds, dset in datasets.items():
        for split in dset.splits:
            qrels = dset.qrels_for(split)
            pairs = sum(len(v) for v in qrels.values())
            usable = len(usable_queries(dset, split))
            rel_per_q = statistics.mean(sum(1 for r in v.values() if r > 0) for v in qrels.values())
            rows.append([DATASETS[ds], len(dset.documents), split, len(qrels), usable, pairs, f"{rel_per_q:.1f}",
                         roles[ds].get(split, "-")])
            summary.setdefault(ds, {"n_docs": len(dset.documents)})[split] = {
                "query_qrels": len(qrels), "query_terpakai": usable, "pasangan_qrels": pairs,
                "rata_relevan_per_query": rel_per_q}
    header = ["Koleksi", "Dokumen", "Split", "Query di qrels", "Query terpakai*", "Pasangan qrels",
              "Rata-rata dok. relevan/query", "Peran"]
    write_csv("tabel_dataset.csv", ["dataset", "dokumen", "split", "query_qrels", "query_terpakai", "pasangan_qrels",
                                    "rata_relevan_per_query", "peran"], rows)
    md = ("## Statistik koleksi\n\n" + md_table(header, rows)
          + "\n\n*Query terpakai = query yang punya minimal satu dokumen relevan di korpus (`data.usable_queries`). "
            "Untuk split penalaan, query yang juga ada di test dikeluarkan.")
    write_md("tabel_dataset.md", md)
    return md, summary


# ====================================================================== 4. statistik keyphrase
def _stemmed_tokens(text: str, lang: str) -> list[str]:
    """Token teks setelah aturan normalisasi yang SAMA dengan normalize_phrase (tokenize → stem)."""
    return [stem(t, lang) for t in tokenize(text)]


def is_present(phrase: str, text_tokens_joined: str, lang: str) -> bool | None:
    """True bila keyphrase (setelah normalize_phrase) muncul utuh berurutan di teks; None bila frasa kosong."""
    norm = normalize_phrase(phrase, lang)
    if not norm:
        return None
    return f" {norm} " in text_tokens_joined


def keyphrase_stats(ds: str, retriever: HybridRetriever, split, table: ScoreTable,
                    per_query: dict[str, dict]) -> tuple[dict, list]:
    lang = retriever.cfg.keyphrase.language
    # --- dokumen
    n_kp_doc = [len(k) for k in retriever.doc_keyphrases]
    present = absent = empty = 0
    absent_by_rank: dict[int, list[int]] = {}
    for doc, kps in zip(retriever.documents, retriever.doc_keyphrases):
        joined = " " + " ".join(_stemmed_tokens(doc.full_text, lang)) + " "
        for i, (p, _) in enumerate(kps):
            r = is_present(p, joined, lang)
            if r is None:
                empty += 1
                continue
            present += r
            absent += not r
            absent_by_rank.setdefault(i, []).append(0 if r else 1)
    # --- query evaluasi (keyphrase dari berkas precomputed, sama seperti saat evaluate)
    eval_ids = list(split.eval)
    eval_texts = [retriever.validate_query(split.eval[q]) for q in eval_ids]
    qkps = retriever.query_extractor.extract([TextItem(t) for t in eval_texts], retriever.cfg.keyphrase.query_top_n)
    n_kp_q = [len(k) for k in qkps]
    qp = qa = 0
    for t, kps in zip(eval_texts, qkps):
        joined = " " + " ".join(_stemmed_tokens(t, lang)) + " "
        for p, _ in kps:
            r = is_present(p, joined, lang)
            if r is not None:
                qp += r
                qa += not r
    # --- skor keyphrase nol
    rows = np.array([table.row[q] for q in eval_ids])
    kp = table.kp[rows]
    zero_all = int((kp.max(axis=1) <= 0).sum())
    n_pos = (kp > 0).sum(axis=1)
    kp_only_zero = int(sum(1 for q in eval_ids if per_query["keyphrase"][q] == 0))
    sem_only_zero = int(sum(1 for q in eval_ids if per_query["semantic"][q] == 0))
    # cakupan: pasangan (query, dokumen relevan) yang skor keyphrase-nya > 0
    di = {d: i for i, d in enumerate(retriever.doc_ids)}
    cov_hit = cov_total = 0
    q_with_hit = 0                      # query yang punya >= 1 dokumen relevan dengan skor keyphrase > 0
    n_rel_per_q = []
    for q in eval_ids:
        hit_q = False
        n_rel = 0
        for d, rel in split.eval_qrels[q].items():
            if rel > 0 and d in di:
                n_rel += 1
                cov_total += 1
                h = bool(table.kp[table.row[q], di[d]] > 0)
                cov_hit += h
                hit_q = hit_q or h
        q_with_hit += hit_q
        n_rel_per_q.append(n_rel)
    stats = {
        "dok_rata_keyphrase": float(np.mean(n_kp_doc)), "dok_median_keyphrase": float(np.median(n_kp_doc)),
        "dok_min_keyphrase": int(min(n_kp_doc)), "dok_maks_keyphrase": int(max(n_kp_doc)),
        "dok_tanpa_keyphrase": int(sum(1 for n in n_kp_doc if n == 0)),
        "dok_present": present, "dok_absent": absent, "dok_frasa_kosong": empty,
        "dok_proporsi_absent": absent / (present + absent),
        "dok_proporsi_absent_per_urutan": {i + 1: float(np.mean(v)) for i, v in sorted(absent_by_rank.items())},
        "dok_jumlah_frasa_per_urutan": {i + 1: len(v) for i, v in sorted(absent_by_rank.items())},
        "query_n": len(eval_ids), "query_rata_keyphrase": float(np.mean(n_kp_q)),
        "query_median_keyphrase": float(np.median(n_kp_q)), "query_tanpa_keyphrase": int(sum(1 for n in n_kp_q if n == 0)),
        "query_present": qp, "query_absent": qa, "query_proporsi_absent": qa / (qp + qa) if qp + qa else 0.0,
        "query_skor_kp_nol_semua_dok": zero_all, "query_proporsi_skor_kp_nol": zero_all / len(eval_ids),
        "median_dok_skor_kp_positif_per_query": float(np.median(n_pos)),
        "query_ndcg_keyphrase_nol": kp_only_zero, "query_proporsi_ndcg_keyphrase_nol": kp_only_zero / len(eval_ids),
        "query_ndcg_semantic_nol": sem_only_zero,
        "pasangan_relevan_skor_kp_positif": cov_hit, "pasangan_relevan_total": cov_total,
        "proporsi_pasangan_relevan_skor_kp_positif": cov_hit / cov_total,
        "rata_dok_relevan_per_query_test": float(np.mean(n_rel_per_q)),
        "query_min1_relevan_skor_kp_positif": q_with_hit,
        "proporsi_query_min1_relevan_skor_kp_positif": q_with_hit / len(eval_ids),
    }
    # --- contoh 5 dokumen (acak deterministik)
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(retriever.documents)), 5))
    examples = []
    for i in idx:
        doc = retriever.documents[i]
        joined = " " + " ".join(_stemmed_tokens(doc.full_text, lang)) + " "
        kps = [(p, round(w, 2), "present" if is_present(p, joined, lang) else "absent")
               for p, w in retriever.doc_keyphrases[i]]
        examples.append({"doc_id": doc.id, "judul": doc.title, "keyphrase": kps})
    return stats, examples


def tabel_keyphrase(stats: dict[str, dict], examples: dict[str, list]) -> str:
    keys = [("Rata-rata keyphrase per dokumen", "dok_rata_keyphrase", 2),
            ("Dokumen tanpa keyphrase", "dok_tanpa_keyphrase", 0),
            ("Proporsi keyphrase dokumen *absent*", "dok_proporsi_absent", 3),
            ("Rata-rata keyphrase per query (test)", "query_rata_keyphrase", 2),
            ("Query tanpa keyphrase", "query_tanpa_keyphrase", 0),
            ("Proporsi keyphrase query *absent*", "query_proporsi_absent", 3),
            ("Query dengan skor keyphrase 0 untuk semua dokumen", "query_skor_kp_nol_semua_dok", 0),
            ("  … sebagai proporsi query test", "query_proporsi_skor_kp_nol", 3),
            ("Median jumlah dokumen dengan skor keyphrase > 0 per query", "median_dok_skor_kp_positif_per_query", 0),
            ("Query dengan nDCG@10 Keyphrase-only = 0", "query_ndcg_keyphrase_nol", 0),
            ("  … sebagai proporsi query test", "query_proporsi_ndcg_keyphrase_nol", 3),
            ("Query dengan nDCG@10 Semantic-only = 0 (pembanding)", "query_ndcg_semantic_nol", 0),
            ("Rata-rata dokumen relevan per query test", "rata_dok_relevan_per_query_test", 1),
            ("Pasangan (query, dokumen relevan) dengan skor keyphrase > 0", "proporsi_pasangan_relevan_skor_kp_positif", 3),
            ("Query dengan ≥ 1 dokumen relevan berskor keyphrase > 0", "query_min1_relevan_skor_kp_positif", 0),
            ("  … sebagai proporsi query test", "proporsi_query_min1_relevan_skor_kp_positif", 3)]
    rows = []
    for label, k, nd in keys:
        rows.append([label] + [(f"{stats[ds][k]:.{nd}f}" if nd else str(int(stats[ds][k]))) for ds in stats])
    write_csv("tabel_keyphrase.csv", ["statistik"] + list(stats), [[k] + [stats[ds][k] for ds in stats] for _, k, _ in keys])
    rank_rows = []
    for i in range(1, 11):
        line = [i]
        for ds in stats:
            line += [fmt(stats[ds]["dok_proporsi_absent_per_urutan"].get(i, float("nan")), 3),
                     stats[ds]["dok_jumlah_frasa_per_urutan"].get(i, 0)]
        rank_rows.append(line)
    md = ["## Statistik keyphrase (BART-KPG)", "",
          md_table(["Statistik"] + [DATASETS[d] for d in stats], rows), "",
          "*Absent* = keyphrase yang setelah `text.normalize_phrase` (huruf kecil, stopword di tepi dibuang, stemming) "
          "tidak muncul sebagai urutan kata utuh di judul + abstrak (atau teks query) yang di-`tokenize` dan di-`stem` "
          "dengan aturan yang sama.", "",
          "### Proporsi *absent* menurut urutan keluaran generator", "",
          md_table(["Urutan ke-"] + [h for d in stats for h in (f"{DATASETS[d]} absent", f"{DATASETS[d]} n frasa")],
                   rank_rows), "",
          "Urutan akhir (9–10) hanya punya sedikit frasa karena kebanyakan dokumen mendapat < 9 keyphrase; "
          "proporsinya tidak stabil.", ""]
    for ds, exs in examples.items():
        md += [f"### Contoh keyphrase dokumen: {DATASETS[ds]} (5 dokumen acak, seed {SEED})", ""]
        for e in exs:
            kps = "; ".join(f"{p} ({w}{', absent' if s == 'absent' else ''})" for p, w, s in e["keyphrase"])
            md.append(f"- **[{e['doc_id']}] {e['judul']}**: {kps}")
        md.append("")
    text = "\n".join(md)
    write_md("tabel_keyphrase.md", text)
    with (OUT / "contoh_keyphrase.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "doc_id", "judul", "urutan", "keyphrase", "bobot", "status"])
        for ds, exs in examples.items():
            for e in exs:
                for i, (p, wt, s) in enumerate(e["keyphrase"], 1):
                    w.writerow([ds, e["doc_id"], e["judul"], i, p, wt, s])
    return text


# ====================================================================== 5. studi kasus
def rankings(table: ScoreTable, qids: list[str], alpha: float) -> dict[str, dict[str, list[str]]]:
    rankers = table.single_signal_rankers()
    rankers["hybrid_weighted"] = table.hybrid_ranker(alpha)
    return {s: table.run(qids, rankers[s]) for s in SCENARIOS}


def first_relevant(ranked: list[str], qrel: dict[str, int]) -> int | None:
    for i, d in enumerate(ranked, 1):
        if qrel.get(d, 0) > 0:
            return i
    return None


def case_study(ds: str, qid: str, kind: str, retriever: HybridRetriever, split, table: ScoreTable,
               runs: dict, qkps: dict, per_query: dict) -> dict:
    lang = retriever.cfg.keyphrase.language
    unit = retriever.cfg.keyphrase.unit
    qrel = split.eval_qrels[qid]
    docs = {d.id: d for d in retriever.documents}
    di = {d: i for i, d in enumerate(retriever.doc_ids)}
    q_units = retriever.kp_index.query_units(qkps[qid])
    q_terms = set(content_tokens(split.eval[qid], lang))
    scen = {}
    for s in SCENARIOS:
        r = runs[s][qid]
        scen[s] = {"ndcg10": per_query[s][qid], "peringkat_relevan_pertama": first_relevant(r, qrel),
                   "top3": [{"doc_id": d, "judul": docs[d].title, "relevansi": qrel.get(d, 0)} for d in r[:3]]}
    # dokumen relevan "utama": relevansi tertinggi, lalu peringkat terbaik di skenario yang dibahas
    focus = "hybrid_weighted" if kind == "hybrid_menolong" else "bm25"
    rels = [d for d, v in qrel.items() if v > 0 and d in di]
    def pos(d, s):
        r = runs[s][qid]
        return r.index(d) + 1 if d in r else 10_000
    main_doc = sorted(rels, key=lambda d: (-qrel[d], pos(d, focus), d))[0]
    row, col = table.row[qid], di[main_doc]
    d_units = unit_weights(retriever.doc_keyphrases[col], lang, unit)
    shared_units = sorted(set(q_units) & set(d_units))
    shared_terms = sorted(q_terms & set(content_tokens(docs[main_doc].full_text, lang)))
    frasa_sama, kata_sama = readable_matches(qkps[qid], shared_units, lang)
    kata_bm25 = surface_words(split.eval[qid], set(shared_terms), lang)
    sem_rank = int((table.sem[row] > table.sem[row, col]).sum()) + 1
    detail = {
        "doc_id": main_doc, "judul": docs[main_doc].title, "relevansi": qrel[main_doc],
        "keyphrase_dokumen": [p for p, _ in retriever.doc_keyphrases[col]],
        "s_sem": float(table.sem[row, col]), "peringkat_sem_seluruh_korpus": sem_rank,
        "s_kp": float(table.kp[row, col]), "s_bm25": float(table.bm25[row, col]),
        "unit_keyphrase_bersama": shared_units, "term_bm25_bersama": shared_terms,
        "frasa_keyphrase_sama": frasa_sama, "kata_keyphrase_sama": kata_sama, "kata_bm25_sama": kata_bm25,
        "peringkat": {s: (pos(main_doc, s) if pos(main_doc, s) < 10_000 else None) for s in SCENARIOS},
    }
    return {"dataset": ds, "qid": qid, "jenis": kind, "query": split.eval[qid],
            "keyphrase_query": [[p, round(w, 2)] for p, w in qkps[qid]],
            "n_relevan": sum(1 for v in qrel.values() if v > 0),
            "selisih_hybrid_semantic": per_query["hybrid_weighted"][qid] - per_query["semantic"][qid],
            "selisih_hybrid_bm25": per_query["hybrid_weighted"][qid] - per_query["bm25"][qid],
            "skenario": scen, "dokumen_relevan_utama": detail, "penjelasan": explain(kind, detail)}


def surface_words(text: str, stems: set[str], lang: str) -> list[str]:
    """Kata asli (belum di-stem) di `text` yang bentuk stem-nya ada di `stems`; urutan kemunculan, tanpa duplikat."""
    out: list[str] = []
    for raw, st in zip(content_tokens(text, lang, stemmed=False), content_tokens(text, lang)):
        if st in stems and raw not in out:
            out.append(raw)
    return out


def readable_matches(qkps: list, shared_units: list[str], lang: str) -> tuple[list[str], list[str]]:
    """Ubah unit internal ('p:kidney pair donat', 't:donat') menjadi bahasa biasa dari keyphrase query:
    (frasa keyphrase query yang sama utuh dengan frasa dokumen, kata keyphrase query yang sama)."""
    shared = set(shared_units)
    phrases = [p for p, _ in qkps if "p:" + normalize_phrase(p, lang) in shared]
    stems = {u[2:] for u in shared if u.startswith("t:")}
    words: list[str] = []
    for p, _ in qkps:
        for w in surface_words(p, stems, lang):
            if w not in words:
                words.append(w)
    return phrases, words


def explain(kind: str, d: dict) -> str:
    """Penjelasan otomatis yang hanya menyusun fakta terukur (bukan tafsiran bebas)."""
    pk = d["peringkat"]
    def r(s):
        return f"#{pk[s]}" if pk[s] else "di luar 100 besar"
    def q(xs):
        return ", ".join(f"'{x}'" for x in xs)
    if d["frasa_keyphrase_sama"]:
        units = f"frasa yang sama {q(d['frasa_keyphrase_sama'])}"
        if d["kata_keyphrase_sama"]:
            units += f" serta kata yang sama {q(d['kata_keyphrase_sama'])}"
    elif d["kata_keyphrase_sama"]:
        units = f"tidak ada frasa yang sama utuh, tetapi ada kata yang sama {q(d['kata_keyphrase_sama'])}"
    else:
        units = "tidak ada frasa maupun kata yang sama"
    terms = q(d["kata_bm25_sama"]) or "tidak ada"
    if kind == "hybrid_menolong":
        return (f"Dokumen relevan [{d['doc_id']}] ada di peringkat {r('semantic')} pada Semantic-only dan naik ke "
                f"{r('hybrid_weighted')} pada Hybrid. Keyphrase query dan keyphrase dokumen: {units} "
                f"(s_kp = {d['s_kp']:.3f}); skor semantiknya {d['s_sem']:.3f}. Sinyal keyphrase menambah skor "
                f"dokumen ini setelah normalisasi min-max, sehingga peringkatnya naik.")
    return (f"Dokumen relevan [{d['doc_id']}] ada di peringkat {r('bm25')} pada BM25 tetapi {r('hybrid_weighted')} "
            f"pada Hybrid ({r('semantic')} pada Semantic-only). Kata query yang juga muncul di dokumen: {terms}. "
            f"Keyphrase query dan keyphrase dokumen: {units} (s_kp = {d['s_kp']:.3f}). BM25 memberi bobot tinggi pada kecocokan kata "
            f"yang langka, sedangkan embedding (s_sem = {d['s_sem']:.3f}, peringkat {d['peringkat_sem_seluruh_korpus']} "
            f"di seluruh korpus) dan keyphrase tidak menangkapnya cukup kuat.")


def pick_cases(per_query: dict, n: int = 3) -> dict[str, list[str]]:
    qids = sorted(per_query["hybrid_weighted"])
    up = sorted(qids, key=lambda q: (-(per_query["hybrid_weighted"][q] - per_query["semantic"][q]), q))
    down = sorted(qids, key=lambda q: (per_query["hybrid_weighted"][q] - per_query["bm25"][q], q))
    return {"hybrid_menolong": up[:n], "hybrid_kalah_bm25": down[:n]}


def studi_kasus_md(cases: list[dict]) -> str:
    md = ["## Studi kasus", "",
          "Dipilih otomatis dari `per_query_ndcg10.csv`: *Hybrid menolong* = selisih nDCG@10 Hybrid − Semantic terbesar; "
          "*Hybrid kalah dari BM25* = selisih Hybrid − BM25 terkecil (seri diputus menurut id query). "
          "✓ = relevan menurut qrels (angka = tingkat relevansi), ✗ = tidak relevan.", ""]
    title = {"hybrid_menolong": "Hybrid menolong dibanding Semantic", "hybrid_kalah_bm25": "Hybrid kalah dari BM25"}
    for c in cases:
        if not c.get("utama"):
            continue
        md += [f"### {title[c['jenis']]}: {DATASETS[c['dataset']]} query {c['qid']}", "",
               f"- **Query:** {c['query']}",
               f"- **Keyphrase query:** {', '.join(f'{p} ({w})' for p, w in c['keyphrase_query']) or '-'}",
               f"- **Dokumen relevan:** {c['n_relevan']}; selisih nDCG@10 Hybrid − Semantic = "
               f"{c['selisih_hybrid_semantic']:+.4f}, Hybrid − BM25 = {c['selisih_hybrid_bm25']:+.4f}", "",
               "| Skenario | nDCG@10 | Relevan pertama | #1 | #2 | #3 |", "|---|---|---|---|---|---|"]
        for s in SCENARIOS:
            sc = c["skenario"][s]
            cells = [f"{'✓' + str(t['relevansi']) if t['relevansi'] > 0 else '✗'} {t['judul'][:60]}" for t in sc["top3"]]
            cells += ["-"] * (3 - len(cells))
            md.append(f"| {NAMA_PENDEK[s]} | {sc['ndcg10']:.4f} | {sc['peringkat_relevan_pertama'] or '>100'} | "
                      + " | ".join(c_.replace('|', '/') for c_ in cells) + " |")
        d = c["dokumen_relevan_utama"]
        md += ["", f"**Dokumen relevan utama:** [{d['doc_id']}] {d['judul']}  ",
               f"Keyphrase dokumen: {', '.join(d['keyphrase_dokumen'])}  ",
               f"**Penyebab (dari angka):** {c['penjelasan']}", ""]
    others = [c for c in cases if not c.get("utama")]
    if others:
        md += ["### Kandidat lain", "", "| Jenis | Dataset | Query | Hybrid − Semantic | Hybrid − BM25 | Teks query |",
               "|---|---|---|---|---|---|"]
        for c in others:
            md.append(f"| {title[c['jenis']]} | {DATASETS[c['dataset']]} | {c['qid']} | {c['selisih_hybrid_semantic']:+.4f} "
                      f"| {c['selisih_hybrid_bm25']:+.4f} | {c['query'][:80]} |")
    text = "\n".join(md)
    write_md("studi_kasus.md", text)
    (OUT / "studi_kasus.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return text


# ====================================================================== 6. lingkungan + waktu
def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def lingkungan() -> tuple[str, dict]:
    pk = ["numpy", "scipy", "torch", "transformers", "sentence-transformers", "snowballstemmer", "streamlit",
          "matplotlib"]
    versions = {"python": platform.python_version()}
    for p in pk:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = "-"
    hw = {}
    # Hanya kunci yang aman dibagikan (TIDAK ada nomor seri / UUID perangkat).
    for line in _cmd(["system_profiler", "SPHardwareDataType"]).splitlines():
        k, _, v = line.strip().partition(":")
        if k in ("Model Name", "Chip", "Total Number of Cores", "Memory"):
            hw[k] = v.strip()
    hw["macOS"] = _cmd(["sw_vers", "-productVersion"])
    mem = _cmd(["sysctl", "-n", "hw.memsize"])
    hw["RAM (sysctl, GB)"] = f"{int(mem) / 2**30:.0f}" if mem.isdigit() else "-"
    hw["CPU (sysctl)"] = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    hw["device torch"] = resolve_device("auto")
    kaggle = {}
    for ds in DATASETS:
        man = json.loads((ROOT / "artifacts" / f"keyphrases_{ds}.jsonl.manifest.json").read_text(encoding="utf-8"))
        kaggle[ds] = {"model": man["model"], "num_beams": man["num_beams"], "top_n": man["top_n"],
                      "entries": man["entries"],
                      "workers": [{k: w[k] for k in ("gpu", "device", "torch", "transformers", "commit", "items", "seconds")}
                                  for w in man["workers"]]}
    env = {"pustaka": versions, "perangkat": hw, "kaggle": kaggle}
    md = ["## Lingkungan", "", "### Komputer lokal (pengindeksan, pencarian, evaluasi)", "",
          md_table(["Komponen", "Nilai"], [[k, v] for k, v in hw.items()]), "",
          md_table(["Pustaka", "Versi"], [[k, v] for k, v in versions.items()]), "",
          "### Kaggle (pembangkitan keyphrase BART-KPG)", ""]
    rows = []
    for ds, k in kaggle.items():
        for i, w in enumerate(k["workers"]):
            rows.append([DATASETS[ds], i, w["gpu"], w["torch"], w["transformers"], w["commit"][:7], w["items"],
                         f"{w['seconds']:.1f}"])
    md += [md_table(["Dataset", "Worker", "GPU", "torch", "transformers", "Commit model", "Teks", "Detik"], rows), "",
           f"Model {kaggle['scifact']['model']}, beam {kaggle['scifact']['num_beams']}, top-n {kaggle['scifact']['top_n']}; "
           f"entri unik (dokumen + query): SciFact {kaggle['scifact']['entries']}, NFCorpus {kaggle['nfcorpus']['entries']}."]
    text = "\n".join(md)
    write_md("lingkungan.md", text)
    (OUT / "lingkungan.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    return text, env


def ukur_waktu(ds: str, retriever: HybridRetriever, queries: list[str], with_bart: bool) -> list[list]:
    from hybrid_ir.bm25 import BM25
    rows = []
    texts = [d.full_text for d in retriever.documents]
    t0 = time.perf_counter()
    BM25(retriever.cfg.keyphrase.language).fit(texts)            # objek baru: ukur pembangunan indeks sungguhan
    rows.append([ds, f"Bangun indeks BM25 di memori ({len(texts)} dokumen, sekali)", 1,
                 (time.perf_counter() - t0) * 1000, ""])
    for s in SCENARIOS:
        mode = MODE_SEARCH[s]
        for q in queries[:5]:                         # pemanasan (cache, MPS)
            retriever.search(q, mode=mode, top_k=10)
        times = []
        for _ in range(ROUNDS):                       # beberapa putaran agar angka milidetik lebih stabil
            for q in queries:
                t = time.perf_counter()
                retriever.search(q, mode=mode, top_k=10)
                times.append((time.perf_counter() - t) * 1000)
        rows.append([ds, NAMA_PENDEK[s], len(queries), statistics.mean(times), statistics.median(times)])
    if with_bart:
        from hybrid_ir.keyphrases import Seq2SeqKeyphraseGenerator
        gen = Seq2SeqKeyphraseGenerator(retriever.cfg.keyphrase)
        sample = queries[:20]
        gen.extract([TextItem(sample[0])], 8)         # pemanasan
        times = []
        for q in sample:
            t = time.perf_counter()
            gen.extract([TextItem(q)], retriever.cfg.keyphrase.query_top_n)
            times.append((time.perf_counter() - t) * 1000)
        rows.append([ds, f"BART-KPG query bebas ({gen.device})", len(times), statistics.mean(times), statistics.median(times)])
    return rows


def tabel_waktu(rows: list[list]) -> str:
    write_csv("tabel_waktu.csv", ["dataset", "langkah", "n_query", "rata_ms", "median_ms"], rows)
    return tabel_waktu_md(rows)


def tabel_waktu_md(rows: list[list]) -> str:
    body = [[DATASETS.get(r[0], r[0]), r[1], r[2], f"{float(r[3]):.1f}", f"{float(r[4]):.1f}" if r[4] != "" else "-"]
            for r in rows]
    md = ("## Waktu pencarian per query (komputer lokal)\n\n"
          + md_table(["Dataset", "Skenario / langkah", "Query", "Rata-rata (ms)", "Median (ms)"], body)
          + "\n\nDiukur dengan `HybridRetriever.search(top_k=10)` atas seluruh query test, 3 putaran, setelah 5 query "
            "pemanasan. "
            "Keyphrase query diambil dari berkas precomputed; untuk query bebas di UI ditambah waktu generasi BART-KPG "
            "(baris terakhir tiap dataset). Waktu bergantung pada mesin dan beban saat pengukuran.")
    write_md("tabel_waktu.md", md)
    return md


# ====================================================================== 7. grafik
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "text.color": C["ink"], "axes.facecolor": C["surface"], "figure.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False, "savefig.dpi": 300,
    })
    return plt


def gambar_batang(metrics: dict[str, dict]) -> Path:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    order = list(SCENARIOS)
    for ax, (ds, m) in zip(axes, metrics.items()):
        vals = [m["scenarios"][s]["metrics"]["ndcg@10"] for s in order]
        ys = np.arange(len(order))[::-1]
        colors = [C["blue"] if s == "hybrid_weighted" else C["neutral_bar"] for s in order]
        ax.barh(ys, vals, height=0.55, color=colors, edgecolor=C["surface"], linewidth=1)
        labels = [NAMA_PENDEK[s] + (f"\n(α = {m['alpha_selected']})" if s == "hybrid_weighted" else "") for s in order]
        ax.set_yticks(ys, labels)
        xmax = max(vals) * 1.22
        ax.set_xlim(0, xmax)
        for y, v in zip(ys, vals):
            ax.text(v + xmax * 0.012, y, f"{v:.4f}", va="center", ha="left", fontsize=8, color=C["ink"])
        ax.xaxis.grid(True, color=C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{DATASETS[ds]} ({m['n_eval_queries']} query test)", loc="left")
        ax.set_xlabel("nDCG@10")
    fig.tight_layout(w_pad=2.5)
    path = FIG / "g1_ndcg10_skenario.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def gambar_alpha(metrics: dict[str, dict]) -> Path:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, (ds, m) in zip(axes, metrics.items()):
        a = [r["alpha"] for r in m["alpha_sweep_tune"]]
        tune = [r["ndcg@10"] for r in m["alpha_sweep_tune"]]
        ev = [r["ndcg@10"] for r in m["alpha_sweep_eval_posthoc"]]
        bm = m["scenarios"]["bm25"]["metrics"]["ndcg@10"]
        ax.yaxis.grid(True, color=C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axhline(bm, color=C["muted"], linewidth=1, linestyle=(0, (1, 2)))
        ax.text(0.01, bm, " BM25 (test)", va="bottom", ha="left", fontsize=7.5, color=C["ink2"])
        ax.plot(a, tune, color=C["blue"], linewidth=2, marker="o", markersize=3.5,
                label="query penalaan (SciFact: train, NFCorpus: dev)")
        ax.plot(a, ev, color=C["orange"], linewidth=2, linestyle=(0, (4, 2)), marker="s", markersize=3.2,
                label="query test (hanya analisis)")
        s = m["alpha_selected"]
        ax.axvline(s, color=C["ink2"], linewidth=0.8)
        ax.text(s, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04, f" α terpilih = {s}",
                fontsize=7.5, color=C["ink"], ha="right", va="bottom")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("α (bobot semantik; 1 − α untuk keyphrase)")
        ax.set_ylabel("nDCG@10")
        ax.set_title(DATASETS[ds], loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
    path = FIG / "g2_pengaruh_alpha.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def gambar_selisih(per_query: dict[str, dict], msk: dict) -> Path:
    plt = _style()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.2), sharey=True)
    for i, ds in enumerate(per_query):
        for j, other in enumerate(("semantic", "bm25")):
            ax = axes[i][j]
            pq = per_query[ds]
            d = np.sort(np.round([pq["hybrid_weighted"][q] - pq[other][q] for q in pq["hybrid_weighted"]], 6))[::-1]
            x = np.arange(len(d))
            colors = np.where(d > 0, C["blue"], np.where(d < 0, C["red"], C["neutral"]))
            ties = np.flatnonzero(d == 0)
            if ties.size:                                   # seri bernilai 0 -> tandai rentangnya dengan latar abu-abu
                ax.axvspan(ties[0] - 0.5, ties[-1] + 0.5, color=C["tie_bg"], linewidth=0, zorder=0)
            ax.bar(x, d, width=1.0, color=colors, linewidth=0, zorder=2)
            ax.axhline(0, color=C["axis"], linewidth=0.8)
            s = msk[ds][other]
            ax.text(0.98, 0.95, f"menang {s['menang']} · seri {s['seri']} · kalah {s['kalah']}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=C["ink"])
            ax.text(0.98, 0.83, f"rata-rata selisih {s['rata_selisih']:+.4f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=C["ink2"])
            ax.set_xlim(-1, len(d))
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([])
            ax.yaxis.grid(True, color=C["grid"], linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_title(f"{DATASETS[ds]}: Hybrid − {NAMA_PENDEK[other]}", loc="left")
            if j == 0:
                ax.set_ylabel("selisih nDCG@10")
            if i == 1:
                ax.set_xlabel("query test, diurutkan menurut selisih")
    from matplotlib.patches import Patch
    fig.legend([Patch(color=C["blue"]), Patch(color=C["tie_bg"]), Patch(color=C["red"])],
               ["Hybrid lebih baik", "seri (selisih 0)", "Hybrid lebih buruk"], loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.5, w_pad=1.5)
    path = FIG / "g3_selisih_per_query.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ====================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tanpa-waktu", action="store_true", help="jangan ukur waktu; pakai analisis/tabel_waktu.csv lama")
    ap.add_argument("--tanpa-bart", action="store_true", help="jangan muat BART untuk mengukur waktu query bebas")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.chdir(ROOT)                                        # path di konfigurasi relatif terhadap hybrid-ir/
    FIG.mkdir(parents=True, exist_ok=True)

    metrics = {ds: load_metrics(ds) for ds in DATASETS}
    per_query = {ds: load_per_query(ds) for ds in DATASETS}
    sections: list[str] = []
    ringkasan: dict = {}

    md, ringkasan["tabel_utama"] = tabel_utama(metrics); sections.append(md)
    md, ringkasan["signifikansi"] = tabel_signifikansi(metrics); sections.append(md)
    md, ringkasan["alpha"] = tabel_alpha(metrics); sections.append(md)
    md, ringkasan["menang_seri_kalah"] = menang_seri_kalah(per_query); sections.append(md)

    datasets, kp_stats, kp_examples, cases, waktu = {}, {}, {}, [], []
    for ds in DATASETS:
        log.info("== %s: memuat indeks dan menghitung ulang skor ...", ds)
        cfg = load_config(f"configs/{ds}_bart.yaml")
        dset = load_beir_dir(cfg.data.path, cfg.data.max_docs)
        datasets[ds] = dset
        retriever = HybridRetriever.load(cfg)
        m = metrics[ds]
        split = _split_queries(dset, cfg.data.eval_split, cfg.data.tune_split, cfg.data.dev_fraction, SEED)
        table = ScoreTable.compute(retriever, split)
        eval_ids = sorted(split.eval)
        runs = rankings(table, eval_ids, m["alpha_selected"])
        # pengaman: hitung ulang harus sama persis dengan results/
        for s in SCENARIOS:
            for q in eval_ids:
                got = round(ndcg_at_k(runs[s][q], split.eval_qrels[q], 10), 6)
                if abs(got - per_query[ds][s][q]) > 5e-7:
                    raise SystemExit(f"GALAT: nDCG@10 {ds}/{s}/{q} hitung ulang {got} != results {per_query[ds][s][q]}")
        log.info("   nDCG@10 hitung ulang cocok dengan per_query_ndcg10.csv (%d query × 4 skenario)", len(eval_ids))
        kp_stats[ds], kp_examples[ds] = keyphrase_stats(ds, retriever, split, table, per_query[ds])
        texts = [retriever.validate_query(split.eval[q]) for q in eval_ids]
        qk = retriever.query_extractor.extract([TextItem(t) for t in texts], cfg.keyphrase.query_top_n)
        qkps = dict(zip(eval_ids, qk))
        picks = pick_cases(per_query[ds])
        for kind, qids in picks.items():
            for rank, q in enumerate(qids):
                c = case_study(ds, q, kind, retriever, split, table, runs, qkps, per_query[ds])
                c["utama"] = rank == 0
                cases.append(c)
        if not args.tanpa_waktu:
            log.info("   mengukur waktu pencarian ...")
            waktu += ukur_waktu(ds, retriever, texts, with_bart=not args.tanpa_bart)
        if getattr(retriever.query_extractor, "_fallback", None) is not None:
            raise SystemExit("GALAT: ada query evaluasi yang tidak ada di berkas precomputed (BART fallback termuat)")

    md, ringkasan["dataset"] = tabel_dataset(datasets); sections.insert(0, md)
    sections.append(tabel_keyphrase(kp_stats, kp_examples))
    ringkasan["keyphrase"] = kp_stats
    sections.append(studi_kasus_md(cases))
    ringkasan["studi_kasus"] = [{k: c[k] for k in ("dataset", "qid", "jenis", "utama", "selisih_hybrid_semantic",
                                                   "selisih_hybrid_bm25")} for c in cases]
    if waktu:
        sections.append(tabel_waktu(waktu))
    elif (OUT / "tabel_waktu.csv").is_file():
        with (OUT / "tabel_waktu.csv").open(encoding="utf-8") as fh:
            waktu = list(csv.reader(fh))[1:]
        sections.append(tabel_waktu_md(waktu))
    ringkasan["waktu"] = [{"dataset": r[0], "langkah": r[1], "n": int(r[2]), "rata_ms": float(r[3]),
                           "median_ms": float(r[4]) if r[4] != "" else None} for r in waktu]
    md, ringkasan["lingkungan"] = lingkungan(); sections.append(md)

    figs = [gambar_batang(metrics), gambar_alpha(metrics), gambar_selisih(per_query, ringkasan["menang_seri_kalah"])]
    (OUT / "ringkasan.json").write_text(json.dumps(ringkasan, indent=2, ensure_ascii=False), encoding="utf-8")

    head = ["# Analisis UAS: Hybrid Semantic-Keyphrase Retrieval", "",
            "Dibangkitkan oleh `scripts/analisis_uas.py` dari `results/`, `indexes/`, `artifacts/`, dan `data/`. "
            "Jangan diedit tangan; jalankan ulang skripnya.", "", "## Gambar", ""]
    head += [f"- `gambar/{p.name}`" for p in figs]
    head += ["", "![nDCG@10](gambar/g1_ndcg10_skenario.png)", "", "![alpha](gambar/g2_pengaruh_alpha.png)", "",
             "![selisih](gambar/g3_selisih_per_query.png)", ""]
    write_md("README.md", "\n".join(head) + "\n" + "\n\n".join(sections))
    log.info("Selesai: %s", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
