"""Evaluasi: metrik IR, empat skenario, penalaan alpha, dan uji signifikansi (perintah `hybrid-ir evaluate`).

Peran di pipeline (Gambar 1)
    Fase 3 — "Data uji BEIR" → "Skenario" (BM25 | semantic | keyphrase | hybrid) → "Tuning α" (split tune)
             → "Metrik" (split test) → "Analisis" (paired bootstrap) → results/<dataset>/metrics.{json,md}.

Masukan  : `HybridRetriever` yang sudah dimuat dan `Dataset` (query + qrels).
Keluaran : dict hasil (ditulis ke metrics.json/metrics.md) + nDCG@10 per query (per_query_ndcg10.csv).

Rumus (nomor persamaan sesuai laporan)
    (8) DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)
    (9) nDCG@k = DCG@k / IDCG@k                      IDCG = DCG dari urutan ideal
    AP@100 dinormalkan oleh jumlah SEMUA dokumen relevan (konvensi trec_eval/BEIR); MAP = rata-ratanya.

Cara kerja yang adil:
  * alpha ditala HANYA pada query "tune" (SciFact: train, NFCorpus: dev), lalu dilaporkan pada query
    "eval" (test) yang tidak dipakai untuk menala. Sapuan alpha pada query eval hanya untuk analisis.
  * Query tanpa dokumen relevan di korpus dilewati.
  * Kemaknaan perbedaan diuji dengan paired bootstrap atas nDCG@10 per query:
    hybrid dibandingkan dengan semantic, keyphrase, dan bm25.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .data import Dataset, split_query_ids, usable_queries
from .fusion import fuse_weighted, positive_topk, topk_indices
from .retriever import HybridRetriever

DEPTH = 100   # panjang daftar peringkat yang dinilai (cukup untuk MAP@100 dan Recall@100)
SCENARIOS = ("bm25", "semantic", "keyphrase", "hybrid_weighted")
SIGNIFICANCE_PAIRS = (("hybrid_weighted", "semantic"), ("hybrid_weighted", "keyphrase"), ("hybrid_weighted", "bm25"))
METRIC_COLUMNS = ["ndcg@10", "map@100", "recall@10", "recall@100", "precision@10", "mrr@10"]

Ranker = Callable[[int], np.ndarray]   # baris query -> indeks dokumen terurut


# ====================================================================== Metrik per query
def _dcg(gains: Sequence[float]) -> float:
    """Persamaan (8). i dihitung dari 0 di kode, sehingga penyebutnya log2(i + 2)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    """Persamaan (9): DCG@k peringkat sistem dibagi DCG@k urutan ideal (relevansi bertingkat)."""
    gains = [max(qrel.get(d, 0), 0) for d in ranked[:k]]
    ideal = sorted((max(v, 0) for v in qrel.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    """Proporsi dokumen relevan di k teratas (pembagi selalu k)."""
    return sum(1 for d in ranked[:k] if qrel.get(d, 0) > 0) / k


def recall_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    """Proporsi dokumen relevan yang berhasil masuk k teratas."""
    n_rel = sum(1 for v in qrel.values() if v > 0)
    return sum(1 for d in ranked[:k] if qrel.get(d, 0) > 0) / n_rel if n_rel else 0.0


def average_precision(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    """AP@k dinormalkan oleh jumlah SEMUA dokumen relevan (konvensi trec_eval/BEIR)."""
    n_rel = sum(1 for v in qrel.values() if v > 0)
    if not n_rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if qrel.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / n_rel


def mrr_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    """1 / posisi dokumen relevan pertama di k teratas (0 bila tidak ada)."""
    for i, d in enumerate(ranked[:k], start=1):
        if qrel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def query_metrics(ranked: Sequence[str], qrel: dict[str, int]) -> dict[str, float]:
    """Keenam metrik yang dilaporkan untuk satu query."""
    return {
        "ndcg@10": ndcg_at_k(ranked, qrel, 10),
        "map@100": average_precision(ranked, qrel, 100),
        "recall@10": recall_at_k(ranked, qrel, 10),
        "recall@100": recall_at_k(ranked, qrel, 100),
        "precision@10": precision_at_k(ranked, qrel, 10),
        "mrr@10": mrr_at_k(ranked, qrel, 10),
    }


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Rata-rata metrik atas query yang punya dokumen relevan, plus metrik per query."""
    per_query = {qid: query_metrics(run.get(qid, []), rel)
                 for qid, rel in qrels.items() if any(v > 0 for v in rel.values())}
    if not per_query:
        raise ValueError("tidak ada query dengan dokumen relevan untuk dievaluasi")
    names = next(iter(per_query.values())).keys()
    summary = {m: float(np.mean([pq[m] for pq in per_query.values()])) for m in names}
    return summary, per_query


# ====================================================================== Analisis: uji signifikansi
def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 13) -> dict:
    """Uji bootstrap berpasangan untuk selisih rata-rata a − b (dua sisi).

    Langkah: hitung selisih per query; ambil sampel ulang n query (dengan pengembalian) sebanyak n_boot kali;
    CI 95% = persentil 2.5 dan 97.5 dari rata-rata sampel; p = 2·min(P(rata-rata ≤ 0), P(rata-rata ≥ 0)).
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = diff.size
    if n < 2:
        return {"mean_diff": float("nan"), "ci95": [float("nan"), float("nan")], "p_value": float("nan"), "n": int(n)}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    step = max(1, 2_000_000 // n)          # batasi memori
    for s in range(0, n_boot, step):
        e = min(n_boot, s + step)
        means[s:e] = diff[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / n_boot)))
    return {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "p_value": p, "n": int(n)}


# ====================================================================== Eksperimen (Fase 3)
@dataclass
class QuerySplit:
    """Query untuk menala alpha (tune) dan query untuk melaporkan hasil (eval), beserta qrels-nya."""

    tune: dict[str, str]
    tune_qrels: dict[str, dict[str, int]]
    eval: dict[str, str]
    eval_qrels: dict[str, dict[str, int]]
    source: str


@dataclass
class ScoreTable:
    """Skor mentah SEMUA query (tune + eval) terhadap SEMUA dokumen untuk ketiga sinyal; dihitung sekali."""

    row: dict[str, int]       # qid -> baris pada matriks skor
    sem: np.ndarray           # (Q, N) persamaan (1)
    kp: np.ndarray            # (Q, N) persamaan (4)
    bm25: np.ndarray          # (Q, N) persamaan (5)
    doc_ids: list[str]
    candidate_k: int

    @classmethod
    def compute(cls, retriever: HybridRetriever, split: QuerySplit) -> "ScoreTable":
        """Encode/ekstrak tiap query satu kali; semua skenario dan nilai alpha memakai ulang skor ini."""
        qids = list(split.tune) + list(split.eval)
        texts = [split.tune.get(q) or split.eval[q] for q in qids]
        sem, kp, _ = retriever.score_matrices(texts)
        bm25 = retriever.bm25_scores(texts)
        return cls({q: i for i, q in enumerate(qids)}, sem, kp, bm25, retriever.doc_ids,
                   retriever.cfg.retrieval.candidate_k)

    def run(self, qids: Iterable[str], ranker: Ranker) -> dict[str, list[str]]:
        """Daftar peringkat (id dokumen, maksimal DEPTH) untuk tiap query memakai `ranker`."""
        return {q: [self.doc_ids[int(i)] for i in ranker(self.row[q])[:DEPTH]] for q in qids}

    def single_signal_rankers(self) -> dict[str, Ranker]:
        """Tiga skenario satu sinyal. bm25/keyphrase membuang dokumen berskor 0 (sama seperti mode search)."""
        return {
            "bm25": lambda r: positive_topk(self.bm25[r], DEPTH),
            "semantic": lambda r: topk_indices(self.sem[r], DEPTH),
            "keyphrase": lambda r: positive_topk(self.kp[r], DEPTH),
        }

    def hybrid_ranker(self, alpha: float) -> Ranker:
        """Skenario hybrid: fusi linear (7) dengan kandidat C = max(candidate_k, DEPTH)."""
        return lambda r: fuse_weighted(self.sem[r], self.kp[r], alpha, max(self.candidate_k, DEPTH))[0]


def _split_queries(dataset: Dataset, eval_split: str, tune_split: str | None, dev_fraction: float,
                   seed: int) -> QuerySplit:
    """Pilih query tune dan eval. Query yang ada di keduanya hanya dipakai sebagai eval."""
    eval_q = usable_queries(dataset, eval_split)
    if tune_split:
        tune_q = {q: t for q, t in usable_queries(dataset, tune_split).items() if q not in eval_q}
        tune_qrels = dataset.qrels_for(tune_split)
        source = f"split:{tune_split}"
    else:
        dev_ids, eval_ids = split_query_ids(list(eval_q), dev_fraction, seed)
        tune_q = {q: eval_q[q] for q in dev_ids}
        tune_qrels = dataset.qrels_for(eval_split)
        eval_q = {q: eval_q[q] for q in eval_ids}
        source = f"dev-fraction:{dev_fraction}"
    if not tune_q or not eval_q:
        raise ValueError("query untuk tuning/evaluasi kosong; periksa split atau perbesar dataset")
    return QuerySplit(tune_q, tune_qrels, eval_q, dataset.qrels_for(eval_split), source)


def _subset(qrels: dict[str, dict[str, int]], qids) -> dict[str, dict[str, int]]:
    """Ambil qrels hanya untuk query yang benar-benar dijalankan (agar query lain tidak dihitung nol)."""
    return {q: qrels[q] for q in qids}


def _alpha_sweep(table: ScoreTable, queries: dict[str, str], qrels: dict[str, dict[str, int]],
                 alphas: Sequence[float]) -> list[dict]:
    """Tuning α: metrik hybrid untuk setiap nilai alpha pada sekumpulan query."""
    sweep = []
    for a in alphas:
        s, _ = evaluate_run(table.run(queries, table.hybrid_ranker(a)), _subset(qrels, queries))
        sweep.append({"alpha": a, **s})
    return sweep


def _select_alpha(tune_sweep: list[dict]) -> float:
    """Alpha dengan nDCG@10 tertinggi pada query tune; bila seri, pilih yang paling dekat ke 0.5."""
    best = max(tune_sweep, key=lambda r: (r["ndcg@10"], -abs(r["alpha"] - 0.5)))
    return float(best["alpha"])


def _evaluate_scenarios(table: ScoreTable, split: QuerySplit, alpha_star: float
                        ) -> tuple[dict[str, dict], dict[str, dict[str, dict[str, float]]]]:
    """Skenario → Metrik pada query eval, untuk keempat skenario (urutan: bm25, semantic, keyphrase, hybrid)."""
    rankers = table.single_signal_rankers()
    rankers["hybrid_weighted"] = table.hybrid_ranker(alpha_star)
    scenarios: dict[str, dict] = {}
    per_query_all: dict[str, dict[str, dict[str, float]]] = {}
    for name, ranker in rankers.items():
        summary, per_q = evaluate_run(table.run(split.eval, ranker), _subset(split.eval_qrels, split.eval))
        params = {"alpha": alpha_star} if name == "hybrid_weighted" else {}
        scenarios[name] = {"metrics": summary, "params": params}
        per_query_all[name] = per_q
    return scenarios, per_query_all


def _significance_tests(per_query_all: dict[str, dict[str, dict[str, float]]], seed: int) -> list[dict]:
    """Analisis: paired bootstrap nDCG@10 untuk tiap pasangan di SIGNIFICANCE_PAIRS (hybrid − pembanding)."""
    qid_order = sorted(per_query_all["hybrid_weighted"])
    sig = []
    for a_name, other in SIGNIFICANCE_PAIRS:
        test = paired_bootstrap([per_query_all[a_name][q]["ndcg@10"] for q in qid_order],
                                [per_query_all[other][q]["ndcg@10"] for q in qid_order], seed=seed)
        sig.append({"a": a_name, "b": other, "metric": "ndcg@10", **test})
    return sig


def run_experiment(retriever: HybridRetriever, dataset: Dataset, eval_split: str = "test",
                   tune_split: str | None = None, dev_fraction: float = 0.3,
                   alphas: Sequence[float] | None = None, seed: int = 13) -> tuple[dict, dict[str, dict[str, float]]]:
    """Jalankan seluruh Fase 3 dan kembalikan (hasil untuk metrics.json, nDCG@10 per query per skenario)."""
    alphas = list(alphas) if alphas is not None else [round(a, 2) for a in np.linspace(0, 1, 11)]
    split = _split_queries(dataset, eval_split, tune_split, dev_fraction, seed)
    table = ScoreTable.compute(retriever, split)

    tune_sweep = _alpha_sweep(table, split.tune, split.tune_qrels, alphas)
    alpha_star = _select_alpha(tune_sweep)
    scenarios, per_query_all = _evaluate_scenarios(table, split, alpha_star)
    eval_sweep = _alpha_sweep(table, split.eval, split.eval_qrels, alphas)   # hanya untuk analisis
    sig = _significance_tests(per_query_all, seed)

    result = {
        "dataset": dataset.name,
        "n_docs": len(table.doc_ids),
        "eval_split": eval_split,
        "tune_source": split.source,
        "n_tune_queries": len(split.tune),
        "n_eval_queries": len(split.eval),
        "alpha_selected": alpha_star,
        "alpha_sweep_tune": tune_sweep,
        "alpha_sweep_eval_posthoc": eval_sweep,
        "scenarios": scenarios,
        "significance": sig,
        "index_meta": retriever.meta,
    }
    flat = {name: {q: m["ndcg@10"] for q, m in pq.items()} for name, pq in per_query_all.items()}
    return result, flat


# ====================================================================== Keluaran (results/<dataset>/)
def _scenario_table_lines(result: dict) -> list[str]:
    names = {"bm25": "BM25 (pembanding)", "semantic": "Semantic-only", "keyphrase": "Keyphrase-only",
             "hybrid_weighted": f"Hybrid weighted (alpha={result['alpha_selected']})"}
    lines = ["| Skenario | " + " | ".join(METRIC_COLUMNS) + " |", "|---|" + "---|" * len(METRIC_COLUMNS)]
    for key in SCENARIOS:
        m = result["scenarios"][key]["metrics"]
        lines.append(f"| {names[key]} | " + " | ".join(f"{m[c]:.4f}" for c in METRIC_COLUMNS) + " |")
    return lines


def _significance_lines(result: dict) -> list[str]:
    lines = ["## Uji signifikansi (paired bootstrap, nDCG@10, model dikurangi pembanding)", "",
             "| Model | Pembanding | selisih rata-rata | CI 95% | p |", "|---|---|---|---|---|"]
    for s in result["significance"]:
        lines.append(f"| {s['a']} | {s['b']} | {s['mean_diff']:+.4f} | [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
                     f"| {s['p_value']:.4f} |")
    return lines


def _eval_sweep_lines(result: dict) -> list[str]:
    lines = ["## Sapuan alpha (query evaluasi; hanya untuk analisis, jangan dipakai memilih alpha)", "",
             "| alpha | nDCG@10 | recall@100 |", "|---|---|---|"]
    for r in result["alpha_sweep_eval_posthoc"]:
        lines.append(f"| {r['alpha']} | {r['ndcg@10']:.4f} | {r['recall@100']:.4f} |")
    return lines


def format_markdown(result: dict) -> str:
    """Ringkasan hasil untuk metrics.md: tabel skenario, uji signifikansi, dan sapuan alpha pada query eval."""
    header = [
        f"# Hasil evaluasi: {result['dataset']}",
        "",
        f"- Dokumen: {result['n_docs']}; query tuning: {result['n_tune_queries']} ({result['tune_source']}); "
        f"query evaluasi: {result['n_eval_queries']} (split `{result['eval_split']}`)",
        f"- alpha dipilih pada query tuning: **{result['alpha_selected']}**",
        "",
    ]
    lines = header + _scenario_table_lines(result) + [""] + _significance_lines(result) + [""] + _eval_sweep_lines(result)
    return "\n".join(lines) + "\n"


def _write_per_query_csv(per_query_ndcg: dict[str, dict[str, float]], path: Path) -> None:
    names = list(per_query_ndcg)
    qids = sorted(next(iter(per_query_ndcg.values())))
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", *names])
        for q in qids:
            w.writerow([q, *[f"{per_query_ndcg[n].get(q, float('nan')):.6f}" for n in names]])


def write_outputs(result: dict, per_query_ndcg: dict[str, dict[str, float]], out_dir: str | Path) -> Path:
    """Tulis metrics.json, metrics.md, dan per_query_ndcg10.csv ke `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.md").write_text(format_markdown(result), encoding="utf-8")
    _write_per_query_csv(per_query_ndcg, out / "per_query_ndcg10.csv")
    return out
