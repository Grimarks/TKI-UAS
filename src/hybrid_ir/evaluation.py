"""Evaluasi: metrik IR, perbandingan skenario, penalaan alpha, dan uji signifikansi.

Skenario yang dibandingkan (tepat empat, sesuai diagram pipeline):
  bm25 (pembanding) | semantic-only | keyphrase-only | hybrid (weighted, alpha hasil tuning)

Cara kerja yang adil:
  * alpha ditala HANYA pada query "tune" (split train, atau 30% query evaluasi bila tidak ada train),
    lalu dilaporkan pada query "eval" yang tidak dipakai untuk menala.
  * Query tanpa dokumen relevan di korpus dilewati.
  * Kemaknaan perbedaan diuji dengan paired bootstrap atas nDCG@10 per query:
    hybrid dibandingkan dengan semantic, keyphrase, dan bm25.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import Dataset, split_query_ids, usable_queries
from .fusion import fuse_weighted, positive_topk, topk_indices
from .retriever import HybridRetriever

DEPTH = 100
SCENARIOS = ("bm25", "semantic", "keyphrase", "hybrid_weighted")
SIGNIFICANCE_PAIRS = (("hybrid_weighted", "semantic"), ("hybrid_weighted", "keyphrase"), ("hybrid_weighted", "bm25"))


# ------------------------------------------------------------------------------ metrik per query
def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    gains = [max(qrel.get(d, 0), 0) for d in ranked[:k]]
    ideal = sorted((max(v, 0) for v in qrel.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    return sum(1 for d in ranked[:k] if qrel.get(d, 0) > 0) / k


def recall_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
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
    for i, d in enumerate(ranked[:k], start=1):
        if qrel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def query_metrics(ranked: Sequence[str], qrel: dict[str, int]) -> dict[str, float]:
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


# ------------------------------------------------------------------------------ uji signifikansi
def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 13) -> dict:
    """Uji bootstrap berpasangan untuk selisih rata-rata a - b (dua sisi)."""
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


# ------------------------------------------------------------------------------ eksperimen
def _split_queries(dataset: Dataset, eval_split: str, tune_split: str | None, dev_fraction: float, seed: int):
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
    return tune_q, tune_qrels, eval_q, dataset.qrels_for(eval_split), source


def _subset(qrels: dict[str, dict[str, int]], qids) -> dict[str, dict[str, int]]:
    """Ambil qrels hanya untuk query yang benar-benar dijalankan (agar query lain tidak dihitung nol)."""
    return {q: qrels[q] for q in qids}


def run_experiment(retriever: HybridRetriever, dataset: Dataset, eval_split: str = "test",
                   tune_split: str | None = None, dev_fraction: float = 0.3,
                   alphas: Sequence[float] | None = None, seed: int = 13) -> tuple[dict, dict[str, dict[str, float]]]:
    alphas = list(alphas) if alphas is not None else [round(a, 2) for a in np.linspace(0, 1, 11)]
    tune_q, tune_qrels, eval_q, eval_qrels, tune_source = _split_queries(dataset, eval_split, tune_split, dev_fraction, seed)

    qids = list(tune_q) + list(eval_q)
    texts = [tune_q.get(q) or eval_q[q] for q in qids]
    row = {q: i for i, q in enumerate(qids)}
    sem, kp, _ = retriever.score_matrices(texts)
    bm = retriever.bm25_scores(texts)
    ids = retriever.doc_ids
    cand = retriever.cfg.retrieval.candidate_k

    def to_ids(idx: np.ndarray) -> list[str]:
        return [ids[int(i)] for i in idx[:DEPTH]]

    # bm25 dan keyphrase: dokumen berskor 0 (tak ada kecocokan) dibuang, sama seperti mode bm25/keyphrase di search
    rankers = {
        "bm25": lambda r: positive_topk(bm[r], DEPTH),
        "semantic": lambda r: topk_indices(sem[r], DEPTH),
        "keyphrase": lambda r: positive_topk(kp[r], DEPTH),
    }

    def run_for(qid_list, ranker) -> dict[str, list[str]]:
        return {q: to_ids(ranker(row[q])) for q in qid_list}

    def weighted(alpha: float):
        return lambda r: fuse_weighted(sem[r], kp[r], alpha, max(cand, DEPTH))[0]

    # --- tuning alpha pada query tune
    tune_sweep = []
    for a in alphas:
        s, _ = evaluate_run(run_for(tune_q, weighted(a)), _subset(tune_qrels, tune_q))
        tune_sweep.append({"alpha": a, **s})
    best = max(tune_sweep, key=lambda r: (r["ndcg@10"], -abs(r["alpha"] - 0.5)))
    alpha_star = float(best["alpha"])

    # --- laporan pada query eval
    rankers["hybrid_weighted"] = weighted(alpha_star)
    scenarios: dict[str, dict] = {}
    per_query_all: dict[str, dict[str, dict[str, float]]] = {}
    for name, ranker in rankers.items():
        summary, per_q = evaluate_run(run_for(eval_q, ranker), _subset(eval_qrels, eval_q))
        params = {"alpha": alpha_star} if name == "hybrid_weighted" else {}
        scenarios[name] = {"metrics": summary, "params": params}
        per_query_all[name] = per_q

    eval_sweep = []
    for a in alphas:
        s, _ = evaluate_run(run_for(eval_q, weighted(a)), _subset(eval_qrels, eval_q))
        eval_sweep.append({"alpha": a, **s})

    qid_order = sorted(per_query_all["hybrid_weighted"])
    sig = []
    for a_name, other in SIGNIFICANCE_PAIRS:
        test = paired_bootstrap([per_query_all[a_name][q]["ndcg@10"] for q in qid_order],
                                [per_query_all[other][q]["ndcg@10"] for q in qid_order], seed=seed)
        sig.append({"a": a_name, "b": other, "metric": "ndcg@10", **test})

    result = {
        "dataset": dataset.name,
        "n_docs": len(ids),
        "eval_split": eval_split,
        "tune_source": tune_source,
        "n_tune_queries": len(tune_q),
        "n_eval_queries": len(eval_q),
        "alpha_selected": alpha_star,
        "alpha_sweep_tune": tune_sweep,
        "alpha_sweep_eval_posthoc": eval_sweep,
        "scenarios": scenarios,
        "significance": sig,
        "index_meta": retriever.meta,
    }
    flat = {name: {q: m["ndcg@10"] for q, m in pq.items()} for name, pq in per_query_all.items()}
    return result, flat


# ------------------------------------------------------------------------------ keluaran
def format_markdown(result: dict) -> str:
    names = {"bm25": "BM25 (pembanding)", "semantic": "Semantic-only", "keyphrase": "Keyphrase-only",
             "hybrid_weighted": f"Hybrid weighted (alpha={result['alpha_selected']})"}
    cols = ["ndcg@10", "map@100", "recall@10", "recall@100", "precision@10", "mrr@10"]
    lines = [
        f"# Hasil evaluasi: {result['dataset']}",
        "",
        f"- Dokumen: {result['n_docs']}; query tuning: {result['n_tune_queries']} ({result['tune_source']}); "
        f"query evaluasi: {result['n_eval_queries']} (split `{result['eval_split']}`)",
        f"- alpha dipilih pada query tuning: **{result['alpha_selected']}**",
        "",
        "| Skenario | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for key in SCENARIOS:
        m = result["scenarios"][key]["metrics"]
        lines.append(f"| {names[key]} | " + " | ".join(f"{m[c]:.4f}" for c in cols) + " |")
    lines += ["", "## Uji signifikansi (paired bootstrap, nDCG@10, model dikurangi pembanding)", "",
              "| Model | Pembanding | selisih rata-rata | CI 95% | p |", "|---|---|---|---|---|"]
    for s in result["significance"]:
        lines.append(f"| {s['a']} | {s['b']} | {s['mean_diff']:+.4f} | [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
                     f"| {s['p_value']:.4f} |")
    lines += ["", "## Sapuan alpha (query evaluasi; hanya untuk analisis, jangan dipakai memilih alpha)", "",
              "| alpha | nDCG@10 | recall@100 |", "|---|---|---|"]
    for r in result["alpha_sweep_eval_posthoc"]:
        lines.append(f"| {r['alpha']} | {r['ndcg@10']:.4f} | {r['recall@100']:.4f} |")
    return "\n".join(lines) + "\n"


def write_outputs(result: dict, per_query_ndcg: dict[str, dict[str, float]], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.md").write_text(format_markdown(result), encoding="utf-8")
    names = list(per_query_ndcg)
    qids = sorted(next(iter(per_query_ndcg.values())))
    with (out / "per_query_ndcg10.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", *names])
        for q in qids:
            w.writerow([q, *[f"{per_query_ndcg[n].get(q, float('nan')):.6f}" for n in names]])
    return out
