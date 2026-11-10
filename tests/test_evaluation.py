import math

import numpy as np
import pytest

from hybrid_ir.bm25 import BM25
from hybrid_ir.evaluation import (average_precision, evaluate_run, mrr_at_k, ndcg_at_k, paired_bootstrap,
                                  precision_at_k, recall_at_k)

QREL = {"a": 2, "b": 1, "c": 0}


def test_ndcg_perfect_and_hand_computed():
    assert ndcg_at_k(["a", "b"], QREL, 10) == pytest.approx(1.0)
    # urutan terbalik: DCG = 1/log2(2) + 2/log2(3); IDCG = 2/log2(2) + 1/log2(3)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], QREL, 10) == pytest.approx(expected)
    assert ndcg_at_k(["x", "y"], QREL, 10) == 0.0


def test_precision_recall_map_mrr():
    run = ["x", "a", "c", "b"]
    assert precision_at_k(run, QREL, 4) == pytest.approx(0.5)
    assert recall_at_k(run, QREL, 2) == pytest.approx(0.5)
    assert recall_at_k(run, QREL, 4) == pytest.approx(1.0)
    assert average_precision(run, QREL, 4) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert mrr_at_k(run, QREL, 4) == pytest.approx(0.5)


def test_evaluate_run_skips_queries_without_relevant():
    summary, per_q = evaluate_run({"q1": ["a"], "q2": ["z"]}, {"q1": {"a": 1}, "q2": {"b": 0}})
    assert set(per_q) == {"q1"} and summary["ndcg@10"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        evaluate_run({}, {"q": {"x": 0}})


def test_bootstrap_detects_consistent_gain_and_null():
    rng = np.random.default_rng(0)
    base = rng.random(100)
    better = base + 0.2
    r = paired_bootstrap(better, base)
    assert r["mean_diff"] == pytest.approx(0.2) and r["p_value"] < 0.01 and r["ci95"][0] > 0
    null = paired_bootstrap(base, base)
    assert null["mean_diff"] == 0 and null["p_value"] > 0.5
    assert math.isnan(paired_bootstrap([1.0], [0.5])["p_value"])


def test_bm25_ranks_matching_doc_first():
    bm = BM25("en").fit(["neural machine translation", "graph neural networks", "tumor segmentation mri"])
    s = bm.scores(["translation with neural models", "completely unseen words"])
    assert s[0].argmax() == 0 and not s[1].any()
