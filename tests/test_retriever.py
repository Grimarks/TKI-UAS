
import pytest

from hybrid_ir.evaluation import evaluate_run, run_experiment
from hybrid_ir.keyphrases import MissingPrecomputed
from hybrid_ir.retriever import HybridRetriever, IndexMismatchError, QueryError


@pytest.fixture()
def retriever(demo_cfg, demo_dataset):
    return HybridRetriever.build(demo_cfg, demo_dataset.documents)


def test_search_finds_expected_document(retriever):
    res = retriever.search("keyphrases to improve search indexing", top_k=3)
    assert res.hits[0].doc_id == "d16"
    assert res.query_keyphrases and res.hits[0].matched_keyphrases


@pytest.mark.parametrize("mode", ["semantic", "keyphrase", "bm25", "hybrid"])
def test_modes_return_ranked_hits(retriever, mode):
    res = retriever.search("graph neural network node embeddings", mode=mode, top_k=5)
    assert 1 <= len(res.hits) <= 5
    assert [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)
    if mode == "keyphrase":
        assert all(h.keyphrase_score > 0 for h in res.hits)
    if mode == "bm25":
        assert all(h.bm25_score > 0 for h in res.hits)
    else:
        assert all(h.bm25_score == 0 for h in res.hits)   # BM25 hanya dihitung di mode bm25


def test_alpha_extremes_match_single_modes(retriever):
    q = "graph neural network node embeddings"
    ids = lambda res: [h.doc_id for h in res.hits]
    assert ids(retriever.search(q, mode="hybrid", alpha=1.0, top_k=3)) == ids(retriever.search(q, mode="semantic", top_k=3))
    assert ids(retriever.search(q, mode="hybrid", alpha=0.0, top_k=3)) == ids(retriever.search(q, mode="keyphrase", top_k=3))


def test_free_query_without_fallback_is_rejected(retriever):
    with pytest.raises(MissingPrecomputed):
        retriever.search("a query that is not in the demo keyphrase file")


def test_search_many_equals_single(retriever):
    qs = ["segment tumors in medical scans", "privacy in federated training"]
    many = retriever.search_many(qs, top_k=3)
    for q, r in zip(qs, many):
        assert [h.doc_id for h in r.hits] == [h.doc_id for h in retriever.search(q, top_k=3).hits]


def test_save_load_roundtrip(retriever, demo_cfg):
    path = retriever.save()
    assert {p.name for p in path.iterdir()} >= {"dense.npy", "kp_matrix.npz", "meta.json", "docs.jsonl"}
    assert not any(p.suffix in (".pkl", ".pickle") for p in path.iterdir())
    loaded = HybridRetriever.load(demo_cfg)
    q = "dense vector search for retrieval"
    assert [h.doc_id for h in loaded.search(q).hits] == [h.doc_id for h in retriever.search(q).hits]


def test_load_missing_index(demo_cfg):
    with pytest.raises(FileNotFoundError):
        HybridRetriever.load(demo_cfg)


def test_load_detects_encoder_mismatch(retriever, demo_cfg):
    retriever.save()
    demo_cfg.encoder.hashing_dim = 256
    with pytest.raises(IndexMismatchError, match="encoder berbeda"):
        HybridRetriever.load(demo_cfg)


def test_load_detects_keyphrase_setting_mismatch(retriever, demo_cfg):
    retriever.save()
    demo_cfg.keyphrase.unit = "phrase"
    with pytest.raises(IndexMismatchError, match="indeks keyphrase"):
        HybridRetriever.load(demo_cfg)


@pytest.mark.parametrize("bad", ["", "   ", "x" * 5000])
def test_query_validation(retriever, bad):
    with pytest.raises(QueryError):
        retriever.search(bad)


def test_parameter_validation(retriever):
    with pytest.raises(QueryError):
        retriever.search("graph", top_k=0)
    with pytest.raises(QueryError):
        retriever.search("graph", alpha=2.0)
    with pytest.raises(QueryError):
        retriever.search("graph", mode="magic")
    with pytest.raises(QueryError):
        retriever.search("graph", mode="hybrid_bm25")
    with pytest.raises(TypeError):
        retriever.search("graph", fusion="rrf")


def test_save_refuses_to_overwrite_foreign_directory(retriever, tmp_path):
    victim = tmp_path / "important"
    victim.mkdir()
    (victim / "thesis.docx").write_text("jangan hapus")
    with pytest.raises(IndexMismatchError, match="menolak menimpa"):
        retriever.save(victim)
    assert (victim / "thesis.docx").read_text() == "jangan hapus"


def test_save_can_overwrite_previous_index(retriever):
    retriever.save()
    retriever.save()


def test_run_experiment_on_demo(retriever, demo_dataset, demo_cfg):
    result, per_query = run_experiment(retriever, demo_dataset, "test", None, 0.3)
    assert result["n_tune_queries"] + result["n_eval_queries"] == 8 and result["n_eval_queries"] == 6
    assert result["alpha_selected"] in [r["alpha"] for r in result["alpha_sweep_tune"]]
    assert list(result["scenarios"]) == ["bm25", "semantic", "keyphrase", "hybrid_weighted"]
    assert result["scenarios"]["hybrid_weighted"]["params"] == {"alpha": result["alpha_selected"]}
    assert "weights_selected" not in result and "weights_sweep_tune" not in result
    for sc in result["scenarios"].values():
        assert all(0.0 <= v <= 1.0 for v in sc["metrics"].values())
    assert [(s["a"], s["b"]) for s in result["significance"]] == [
        ("hybrid_weighted", "semantic"), ("hybrid_weighted", "keyphrase"), ("hybrid_weighted", "bm25")]
    assert set(per_query) == set(result["scenarios"])
    assert len(result["alpha_sweep_eval_posthoc"]) == 11


@pytest.mark.parametrize("scenario,mode", [("bm25", "bm25"), ("keyphrase", "keyphrase"), ("semantic", "semantic")])
def test_evaluation_ranks_like_search(retriever, demo_dataset, scenario, mode):
    """Skenario evaluasi harus memperingkat persis seperti mode `search` yang sama (top 100).

    Untuk bm25/keyphrase artinya dokumen berskor 0 dibuang. Di korpus demo (24 dokumen) setiap query punya
    dokumen tanpa kata yang cocok, jadi tes ini gagal bila evaluasi kembali mengisi ekor dengan dokumen berskor 0.
    """
    result, per_query = run_experiment(retriever, demo_dataset, "test", None, 0.3)
    eval_qids = sorted(per_query[scenario])
    texts = [demo_dataset.queries[q] for q in eval_qids]
    assert any((retriever.bm25_scores([t])[0] == 0).any() for t in texts)
    run = {q: [h.doc_id for h in res.hits]
           for q, res in zip(eval_qids, retriever.search_many(texts, mode=mode, top_k=100))}
    expected, _ = evaluate_run(run, {q: demo_dataset.qrels_for("test")[q] for q in eval_qids})
    assert result["scenarios"][scenario]["metrics"] == expected
    if mode == "bm25":
        assert all(len(run[q]) == int((retriever.bm25_scores([t])[0] > 0).sum()) for q, t in zip(eval_qids, texts))
