import json

import pytest

from hybrid_ir.data import DataError, load_beir_dir, split_query_ids, usable_queries


def test_load_demo(demo_dataset):
    assert len(demo_dataset.documents) == 24
    assert len(demo_dataset.queries) == 8
    assert "test" in demo_dataset.qrels
    assert demo_dataset.documents[0].full_text.startswith("Attention-based neural machine translation.")


def test_usable_queries(demo_dataset):
    assert set(usable_queries(demo_dataset, "test")) == set(demo_dataset.queries)


def test_missing_split(demo_dataset):
    with pytest.raises(DataError, match="split"):
        demo_dataset.qrels_for("train")


def test_split_deterministic_and_disjoint():
    ids = [f"q{i}" for i in range(20)]
    dev1, ev1 = split_query_ids(ids, 0.3)
    dev2, ev2 = split_query_ids(ids, 0.3)
    assert dev1 == dev2 and ev1 == ev2
    assert not set(dev1) & set(ev1) and len(dev1) + len(ev1) == 20 and len(dev1) == 6


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"_id": "a", "text": "x"}) + "\n" + json.dumps({"_id": "a", "text": "y"}) + "\n")
    (tmp_path / "queries.jsonl").write_text(json.dumps({"_id": "q", "text": "x"}) + "\n")
    with pytest.raises(DataError, match="ganda"):
        load_beir_dir(tmp_path)


def test_bad_json_line(tmp_path):
    (tmp_path / "corpus.jsonl").write_text("{not json}\n")
    (tmp_path / "queries.jsonl").write_text("")
    with pytest.raises(DataError, match="JSON"):
        load_beir_dir(tmp_path)


def test_missing_folder():
    with pytest.raises(DataError, match="tidak ditemukan"):
        load_beir_dir("folder_yang_tidak_ada")
