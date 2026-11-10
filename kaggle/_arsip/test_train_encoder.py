"""ARSIP: tes untuk train_encoder.py (fine-tune encoder; bukan bagian desain final).

Berkas ini sengaja diletakkan di luar `tests/`, sehingga tidak ikut dijalankan oleh `pytest` biasa.
Kalau ingin menjalankannya lagi (dari folder hybrid-ir/):

    ../.venv/bin/python -m pytest kaggle/_arsip/test_train_encoder.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train_encoder", Path(__file__).with_name("train_encoder.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_encoder"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_kp20k_pairs_modes_and_filters(train):
    recs = [{"title": f"A long enough title {i}", "abstract": "word " * 60, "keyphrases": ["kp a", "kp b"]} for i in range(10)]
    recs.append({"title": "short", "abstract": "tiny", "keyphrases": []})
    assert len(train.kp20k_pairs(recs, "kp20k_title", 100, 1)) == 10
    assert len(train.kp20k_pairs(recs, "kp20k_mixed", 100, 1)) == 20
    kp = train.kp20k_pairs(recs, "kp20k_keyphrase", 100, 1)
    assert all(a == "kp a; kp b" and p.startswith("A long enough title") for a, p in kp)
    assert len(train.kp20k_pairs(recs, "kp20k_mixed", 5, 1)) == 5
    assert train.kp20k_pairs(recs, "kp20k_mixed", 100, 7) == train.kp20k_pairs(recs, "kp20k_mixed", 100, 7)


def test_json_reader_array_and_lines(train, tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"x": 1}, {"x": 2}]))
    (tmp_path / "b.jsonl").write_text('{"x": 1}\n\n{"x": 2}\n')
    assert list(train.iter_json_records(tmp_path / "a.json")) == list(train.iter_json_records(tmp_path / "b.jsonl"))


def test_beir_train_pairs(train, tmp_path):
    d = tmp_path / "ds"
    (d / "qrels").mkdir(parents=True)
    for f in ("corpus.jsonl", "queries.jsonl"):
        (d / f).write_text((ROOT / "data" / "demo" / f).read_text())
    (d / "qrels" / "train.tsv").write_text((ROOT / "data" / "demo" / "qrels" / "test.tsv").read_text())
    pairs = train.beir_train_pairs(d, "train", 1000, 1)
    assert len(pairs) == 24 and all(isinstance(a, str) and isinstance(p, str) for a, p in pairs)


def test_resolve_data_dir(train, tmp_path):
    direct = tmp_path / "direct"
    direct.mkdir(parents=True)
    (direct / "corpus.jsonl").write_text("{}")
    assert train.resolve_data_dir(direct) == direct
    nested = tmp_path / "input" / "datasets" / "alice" / "hybrid-ir-scifact"
    nested.mkdir(parents=True)
    (nested / "corpus.jsonl").write_text("{}")
    assert train.resolve_data_dir("/kaggle/input/hybrid-ir-scifact", search_root=tmp_path / "input") == nested
    with pytest.raises(SystemExit):
        train.resolve_data_dir("/nope", search_root=tmp_path / "kosong")
