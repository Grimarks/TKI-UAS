import json

import pytest

from hybrid_ir.config import KeyphraseConfig
from hybrid_ir import keyphrases
from hybrid_ir.keyphrases import (FallbackExtractor, MissingPrecomputed, PrecomputedKeyphrases, TextItem,
                                  build_extractor, parse_generated_keyphrases, rank_weights)
from hybrid_ir.text import text_fingerprint


def test_rank_weights_monotonic():
    w = rank_weights(5)
    assert w[0] == 1.0 and all(a > b for a, b in zip(w, w[1:])) and all(0 < x <= 1 for x in w)


def test_parse_generated_basic_and_dedupe():
    out = parse_generated_keyphrases("neural network; Neural Network ;; machine translation;a;b", 10)
    assert [k for k, _ in out] == ["neural network", "machine translation", "a", "b"]
    assert out[0][1] == 1.0


def test_parse_generated_limits():
    assert len(parse_generated_keyphrases("a;b;c;d", 2)) == 2
    long = " ".join(["w"] * 20)
    assert parse_generated_keyphrases(f"{long};ok", 5) == [("ok", 1.0)]
    assert parse_generated_keyphrases("", 5) == []


def test_yake_extractor_removed():
    assert not hasattr(keyphrases, "YakeExtractor")


def _precomputed_cfg(tmp_path, fallback):
    item = TextItem("Title", "Body")
    p = tmp_path / "kp.jsonl"
    p.write_text(json.dumps({"key": text_fingerprint(item.full), "keyphrases": [["alpha", 1.0]]}) + "\n")
    return item, KeyphraseConfig(method="precomputed", query_fallback=fallback, precomputed_files=[str(p)])


def test_query_fallback_seq2seq_used_for_unknown_text(tmp_path, fake_seq2seq):
    item, cfg = _precomputed_cfg(tmp_path, "seq2seq")
    ex = build_extractor(cfg, "query")
    assert isinstance(ex, FallbackExtractor)
    assert ex.extract([item, TextItem("free query text")], 5) == [[("alpha", 1.0)], [("free query", 1.0)]]


def test_query_fallback_none_is_strict(tmp_path):
    item, cfg = _precomputed_cfg(tmp_path, "none")
    ex = build_extractor(cfg, "query")
    assert isinstance(ex, PrecomputedKeyphrases)
    with pytest.raises(MissingPrecomputed, match="seq2seq"):
        ex.extract([TextItem("free query text")], 5)


def test_precomputed_lookup_and_missing(tmp_path):
    item = TextItem("Title", "Body")
    p = tmp_path / "kp.jsonl"
    p.write_text(json.dumps({"key": text_fingerprint(item.full), "keyphrases": [["alpha", 1.0], ["beta", 0.8]]}) + "\n")
    pre = PrecomputedKeyphrases([p])
    assert pre.extract([item], 1) == [[("alpha", 1.0)]]
    with pytest.raises(MissingPrecomputed, match="1 dari 2"):
        pre.extract([item, TextItem("Other", "text")], 5)


@pytest.mark.parametrize("line", ['{"key": "short", "keyphrases": []}', '{"key": "%s", "keyphrases": [["x", -1]]}' % ("a" * 40), "not json"])
def test_precomputed_rejects_bad_rows(tmp_path, line):
    p = tmp_path / "bad.jsonl"
    p.write_text(line + "\n")
    with pytest.raises(ValueError):
        PrecomputedKeyphrases([p])
