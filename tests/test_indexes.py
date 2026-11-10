import numpy as np
import pytest

from hybrid_ir.indexes import DenseIndex, KeyphraseIndex

DOCS = [
    [("neural network", 1.0), ("machine translation", 0.8)],
    [("graph neural network", 1.0), ("node classification", 0.7)],
    [("keyphrase extraction", 1.0)],
    [],
]


def _idx(similarity="cosine", unit="both"):
    return KeyphraseIndex.build(DOCS, "en", unit, similarity)


def test_cosine_prefers_overlapping_doc():
    s = _idx().scores([[("neural networks", 1.0)]])[0]
    assert s[0] > 0 and s[1] > 0 and s[2] == 0 and s[3] == 0
    assert s.argmax() in (0, 1)


def test_exact_phrase_scores_highest():
    s = _idx().scores([[("keyphrase extraction", 1.0)]])[0]
    assert s.argmax() == 2 and s[2] == pytest.approx(1.0, abs=1e-5)


def test_stemming_matches_plurals():
    s = _idx(unit="phrase").scores([[("keyphrases extraction", 1.0)]])[0]
    assert s[2] > 0


def test_unknown_query_scores_zero():
    assert not _idx().scores([[("quantum chromodynamics", 1.0)]]).any()


def test_jaccard_bounds_and_identity():
    s = _idx("jaccard", "phrase").scores([[("keyphrase extraction", 1.0)]])[0]
    assert s[2] == pytest.approx(1.0) and ((0 <= s) & (s <= 1)).all()


def test_jaccard_counts_unknown_units_in_union():
    full = _idx("jaccard", "phrase").scores([[("keyphrase extraction", 1.0)]])[0][2]
    diluted = _idx("jaccard", "phrase").scores([[("keyphrase extraction", 1.0), ("unseen thing", 1.0)]])[0][2]
    assert diluted == pytest.approx(0.5) and full == pytest.approx(1.0)


def test_roundtrip_without_pickle(tmp_path):
    idx = _idx()
    idx.save(tmp_path)
    loaded = KeyphraseIndex.load(tmp_path, "en", "both", "cosine")
    q = [[("graph neural network", 1.0)]]
    assert np.allclose(idx.scores(q), loaded.scores(q))
    assert not any(p.suffix == ".pkl" for p in tmp_path.iterdir())


def test_dense_scores_and_roundtrip(tmp_path):
    m = np.eye(3, dtype=np.float32)
    d = DenseIndex(["a", "b", "c"], m)
    assert np.allclose(d.scores(np.array([[0, 1, 0]], dtype=np.float32)), [[0, 1, 0]])
    d.save(tmp_path)
    assert DenseIndex.load(tmp_path).ids == ["a", "b", "c"]
    with pytest.raises(ValueError):
        d.scores(np.ones((1, 5), dtype=np.float32))
