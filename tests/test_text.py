from hybrid_ir.text import (content_tokens, make_full_text, normalize_phrase, normalize_text,
                            phrase_units, text_fingerprint)


def test_normalize_text_collapses_whitespace_and_controls():
    assert normalize_text("  a\t\tb\n\nc\x00d  ") == "a b c d"
    assert normalize_text(None) == ""


def test_normalize_text_is_nfkc():
    assert normalize_text("ﬁnal") == "final"  # ligatur fi -> fi


def test_make_full_text():
    assert make_full_text("Title", "Body text") == "Title. Body text"
    assert make_full_text("Title?", "Body") == "Title? Body"
    assert make_full_text("", "Body") == "Body"
    assert make_full_text("Title", "") == "Title"


def test_fingerprint_stable_and_whitespace_insensitive():
    assert text_fingerprint("a  b") == text_fingerprint(" a b ")
    assert len(text_fingerprint("x")) == 40


def test_content_tokens_drop_stopwords_and_stem():
    assert content_tokens("The neural networks are learning", "en") == ["neural", "network", "learn"]


def test_normalize_phrase_strips_edge_stopwords_and_stems():
    assert normalize_phrase("The Neural Networks", "en") == "neural network"
    assert normalize_phrase("the of", "en") == ""


def test_phrase_units_modes():
    assert phrase_units("neural networks", "en", "phrase") == {"p:neural network"}
    assert phrase_units("neural networks", "en", "token") == {"t:neural", "t:network"}
    assert phrase_units("neural networks", "en", "both") == {"p:neural network", "t:neural", "t:network"}


def test_indonesian_stemmer_available():
    assert content_tokens("pencarian dokumen akademik", "id")
