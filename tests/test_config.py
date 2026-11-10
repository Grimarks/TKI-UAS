import pytest

from hybrid_ir.config import ConfigError, config_from_dict, load_config


def test_defaults_valid():
    cfg = config_from_dict({})
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.keyphrase.method == "seq2seq" and cfg.keyphrase.query_fallback == "seq2seq"


def test_unknown_section_rejected():
    with pytest.raises(ConfigError, match="bagian tidak dikenal"):
        config_from_dict({"retrival": {}})


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="kunci tidak dikenal"):
        config_from_dict({"retrieval": {"alfa": 0.5}})


@pytest.mark.parametrize("section,key,val", [
    ("retrieval", "alpha", 1.5), ("retrieval", "top_k", 0), ("encoder", "batch_size", 0),
    ("retrieval", "mode", "magic"), ("keyphrase", "query_method", "magic"), ("keyphrase", "query_fallback", "magic"),
    ("keyphrase", "unit", "word"), ("keyphrase", "method", "magic"), ("data", "dev_fraction", 1.0),
])
def test_out_of_range_rejected(section, key, val):
    with pytest.raises(ConfigError):
        config_from_dict({section: {key: val}})


def test_precomputed_requires_files():
    with pytest.raises(ConfigError, match="precomputed_files"):
        config_from_dict({"keyphrase": {"method": "precomputed"}})


def test_yaml_is_loaded_safely(tmp_path):
    p = tmp_path / "evil.yaml"
    p.write_text("data: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(Exception):  # safe_load menolak tag python/object
        load_config(p)


def test_missing_file():
    with pytest.raises(ConfigError):
        load_config("tidak_ada.yaml")


@pytest.mark.parametrize("section,key,val,msg", [
    ("keyphrase", "method", "yake", "YAKE sudah tidak didukung"),
    ("keyphrase", "query_method", "yake", "YAKE sudah tidak didukung"),
    ("keyphrase", "query_fallback", "yake", "YAKE sudah tidak didukung"),
    ("keyphrase", "max_ngram", 3, "YAKE sudah tidak didukung"),
    ("retrieval", "fusion", "rrf", "fusi RRF tidak dipakai lagi"),
    ("retrieval", "fusion", "weighted", "fusi RRF tidak dipakai lagi"),
    ("retrieval", "rrf_k", 60, "fusi RRF tidak dipakai lagi"),
    ("retrieval", "mode", "hybrid_bm25", "hybrid_bm25 sudah dihapus"),
    ("retrieval", "bm25_weights", [0.5, 0.0, 0.5], "bm25_weights sudah dihapus"),
])
def test_legacy_settings_rejected_with_clear_message(section, key, val, msg):
    with pytest.raises(ConfigError, match=msg):
        config_from_dict({section: {key: val}})


def test_shipped_configs_valid(root):
    for name in ("demo", "scifact_bart", "nfcorpus_bart"):
        load_config(root / "configs" / f"{name}.yaml")


@pytest.mark.parametrize("name", ["scifact", "nfcorpus", "scifact_ft"])
def test_archived_configs_rejected(root, name):
    with pytest.raises(ConfigError, match="YAKE sudah tidak didukung|fusi RRF"):
        load_config(root / "configs" / "_arsip" / f"{name}.yaml")
