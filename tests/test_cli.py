import json

import pytest
import yaml

from hybrid_ir.cli import main


@pytest.fixture()
def cfg_file(tmp_path, root):
    raw = yaml.safe_load((root / "configs" / "demo.yaml").read_text())
    raw["data"]["path"] = str(root / "data" / "demo")
    raw["keyphrase"]["precomputed_files"] = [str(root / "data" / "demo" / "keyphrases_demo.jsonl")]
    raw["index"]["dir"] = str(tmp_path / "idx")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


Q = "graph neural network node embeddings"   # query dataset demo (keyphrase-nya ada di berkas precomputed)


def _json_out(capsys) -> dict:
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])


def test_build_search_evaluate(cfg_file, tmp_path, capsys):
    assert main(["build", "--config", cfg_file]) == 0
    assert main(["search", "--config", cfg_file, Q, "--top-k", "2", "--json"]) == 0
    payload = _json_out(capsys)
    assert len(payload["hits"]) == 2 and payload["mode"] == "hybrid"
    assert main(["search", "--config", cfg_file, Q, "--mode", "bm25", "--top-k", "2", "--json"]) == 0
    payload = _json_out(capsys)
    assert payload["mode"] == "bm25" and payload["hits"][0]["bm25_score"] > 0
    assert main(["evaluate", "--config", cfg_file, "--out", str(tmp_path / "res")]) == 0
    assert (tmp_path / "res" / "metrics.json").is_file() and (tmp_path / "res" / "metrics.md").is_file()
    assert (tmp_path / "res" / "per_query_ndcg10.csv").is_file()
    md = (tmp_path / "res" / "metrics.md").read_text()
    assert "BM25 (pembanding)" in md and "RRF" not in md


def test_demo_build_and_search_offline(cfg_file, capsys, no_network):
    """configs/demo.yaml harus jalan tanpa jaringan: encoder hashing + keyphrase precomputed tanpa fallback."""
    assert main(["build", "--config", cfg_file]) == 0
    for mode in ("hybrid", "semantic", "keyphrase", "bm25"):
        assert main(["search", "--config", cfg_file, Q, "--mode", mode, "--top-k", "3", "--json"]) == 0
        assert _json_out(capsys)["hits"]


@pytest.mark.parametrize("extra", [["--mode", "hybrid_bm25"], ["--fusion", "rrf"], ["--weights", "0.5,0,0.5"]])
def test_removed_search_options_rejected(cfg_file, extra):
    with pytest.raises(SystemExit) as exc:
        main(["search", "--config", cfg_file, Q, *extra])
    assert exc.value.code == 2


def test_search_without_index_is_clean_error(cfg_file, capsys):
    assert main(["search", "--config", cfg_file, "x"]) == 2
    assert "Galat" in capsys.readouterr().err


def test_bad_config_is_clean_error(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("retrieval:\n  alpha: 3\n")
    assert main(["build", "--config", str(p)]) == 2
    assert "alpha" in capsys.readouterr().err


def test_download_rejects_unknown_dataset(tmp_path, capsys):
    assert main(["download", "../../etc/passwd", "--dest", str(tmp_path)]) == 2
