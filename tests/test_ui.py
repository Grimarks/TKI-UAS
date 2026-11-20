import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from hybrid_ir.retriever import HybridRetriever  # noqa: E402
from hybrid_ir.ui_app import chips, esc  # noqa: E402


def test_escape_helpers():
    assert esc("**bold** [x](http://evil) <img src=x>") == r"\*\*bold\*\* \[x\]\(http://evil\) \<img src=x\>"
    assert chips(["a`b", "c"]) == "`ab`, `c`"   # backtick di data dibuang agar tidak memecah format
    assert chips([]) == "-"


@pytest.fixture()
def app_env(demo_cfg, demo_dataset, monkeypatch, tmp_path, root):
    import yaml
    HybridRetriever.build(demo_cfg, demo_dataset.documents).save()
    cfg_path = tmp_path / "ui.yaml"
    raw = yaml.safe_load((root / "configs" / "demo.yaml").read_text())
    raw["index"]["dir"] = demo_cfg.index.dir
    raw["data"]["path"] = demo_cfg.data.path
    raw["keyphrase"]["precomputed_files"] = demo_cfg.keyphrase.precomputed_files
    cfg_path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("HYBRID_IR_CONFIG", str(cfg_path))
    return root / "src" / "hybrid_ir" / "ui_app.py"


def test_ui_renders_and_searches(app_env):
    at = AppTest.from_file(str(app_env), default_timeout=60).run()
    assert not at.exception
    assert len(at.sidebar.selectbox) == 0      # satu konfigurasi: tanpa pilihan dataset
    at.text_input[0].set_value("keyphrases to improve search indexing")
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Keyphrase query" in text and "Keyphrases improve document indexing" in text.replace("\\", "")


def test_config_paths_and_dataset_title(monkeypatch):
    from hybrid_ir.ui_app import config_paths, dataset_title
    monkeypatch.setenv("HYBRID_IR_CONFIG", " configs/scifact_bart.yaml , configs/nfcorpus_bart.yaml,")
    assert config_paths() == ["configs/scifact_bart.yaml", "configs/nfcorpus_bart.yaml"]
    assert dataset_title("configs/nfcorpus_bart.yaml") == "NFCorpus"
    assert dataset_title("/tmp/lain.yaml") == "lain"


def test_ui_two_datasets_shows_selector(app_env, monkeypatch, tmp_path):
    import os
    import shutil
    first = os.environ["HYBRID_IR_CONFIG"]
    second = tmp_path / "ui_kedua.yaml"
    shutil.copy(first, second)
    monkeypatch.setenv("HYBRID_IR_CONFIG", f"{first},{second}")
    at = AppTest.from_file(str(app_env), default_timeout=60).run()
    assert not at.exception
    pilih = at.sidebar.selectbox[0]
    assert pilih.label == "Dataset" and len(pilih.options) == 2
    pilih.select(str(second)).run()
    assert not at.exception
    at.text_input[0].set_value("keyphrases to improve search indexing")
    at.button[0].click().run()
    assert not at.exception
    assert "Keyphrases improve document indexing" in " ".join(m.value for m in at.markdown).replace("\\", "")


def test_ui_without_env_shows_error(monkeypatch, root):
    monkeypatch.delenv("HYBRID_IR_CONFIG", raising=False)
    at = AppTest.from_file(str(root / "src" / "hybrid_ir" / "ui_app.py"), default_timeout=30).run()
    assert at.error
