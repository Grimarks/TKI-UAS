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
    at.text_input[0].set_value("keyphrases to improve search indexing")
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Keyphrase query" in text and "Keyphrases improve document indexing" in text.replace("\\", "")


def test_ui_without_env_shows_error(monkeypatch, root):
    monkeypatch.delenv("HYBRID_IR_CONFIG", raising=False)
    at = AppTest.from_file(str(root / "src" / "hybrid_ir" / "ui_app.py"), default_timeout=30).run()
    assert at.error
