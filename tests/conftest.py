import socket
from pathlib import Path

import pytest

from hybrid_ir import keyphrases
from hybrid_ir.config import load_config
from hybrid_ir.data import load_beir_dir

ROOT = Path(__file__).resolve().parents[1]
DEMO_KEYPHRASES = ROOT / "data" / "demo" / "keyphrases_demo.jsonl"


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture()
def demo_cfg(tmp_path):
    cfg = load_config(ROOT / "configs" / "demo.yaml")
    cfg.data.path = str(ROOT / "data" / "demo")
    cfg.keyphrase.precomputed_files = [str(DEMO_KEYPHRASES)]
    cfg.index.dir = str(tmp_path / "idx")
    return cfg


@pytest.fixture(scope="session")
def demo_dataset():
    return load_beir_dir(ROOT / "data" / "demo")


class FakeSeq2Seq:
    """Pengganti generator BART untuk tes (tanpa unduhan model): dua kata pertama teks sebagai keyphrase."""

    name = "seq2seq"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def extract(self, items, top_n):
        return [[(" ".join(item.full.lower().split()[:2]), 1.0)][:top_n] if item.full else [] for item in items]

    def fingerprint(self) -> dict:
        return {"method": "seq2seq", "model": "fake"}


@pytest.fixture()
def fake_seq2seq(monkeypatch):
    monkeypatch.setattr(keyphrases, "Seq2SeqKeyphraseGenerator", FakeSeq2Seq)
    return FakeSeq2Seq


@pytest.fixture()
def no_network(monkeypatch):
    """Gagalkan setiap upaya koneksi jaringan selama tes."""
    def refuse(*_args, **_kwargs):
        raise OSError("akses jaringan diblokir dalam tes")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
