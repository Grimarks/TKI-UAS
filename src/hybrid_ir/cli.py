"""Antarmuka baris perintah: lapisan tipis yang memanggil HybridRetriever dan evaluation.

Peran di pipeline (Gambar 1): titik masuk ketiga fase.
    build    → Fase 1 (indexing dokumen, offline)
    search   → Fase 2 (query & retrieval, online); hasilnya = kotak "Ranking top-k" di CLI
    ui       → Fase 2 lewat antarmuka Streamlit (ui_app.py)
    evaluate → Fase 3 (evaluasi), menulis results/<dataset>/

Contoh:
    hybrid-ir download scifact
    hybrid-ir build    --config configs/scifact_bart.yaml
    hybrid-ir search   --config configs/scifact_bart.yaml "query saya"
    hybrid-ir evaluate --config configs/scifact_bart.yaml --out results/scifact_bart
    hybrid-ir ui       --config configs/scifact_bart.yaml
    hybrid-ir demo     (tanpa unduhan; memakai encoder hashing dan data mini)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import MODES, AppConfig, ConfigError, load_config
from .data import DataError, load_beir_dir
from .download import DownloadError, download_beir
from .evaluation import format_markdown, run_experiment, write_outputs
from .keyphrases import MissingPrecomputed
from .retriever import HybridRetriever, IndexMismatchError, QueryError, SearchResult

log = logging.getLogger("hybrid_ir")
# Galat yang disebabkan masukan/konfigurasi pengguna: dicetak singkat tanpa traceback (kode keluar 2).
USER_ERRORS = (ConfigError, DataError, DownloadError, IndexMismatchError, QueryError, FileNotFoundError,
               MissingPrecomputed)


def _parser() -> argparse.ArgumentParser:
    """Definisi subperintah dan argumennya (nama perintah dan opsi tidak boleh diubah: dipakai README/modul)."""
    p = argparse.ArgumentParser(prog="hybrid-ir", description="Hybrid Semantic-Keyphrase Retrieval")
    p.add_argument("-v", "--verbose", action="store_true", help="tampilkan log rinci")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="unduh dataset BEIR (scifact, nfcorpus, scidocs)")
    d.add_argument("name")
    d.add_argument("--dest", default="data")

    for name, helptext in (("build", "bangun indeks"), ("evaluate", "evaluasi skenario"),
                           ("ui", "jalankan antarmuka web lokal")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--config", required=True)
        if name == "evaluate":
            s.add_argument("--out", default=None, help="folder keluaran (bawaan: results/<nama dataset>)")
            s.add_argument("--alphas", default=None, help="daftar alpha dipisah koma, mis. 0,0.25,0.5,0.75,1")
        if name == "ui":
            s.add_argument("--port", type=int, default=8501)

    s = sub.add_parser("search", help="cari dokumen")
    s.add_argument("--config", required=True)
    s.add_argument("query")
    s.add_argument("--mode", choices=list(MODES), help="bm25 hanya pembanding; hybrid = semantik + keyphrase")
    s.add_argument("--alpha", type=float, help="bobot semantik pada mode hybrid (1 - alpha untuk keyphrase)")
    s.add_argument("--top-k", type=int)
    s.add_argument("--json", action="store_true", help="keluaran JSON")

    sub.add_parser("demo", help="demo offline dengan data mini")
    return p


def _print_result(res: SearchResult) -> None:
    """Cetak hasil dalam bentuk teks. Skor BM25 hanya ditampilkan di mode bm25 (mode lain tidak menghitungnya)."""
    kps = ", ".join(f"{k} ({w:.2f})" for k, w in res.query_keyphrases) or "-"
    print(f"Query     : {res.query}\nMode      : {res.mode}\nKeyphrase : {kps}\n")
    for h in res.hits:
        print(f"{h.rank:>2}. [{h.doc_id}] {h.title}")
        bm25 = f"  bm25={h.bm25_score:.4f}" if res.mode == "bm25" else ""
        print(f"    skor={h.score:.4f}  semantik={h.semantic_score:.4f}  keyphrase={h.keyphrase_score:.4f}{bm25}")
        if h.matched_keyphrases:
            print(f"    cocok: {', '.join(h.matched_keyphrases)}")


def _cmd_download(args: argparse.Namespace) -> int:
    """`hybrid-ir download <nama>`: unduh dataset BEIR ke data/<nama>/."""
    path = download_beir(args.name, args.dest)
    print(f"Dataset siap di: {path}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    """Fase 1: muat korpus, bangun kedua indeks, simpan ke `index.dir`."""
    cfg = load_config(args.config)
    dataset = load_beir_dir(cfg.data.path, cfg.data.max_docs)
    retriever = HybridRetriever.build(cfg, dataset.documents)
    out = retriever.save()
    print(f"Indeks tersimpan di {out} ({len(dataset.documents)} dokumen)")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Fase 2: muat indeks lalu cari satu query (opsi --mode/--alpha/--top-k menimpa konfigurasi)."""
    cfg = load_config(args.config)
    retriever = HybridRetriever.load(cfg)
    res = retriever.search(args.query, mode=args.mode, alpha=args.alpha, top_k=args.top_k)
    if args.json:
        print(json.dumps({"query": res.query, "mode": res.mode, "query_keyphrases": res.query_keyphrases,
                          "hits": [h.__dict__ for h in res.hits]}, ensure_ascii=False, indent=2))
    else:
        _print_result(res)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Fase 3: jalankan empat skenario, tala alpha, tulis metrics.json/.md dan per_query_ndcg10.csv."""
    cfg = load_config(args.config)
    dataset = load_beir_dir(cfg.data.path, cfg.data.max_docs)
    retriever = HybridRetriever.load(cfg)
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else None
    result, per_query = run_experiment(retriever, dataset, cfg.data.eval_split, cfg.data.tune_split,
                                       cfg.data.dev_fraction, alphas)
    out = write_outputs(result, per_query, args.out or Path("results") / dataset.name)
    print(format_markdown(result))
    print(f"Hasil lengkap disimpan di {out}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    """Fase 2 lewat UI: jalankan Streamlit hanya di 127.0.0.1, konfigurasi diteruskan lewat env HYBRID_IR_CONFIG."""
    load_config(args.config)  # validasi dini
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Galat: streamlit belum terpasang. Jalankan: pip install -e \".[ui]\"", file=sys.stderr)
        return 2
    app = Path(__file__).with_name("ui_app.py")
    env = dict(os.environ, HYBRID_IR_CONFIG=str(Path(args.config).resolve()))
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.address", "127.0.0.1",            # hanya bisa diakses dari komputer ini
           "--server.port", str(args.port),
           "--server.headless", "true",
           "--server.fileWatcherType", "none",          # watcher memindai modul transformers -> galat palsu
           "--browser.gatherUsageStats", "false"]
    print(f"Membuka UI di http://127.0.0.1:{args.port} (Ctrl+C untuk berhenti)")
    return subprocess.call(cmd, env=env)


def _cmd_demo(_args: argparse.Namespace) -> int:
    """Demo offline: data mini + encoder hashing (bukan model semantik), tanpa unduhan."""
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "configs" / "demo.yaml"
    cfg: AppConfig = load_config(cfg_path if cfg_path.is_file() else "configs/demo.yaml")
    dataset = load_beir_dir(cfg.data.path)
    retriever = HybridRetriever.build(cfg, dataset.documents)
    retriever.save()
    # keduanya query dataset demo: keyphrase-nya ada di data/demo/keyphrases_demo.jsonl (demo tanpa fallback)
    for q in ("graph neural network node embeddings", "keyphrases to improve search indexing"):
        _print_result(retriever.search(q, top_k=3))
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Titik masuk `hybrid-ir`. Galat pengguna dicetak rapi dengan kode keluar 2 (tanpa traceback)."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if not args.verbose:
        # log tiap permintaan HTTP ke Hugging Face hanya mengganggu keluaran (mis. saat demo); tampil dengan -v
        for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    handlers = {"download": _cmd_download, "build": _cmd_build, "search": _cmd_search,
                "evaluate": _cmd_evaluate, "ui": _cmd_ui, "demo": _cmd_demo}
    try:
        return handlers[args.command](args)
    except USER_ERRORS as exc:
        # KeyError (induk MissingPrecomputed) membungkus pesannya dengan tanda kutip; ambil pesan aslinya
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"Galat: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
