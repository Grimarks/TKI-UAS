#!/usr/bin/env python3
"""Hasilkan keyphrase untuk korpus (+ query) format BEIR memakai model seq2seq (mis. BART-KPG).

Dirancang untuk Kaggle "GPU T4 x2": skrip induk menjalankan satu proses per GPU (data-parallel),
tiap proses mengerjakan satu shard data, lalu hasilnya digabung menjadi satu berkas JSONL yang
dibaca `hybrid-ir` lewat `keyphrase.method: precomputed`.

Format keluaran (satu baris per teks unik):
    {"key": "<sha1 teks ternormalisasi>", "keyphrases": [["frasa", bobot], ...]}

Skrip ini SENGAJA mandiri (tidak meng-import hybrid_ir) supaya bisa dijalankan sebagai satu berkas di
Kaggle. Fungsi di bagian "KONTRAK" harus identik dengan hybrid_ir; tests/test_kaggle_scripts.py
menguji kesamaannya. Skrip ini tidak membaca kredensial Kaggle apa pun.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import hashlib
from pathlib import Path

# ====================================================================================
# KONTRAK (harus sama dengan src/hybrid_ir/text.py dan keyphrases.py)
# ====================================================================================
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def make_full_text(title, body):
    title = normalize_text(title)
    body = normalize_text(body)
    if title and body:
        sep = "" if title.endswith((".", "?", "!")) else "."
        return f"{title}{sep} {body}"
    return title or body


def text_fingerprint(text):
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def rank_weights(n):
    return [1.0 / (1.0 + 0.25 * i) for i in range(n)]


def parse_generated_keyphrases(text, max_phrases, sep=";", max_words=8, max_chars=80):
    seen, phrases = set(), []
    for part in text.split(sep):
        phrase = normalize_text(part)
        if not phrase or len(phrase) > max_chars or len(phrase.split()) > max_words:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return list(zip(phrases, rank_weights(len(phrases))))


def source_text(title, body):
    """Format masukan model taln-ls2n/bart-base-kp20k: 'judul<s>abstrak'."""
    return f"{title}<s>{body}"


# ====================================================================================
# Pembacaan data
# ====================================================================================
def resolve_data_dir(path, search_root="/kaggle/input"):
    """Kaggle memasang dataset di lokasi yang bisa berbeda antar-versi; cari folder yang berisi corpus.jsonl."""
    p = Path(path)
    if (p / "corpus.jsonl").is_file():
        return p
    root = Path(search_root)
    if root.is_dir():
        hits = sorted(c.parent for c in root.rglob("corpus.jsonl") if c.parent.name == p.name or c.parent.parent.name == p.name)
        if len(hits) == 1:
            return hits[0]
        if not hits:
            hits = sorted(c.parent for c in root.rglob("corpus.jsonl"))
            if len(hits) == 1:
                return hits[0]
    seen = [str(x) for x in sorted(root.glob("*"))][:20] if root.is_dir() else []
    raise SystemExit(f"folder data tidak ditemukan: {path}. Isi {search_root}: {seen}")

def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_items(data_dir, include_queries=True, max_items=None):
    """Kembalikan daftar item unik: {'key', 'title', 'body'} terurut menurut key (deterministik)."""
    root = resolve_data_dir(data_dir)
    items = {}
    for obj in _iter_jsonl(root / "corpus.jsonl"):
        title, body = normalize_text(str(obj.get("title") or "")), normalize_text(str(obj.get("text") or ""))
        if title or body:
            items.setdefault(text_fingerprint(make_full_text(title, body)), {"title": title, "body": body})
    if include_queries and (root / "queries.jsonl").is_file():
        for obj in _iter_jsonl(root / "queries.jsonl"):
            q = normalize_text(str(obj.get("text") or ""))
            if q:
                items.setdefault(text_fingerprint(make_full_text(q, "")), {"title": q, "body": ""})
    ordered = [{"key": k, **v} for k, v in sorted(items.items())]
    return ordered[:max_items] if max_items else ordered


# ====================================================================================
# Worker (satu proses = satu GPU)
# ====================================================================================
def run_worker(items, generate_fn, part_path, batch_size, top_n, log=print):
    """Proses `items` per batch dengan `generate_fn(list[str]) -> list[str]`; tulis ke `part_path`.

    Dapat dilanjutkan (resume): kunci yang sudah ada di part_path dilewati. Jika terjadi
    kehabisan memori GPU, ukuran batch dipangkas dua dan batch diulang.
    """
    part_path = Path(part_path)
    done = set()
    if part_path.exists():
        for obj in _iter_jsonl(part_path):
            done.add(obj["key"])
    todo = [it for it in items if it["key"] not in done]
    todo.sort(key=lambda it: -(len(it["title"]) + len(it["body"])))   # panjang dulu: padding minimal & OOM terdeteksi awal
    log(f"[worker] total={len(items)} sudah={len(done)} dikerjakan={len(todo)}")
    t0, n_done, i, bs = time.time(), 0, 0, batch_size
    with open(part_path, "a", encoding="utf-8") as out:
        while i < len(todo):
            batch = todo[i:i + bs]
            try:
                texts = generate_fn([source_text(it["title"], it["body"]) for it in batch])
            except RuntimeError as exc:   # torch.cuda.OutOfMemoryError adalah subclass RuntimeError
                if "out of memory" in str(exc).lower() and bs > 1:
                    bs = max(1, bs // 2)
                    log(f"[worker] OOM -> batch diperkecil menjadi {bs}")
                    continue
                raise
            for it, text in zip(batch, texts):
                kps = parse_generated_keyphrases(text, top_n)
                out.write(json.dumps({"key": it["key"], "keyphrases": [[k, round(w, 6)] for k, w in kps]},
                                     ensure_ascii=False) + "\n")
            out.flush()
            i += len(batch)
            n_done += len(batch)
            if (n_done // max(1, len(batch))) % 20 == 0 or i >= len(todo):
                rate = n_done / max(time.time() - t0, 1e-9)
                log(f"[worker] {i}/{len(todo)}  {rate:.1f} teks/detik")
    return {"processed": n_done, "seconds": round(time.time() - t0, 1)}


def load_generator(args):
    """Muat model ke GPU yang terlihat proses ini dan kembalikan (generate_fn, info)."""
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    kwargs = {"trust_remote_code": False}
    if args.revision:
        kwargs["revision"] = args.revision
    tok = AutoTokenizer.from_pretrained(args.model, **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, use_safetensors=True, **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    if device == "cuda":
        model.half()

    def generate(sources):
        enc = tok(sources, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_source_tokens).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                                 early_stopping=True)
        return tok.batch_decode(out, skip_special_tokens=True)

    info = {"device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "commit": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__}
    return generate, info


def worker_main(args):
    items = read_items(args.data_dir, not args.no_queries, args.max_items)
    mine = items[args.shard::args.num_shards]
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    generate, info = load_generator(args)
    print(f"[worker {args.shard}] {info}", flush=True)
    stats = run_worker(mine, generate, work / f"part-{args.shard}.jsonl", args.batch_size, args.top_n,
                       log=lambda m: print(f"[shard {args.shard}] {m}", flush=True))
    (work / f"part-{args.shard}.meta.json").write_text(json.dumps({**info, **stats, "items": len(mine)}), encoding="utf-8")
    return 0


# ====================================================================================
# Induk: meluncurkan worker per GPU dan menggabungkan hasil
# ====================================================================================
def merge_parts(work_dir, num_shards, out_path, expected_keys):
    rows = {}
    for s in range(num_shards):
        p = Path(work_dir) / f"part-{s}.jsonl"
        if p.exists():
            for obj in _iter_jsonl(p):
                rows[obj["key"]] = obj
    missing = set(expected_keys) - set(rows)
    if missing:
        raise RuntimeError(f"{len(missing)} teks belum punya keyphrase; jalankan ulang (proses akan melanjutkan)")
    tmp = Path(str(out_path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for key in sorted(rows):
            fh.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    return len(rows)


def count_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def parent_main(args):
    n_gpu = count_gpus()
    if args.require_gpus and n_gpu != args.require_gpus:
        print(f"GALAT: diminta {args.require_gpus} GPU, terdeteksi {n_gpu}. "
              "Pastikan kernel memakai accelerator 'NvidiaTeslaT4' (GPU T4 x2).", file=sys.stderr)
        return 3
    n = max(1, n_gpu)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    forwarded = [a for a in sys.argv[1:]]
    procs = []
    for shard in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard) if n_gpu else "", TOKENIZERS_PARALLELISM="false")
        cmd = [sys.executable, os.path.abspath(__file__), *forwarded, "--shard", str(shard), "--num-shards", str(n)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"GALAT: worker gagal dengan kode {codes}", file=sys.stderr)
        return 1
    items = read_items(args.data_dir, not args.no_queries, args.max_items)
    total = merge_parts(work, n, args.out, [it["key"] for it in items])
    metas = [json.loads(p.read_text()) for p in sorted(work.glob("part-*.meta.json"))]
    Path(str(args.out) + ".manifest.json").write_text(json.dumps(
        {"model": args.model, "revision": args.revision, "num_beams": args.num_beams, "top_n": args.top_n,
         "entries": total, "workers": metas}, indent=2), encoding="utf-8")
    print(f"Selesai: {total} entri -> {args.out}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder BEIR (corpus.jsonl, queries.jsonl)")
    p.add_argument("--out", required=True, help="berkas JSONL keluaran")
    p.add_argument("--work-dir", default="/kaggle/working/kp_parts")
    p.add_argument("--model", default="taln-ls2n/bart-base-kp20k")
    p.add_argument("--revision", default=None, help="commit HF untuk dipin (disarankan)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-source-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--no-queries", action="store_true")
    p.add_argument("--max-items", type=int, default=None, help="batasi jumlah teks (uji asap)")
    p.add_argument("--require-gpus", type=int, default=0, help="gagal cepat bila jumlah GPU berbeda")
    p.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()
    if args.shard is not None:
        return worker_main(args)
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
