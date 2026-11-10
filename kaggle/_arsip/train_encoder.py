#!/usr/bin/env python3
"""EKSPERIMEN OPSIONAL: fine-tune encoder embedding (MultipleNegativesRankingLoss) di Kaggle 2x T4.

Hasil fine-tuning BELUM TENTU lebih baik daripada model bawaan. Selalu bandingkan dengan
`hybrid-ir evaluate` (sebelum vs sesudah) pada split uji; pakai hasil ini hanya bila nDCG@10 naik.

Sumber pasangan (anchor -> positive):
  kp20k_title     : judul -> abstrak                       (korpus ilmiah bidang komputer)
  kp20k_keyphrase : keyphrase penulis digabung -> judul. abstrak   (melatih encoder paham query berbentuk keyphrase)
  kp20k_mixed     : gabungan keduanya
  beir_train      : query -> dokumen relevan dari qrels/train.tsv dataset BEIR (butuh --data-dir)

Multi-GPU: tanpa argumen tambahan, skrip ini memanggil dirinya sendiri lewat `torchrun` (DDP, satu
proses per GPU). Negatif dalam-batch dihitung per GPU. fp16 aktif (T4 tidak mendukung bf16).
Skrip tidak membaca kredensial apa pun; unduhan dataset memakai huggingface_hub (internet kernel harus aktif)
atau berkas lokal lewat --kp20k-file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


# ====================================================================================
# Data (fungsi murni; diuji lokal tanpa GPU)
# ====================================================================================
def iter_json_records(path):
    """Baca berkas JSON array ATAU JSON-lines."""
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.read(1)
        while head and head.isspace():
            head = fh.read(1)
        fh.seek(0)
        if head == "[":
            yield from json.load(fh)
        else:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


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


def _clean(s):
    return " ".join(str(s or "").split())


def kp20k_pairs(records, mode, max_pairs, seed, min_abstract_chars=200, min_title_chars=10):
    rng = random.Random(seed)
    pairs = []
    for rec in records:
        title, abstract = _clean(rec.get("title")), _clean(rec.get("abstract"))
        kps = [_clean(k) for k in (rec.get("keyphrases") or []) if _clean(k)]
        if len(abstract) < min_abstract_chars or len(title) < min_title_chars:
            continue
        if mode in ("kp20k_title", "kp20k_mixed"):
            pairs.append((title, abstract))
        if mode in ("kp20k_keyphrase", "kp20k_mixed") and kps:
            pairs.append(("; ".join(kps[:8]), f"{title}. {abstract}"))
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def beir_train_pairs(data_dir, split, max_pairs, seed):
    root = resolve_data_dir(data_dir)
    docs = {}
    with open(root / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                o = json.loads(line)
                t, x = _clean(o.get("title")), _clean(o.get("text"))
                docs[str(o["_id"])] = f"{t}. {x}" if t and x else (t or x)
    queries = {}
    with open(root / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                o = json.loads(line)
                queries[str(o["_id"])] = _clean(o["text"])
    pairs = []
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) >= 3 and row[0] != "query-id" and int(float(row[2])) > 0:
                if row[0] in queries and row[1] in docs:
                    pairs.append((queries[row[0]], docs[row[1]]))
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs]


def load_pairs(args):
    if args.source == "beir_train":
        if not args.data_dir:
            raise SystemExit("--data-dir wajib untuk source beir_train")
        return beir_train_pairs(args.data_dir, args.beir_split, args.max_pairs, args.seed)
    path = args.kp20k_file
    if not path:
        raise SystemExit("--kp20k-file wajib (induk mengunduhnya otomatis; lihat main())")
    return kp20k_pairs(iter_json_records(path), args.source, args.max_pairs, args.seed)


# ====================================================================================
# Pelatihan (butuh torch, sentence-transformers, datasets)
# ====================================================================================
def train(args):
    import torch
    import transformers
    import sentence_transformers
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.require_gpus and torch.cuda.device_count() != args.require_gpus:
        raise SystemExit(f"diminta {args.require_gpus} GPU, terdeteksi {torch.cuda.device_count()}")

    pairs = load_pairs(args)
    if len(pairs) < 100:
        raise SystemExit(f"hanya {len(pairs)} pasangan; terlalu sedikit untuk dilatih")
    if rank == 0:
        print(f"pasangan latih: {len(pairs)}; dunia: {world} proses; contoh: {pairs[0][0][:80]!r}", flush=True)

    model = SentenceTransformer(args.base_model, trust_remote_code=False)
    model.max_seq_length = args.max_seq_length
    ds = Dataset.from_dict({"anchor": [a for a, _ in pairs], "positive": [p for _, p in pairs]})
    loss = losses.MultipleNegativesRankingLoss(model)
    out = Path(args.output_dir)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=args.per_device_batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # hindari duplikat anchor/positive dalam satu batch
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=True,    # pooler BERT (mis. MiniLM) tidak ikut loss; tanpa ini DDP galat
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    resume = args.resume and any((out / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    final = out / "final"
    trainer.save_model(str(final))          # hanya rank 0 yang menulis
    if rank == 0:
        (out / "train_manifest.json").write_text(json.dumps({
            "base_model": args.base_model, "source": args.source, "pairs": len(pairs), "epochs": args.epochs,
            "per_device_batch": args.per_device_batch, "world_size": world, "lr": args.lr,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "log_history": trainer.state.log_history[-20:],
        }, indent=2), encoding="utf-8")
        print(f"Model akhir: {final}", flush=True)
        if not args.keep_checkpoints:
            shutil.rmtree(out / "checkpoints", ignore_errors=True)   # output kernel jadi kecil
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--source", choices=["kp20k_title", "kp20k_keyphrase", "kp20k_mixed", "beir_train"], default="kp20k_mixed")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--beir-split", default="train")
    p.add_argument("--kp20k-file", default=None, help="berkas train KP20k lokal (JSON/JSONL)")
    p.add_argument("--max-pairs", type=int, default=150_000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per-device-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=0, help="batasi langkah (uji asap)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output-dir", default="/kaggle/working/encoder_ft")
    p.add_argument("--require-gpus", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-checkpoints", action="store_true", help="simpan checkpoint antara (memperbesar output kernel)")
    p.add_argument("--no-launch", action="store_true", help=argparse.SUPPRESS)
    return p


def maybe_download_kp20k(args):
    if args.source != "beir_train" and not args.kp20k_file:
        from huggingface_hub import hf_hub_download
        args.kp20k_file = hf_hub_download(repo_id="taln-ls2n/kp20k", filename="train.json", repo_type="dataset")
    return args


def main():
    args = build_parser().parse_args()
    in_worker = "LOCAL_RANK" in os.environ or args.no_launch
    if in_worker:
        return train(args)
    # Induk: unduh data sekali, lalu luncurkan DDP lewat torchrun satu proses per GPU.
    args = maybe_download_kp20k(args)
    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except ImportError:
        n_gpu = 0
    if args.require_gpus and n_gpu != args.require_gpus:
        print(f"GALAT: diminta {args.require_gpus} GPU, terdeteksi {n_gpu}.", file=sys.stderr)
        return 3
    # buang --kp20k-file lama (dua token / satu token) agar tidak ganda saat diteruskan ke worker
    cleaned, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--kp20k-file":
            skip = True
            continue
        if a.startswith("--kp20k-file="):
            continue
        cleaned.append(a)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={max(1, n_gpu)}",
           os.path.abspath(__file__), *cleaned]
    if args.kp20k_file:
        cmd += ["--kp20k-file", args.kp20k_file]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
