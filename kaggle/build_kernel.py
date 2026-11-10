#!/usr/bin/env python3
"""Susun folder siap-push untuk `kaggle kernels push` (dijalankan LOKAL; tidak memanggil jaringan).

Kaggle "script kernel" tidak bisa menerima argumen baris perintah, jadi kita menyisipkan
`sys.argv` di bagian atas salinan skrip. Contoh:

  python kaggle/build_kernel.py generate --username USER --slug hybrid-ir-kp-scifact \
      --dataset-source USER/hybrid-ir-scifact \
      -- --data-dir /kaggle/input/hybrid-ir-scifact --out /kaggle/working/keyphrases_scifact.jsonl --require-gpus 2

Hasil: kaggle/_build/<slug>/{script.py, kernel-metadata.json}. Lalu:
  kaggle kernels push -p kaggle/_build/<slug> --accelerator NvidiaTeslaT4

Catatan: pada CLI Kaggle saat ini, 'NvidiaTeslaT4' = GPU T4 x2 (satu T4 tidak tersedia).
Skrip ini TIDAK membaca atau menyimpan kredensial apa pun.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = {"generate": "generate_keyphrases.py", "train": "train_encoder.py"}
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{2,49}$")
USER = re.compile(r"^[a-z0-9][a-z0-9_-]{1,49}$")
DATASET = re.compile(r"^[a-z0-9][a-z0-9_-]{1,49}/[a-z0-9][a-z0-9_-]{2,49}$")


def build(job: str, username: str, slug: str, dataset_sources: list[str], internet: bool,
          script_args: list[str], out_root: Path) -> Path:
    if job not in JOBS:
        raise ValueError(f"job tidak dikenal: {job}")
    if not USER.match(username):
        raise ValueError("username Kaggle tidak valid (huruf kecil, angka, - atau _)")
    if not SLUG.match(slug):
        raise ValueError("slug tidak valid: 3-50 karakter huruf kecil/angka/tanda hubung")
    for ds in dataset_sources:
        if not DATASET.match(ds):
            raise ValueError(f"dataset source tidak valid (format pemilik/nama): {ds}")
    body = (HERE / JOBS[job]).read_text(encoding="utf-8")
    header = (
        "import sys\n"
        "# Argumen job disisipkan oleh build_kernel.py (hanya bila skrip dijalankan tanpa argumen).\n"
        "if len(sys.argv) == 1:\n"
        f"    sys.argv = [\"job.py\"] + {json.dumps(script_args)}\n\n"
    )
    folder = out_root / slug
    folder.mkdir(parents=True, exist_ok=True)
    # `from __future__` harus berada paling atas modul; pindahkan ke atas header.
    future_line = "from __future__ import annotations\n"
    if future_line in body:
        body = body.replace(future_line, "", 1)
        header = future_line + header
    (folder / "script.py").write_text(header + body, encoding="utf-8")
    meta = {
        "id": f"{username}/{slug}",
        "title": slug,                       # judul harus 'resolve' ke slug yang sama
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": internet,
        "machine_shape": "NvidiaTeslaT4",    # = GPU T4 x2 pada Kaggle CLI saat ini
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    script_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, script_args = argv[:cut], argv[cut + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job", choices=sorted(JOBS))
    p.add_argument("--username", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--dataset-source", action="append", default=[])
    p.add_argument("--no-internet", action="store_true")
    p.add_argument("--out", default=str(HERE / "_build"))
    a = p.parse_args(argv)
    folder = build(a.job, a.username, a.slug, a.dataset_source, not a.no_internet, script_args, Path(a.out))
    print(f"Siap di: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
