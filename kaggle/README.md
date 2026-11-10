# Skrip GPU untuk Kaggle

| Skrip | Fungsi | Perlu GPU? |
|---|---|---|
| `generate_keyphrases.py` | generate keyphrase (BART-KPG) untuk korpus + query; satu proses per GPU; hasil JSONL. **Bagian dari pipeline (Fase 1 — Jalur B).** | ya (T4 x2) |
| `build_kernel.py` | menyusun folder siap-push (`script.py` + `kernel-metadata.json`); jalan lokal | tidak |
| `_arsip/train_encoder.py` | ARSIP: fine-tune encoder (desain lama, bukan bagian desain final). Tesnya ikut diarsipkan di `_arsip/test_train_encoder.py` | ya (T4 x2) |

## Cara manual (ringkas)

```bash
pip install -U kaggle                       # lalu autentikasi sendiri (lihat README utama; jangan tempel token di chat)
# 1) unggah data (hanya corpus.jsonl + queries.jsonl) sebagai dataset private, mis. <user>/hybrid-ir-scifact
# 2) susun kernel
python kaggle/build_kernel.py generate --username <user> --slug hybrid-ir-kp-scifact \
  --dataset-source <user>/hybrid-ir-scifact \
  -- --data-dir /kaggle/input/hybrid-ir-scifact --out /kaggle/working/keyphrases_scifact.jsonl --require-gpus 2
# 3) jalankan di T4 x2 dan ambil hasilnya
kaggle kernels push   -p kaggle/_build/hybrid-ir-kp-scifact --accelerator NvidiaTeslaT4
kaggle kernels status <user>/hybrid-ir-kp-scifact
kaggle kernels output <user>/hybrid-ir-kp-scifact -p artifacts_raw
```

Catatan:

- Pada CLI Kaggle saat ini `NvidiaTeslaT4` berarti "GPU T4 x2". Skrip memakai `--require-gpus 2` agar gagal cepat bila
  yang didapat bukan dua GPU.
- Kernel script tidak bisa menerima argumen CLI; `build_kernel.py` menyisipkan `sys.argv` di atas salinan skrip.
- Mengunduh model dari Hugging Face di dalam kernel butuh internet aktif (akun terverifikasi telepon). Alternatif: unggah
  model sebagai dataset private dan pakai `--model /kaggle/input/<dataset>` dengan `--no-internet`.
- Format keluaran dibaca `hybrid-ir` lewat `keyphrase.method: precomputed` (kunci = SHA-1 teks ternormalisasi;
  kesamaan fungsi normalisasi dijaga oleh `tests/test_kaggle_scripts.py`).
- `generate_keyphrases.py` sudah dijalankan di Kaggle 2×T4 (Oktober 2026); hasilnya ada di
  `artifacts/keyphrases_*.jsonl` + `.manifest.json`. Tetap lakukan uji asap dulu (`--max-items 64`) bila versi library
  Kaggle berubah.
- Bug yang dilaporkan pada sebagian versi CLI 2.x: `kaggle datasets create` gagal dengan "dataset slugs and hashlink are
  all null". Jalan keluar: jalankan perintah itu dari venv terpisah berisi `kaggle==1.6.17` (hanya untuk membuat dataset;
  `kernels push --accelerator` butuh CLI yang lebih baru), atau unggah folder dataset lewat browser.
