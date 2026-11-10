# Bab 11 — Menjalankan program dan demo

## Tujuan belajar

1. Memasang proyek dari nol dan memastikan tes lulus.
2. Menjalankan keempat perintah utama: `build`, `search`, `ui`, `evaluate`.
3. Menjalankan demo presentasi dengan kueri yang sudah diuji.
4. Mengatasi kesalahan umum.

## 1. Instalasi

Lingkungan yang dipakai kelompok: Python 3.14 di MacBook Pro M4 Pro (`analisis/lingkungan.md`). Langkah dari nol:

```bash
cd "Program"
python3 -m venv .venv
.venv/bin/pip install -e "./hybrid-ir[models,ui,dev]"
cd hybrid-ir
../.venv/bin/python -m pytest        # harus: 156 passed
```

Dataset BEIR tidak ada di repositori. Unduh dengan:

```bash
../.venv/bin/hybrid-ir download scifact
../.venv/bin/hybrid-ir download nfcorpus
```

Keyphrase hasil Kaggle **sudah ada** di `artifacts/keyphrases_*.jsonl`, jadi tidak perlu GPU.

## 2. Perintah utama

```bash
# Fase 1: bangun indeks (encode dokumen + baca keyphrase), sekali per dataset
../.venv/bin/hybrid-ir build    --config configs/scifact_bart.yaml

# Fase 2: cari (mode: hybrid | semantic | keyphrase | bm25)
../.venv/bin/hybrid-ir search   --config configs/scifact_bart.yaml "betel nuts" --mode hybrid --top-k 5
../.venv/bin/hybrid-ir ui       --config configs/nfcorpus_bart.yaml      # buka http://127.0.0.1:8501

# Fase 3: evaluasi -> SIMPAN KE FOLDER BARU, jangan menimpa results/scifact_bart
../.venv/bin/hybrid-ir evaluate --config configs/scifact_bart.yaml --out results/latihan_scifact

# Tanpa internet / tanpa data besar: demo mini dengan encoder hashing
../.venv/bin/hybrid-ir demo
```

`--alpha` di `search` menimpa α dari konfigurasi (bawaan 0.9). Opsi `--json` menghasilkan keluaran JSON.

## 3. Skrip demo presentasi (sudah diuji)

Jalankan dengan `HF_HUB_OFFLINE=1` supaya model tidak menghubungi internet. Skrip lengkap dengan rencana cadangan ada
di `presentasi/SKRIP_DEMO.md`.

**Demo 1: hybrid menolong (NFCorpus "betel nuts").**

```bash
HF_HUB_OFFLINE=1 ../.venv/bin/hybrid-ir search --config configs/nfcorpus_bart.yaml "betel nuts" --mode semantic --top-k 2
HF_HUB_OFFLINE=1 ../.venv/bin/hybrid-ir search --config configs/nfcorpus_bart.yaml "betel nuts" --mode hybrid   --top-k 2
```

Yang diharapkan:

- semantic: #1 *The role of nuts in the optimal diet…* (tidak relevan), #2 *Association between betel-nut chewing and
  chronic kidney disease in men* (relevan);
- hybrid: #1 *Association between betel-nut chewing…*, dengan keterangan `cocok: betel-nut chewing`.

**Demo 2: kidney paired donation (SciFact 1185).** Tempel teks kueri 1185 (Bab 07) ke `--mode semantic`, lalu ke
`--mode hybrid`. Dokumen relevan *Kidney paired donation and optimizing the use of live donor organs* naik dari #3 ke #1,
dengan skor hybrid 0.9868 dan `cocok: kidney paired donation, optimization, renal transplantation`.

**Demo 3: kueri bebas di UI.** Misalnya "does coffee raise blood pressure" di NFCorpus. Kueri ini tidak ada di berkas
precomputed, jadi keyphrase-nya dibangkitkan BART-KPG lokal: `coffee (1.00), blood pressure (0.80)`. Pemanggilan pertama
butuh beberapa detik untuk memuat BART. Tab "Bandingkan mode" menampilkan keempat skenario berdampingan.

Catatan untuk demo: NFCorpus memuat beberapa dokumen ganda (misalnya MED-2155 dan MED-5244 berjudul sama). Kalau dua
hasil teratas sama persis, itu berasal dari data, bukan bug.

## 4. Kesalahan umum

| Gejala | Penyebab | Solusi |
|---|---|---|
| `Galat: indeks tidak ditemukan di indexes/...` | belum `build` | jalankan `hybrid-ir build --config ...` |
| `encoder berbeda ... build ulang indeks` | konfigurasi encoder diubah setelah build | samakan konfigurasi atau build ulang |
| *traceback* `MissingPrecomputed` di `demo.yaml` | kueri bebas, padahal demo tidak punya fallback BART | pakai salah satu kueri demo, atau konfigurasi `*_bart.yaml` |
| UI crash / *Metal assertion* di Mac | pemakaian model bersamaan di GPU Apple | sudah dicegah dengan lock; bila tetap terjadi, set `encoder.device: cpu` |
| Lambat / mencoba mengunduh model | tidak ada internet | tambahkan `HF_HUB_OFFLINE=1` |
| `streamlit belum terpasang` | instalasi tanpa `[ui]` | `pip install -e ".[ui]"` |
| angka `evaluate` sedikit beda di digit float terakhir | urutan `set` (lihat README) | metrik tetap sama; pakai `PYTHONHASHSEED=0` untuk run deterministik |

## Ringkasan

- Pasang dengan `pip install -e "./hybrid-ir[models,ui,dev]"`, lalu pastikan 156 tes lulus.
- Urutan kerja: `download` → `build` → `search`/`ui` → `evaluate` (ke folder baru).
- Demo terbaik: "betel nuts" (semantic vs hybrid) dan kueri 1185; kueri bebas memperlihatkan BART lokal.

## Soal latihan

**Soal 1.** Kenapa `evaluate` saat latihan harus memakai `--out` ke folder baru?

**Soal 2.** Perintah apa yang menampilkan hasil keempat skenario berdampingan?

**Soal 3.** Kamu mengetik kueri bebas di `configs/demo.yaml` dan muncul `MissingPrecomputed`. Kenapa?

## Kunci jawaban

**Jawaban 1.** `results/scifact_bart` dan `results/nfcorpus_bart` adalah sumber semua angka laporan, modul, dan slide.
Menimpanya bisa merusak keterlacakan angka.

**Jawaban 2.** `hybrid-ir ui --config ...`, lalu buka tab "Bandingkan mode".

**Jawaban 3.** `demo.yaml` sengaja memakai `query_fallback: none`, supaya demo jalan tanpa mengunduh BART. Keyphrase
hanya tersedia untuk 8 kueri demo di `data/demo/keyphrases_demo.jsonl`, sehingga kueri lain berhenti dengan galat
`MissingPrecomputed`. Pesannya menjelaskan sebabnya, walaupun CLI menampilkannya sebagai *traceback*.
