# Hybrid Semantic-Keyphrase Retrieval

Sistem pencarian dokumen akademik yang menggabungkan **embedding (semantik)** dan **keyphrase generatif (BART-KPG)**,
dengan **BM25** sebagai pembanding (baseline). Mengikuti diagram draw.io
`../Pipeline_Hybrid_Semantic_Keyphrase_Retrieval_V2.xml`: indexing (offline) → query & retrieval (online) → evaluasi.
Penjelasan sederhana untuk pembaca awam: `docs/PENJELASAN_PROYEK.md`.

## Pipeline

| Komponen | Peran | Keterangan |
|---|---|---|
| Semantic (SBERT, `all-MiniLM-L6-v2`) | bahan utama | dokumen/query → vektor → cosine similarity |
| Keyphrase (BART-KPG, `taln-ls2n/bart-base-kp20k`) | bahan utama | satu-satunya sumber keyphrase; dokumen/query → keyphrase berbobot → cosine (atau Jaccard) |
| BM25 (k1=1.2, b=0.75) | pembanding kecil (baseline) | **tidak** ikut difusikan |
| Hybrid | hasil utama | **hanya** semantic + keyphrase: `s = α·minmax(sem) + (1−α)·minmax(kp)` pada kandidat union top-C; α ditala di split tune |

Evaluasi melaporkan tepat empat skenario: `bm25`, `semantic`, `keyphrase`, `hybrid` (weighted, α hasil tuning);
metrik nDCG@10, MAP@100, Recall@10/100, Precision@10, MRR@10; uji signifikansi paired bootstrap (10.000 resample)
untuk hybrid vs semantic, hybrid vs keyphrase, dan hybrid vs bm25.

Rujukan: **A1** = Sharma & Kumar (2023), **A2** = Sarwar, Noor & Miah (2022). Pemetaan lengkap ada di bagian bawah.

## 1. Isi proyek

```
configs/            demo.yaml (offline), scifact_bart.yaml, nfcorpus_bart.yaml; _arsip/ = konfigurasi desain lama
data/               demo/ (korpus mini 24 dokumen, 8 query, keyphrases_demo.jsonl); scifact/, nfcorpus/ (BEIR)
src/hybrid_ir/      paket Python (lihat docs/PETA_KODE.md)
kaggle/             skrip GPU (generate keyphrase; fine-tune encoder = eksperimen, bukan pipeline) + pembuat kernel
artifacts/          keyphrase BART hasil Kaggle (keyphrases_scifact.jsonl, keyphrases_nfcorpus.jsonl + manifest)
indexes/, results/  index hasil `build` dan hasil `evaluate`; _arsip/ = hasil desain lama
docs/               PENJELASAN_PROYEK.md (ringkasan sederhana), PETA_KODE.md
tests/              156 tes otomatis (153 bila Streamlit belum terpasang; 3 tes UI dilewati)
```

## 2. Instalasi

Butuh Python 3.10+ (sudah dijalankan di Python 3.14, macOS Apple Silicon).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[models,ui,dev]"    # tanpa model/UI: pip install -e .
pytest                               # semua tes harus lulus; tidak ada unduhan model
```

Setelah semuanya berjalan di komputermu, kunci versi persis: `pip freeze > requirements.lock.txt`.

Perangkat otomatis dipilih: CUDA, lalu MPS (Apple Silicon), lalu CPU (`encoder.device: auto`).

## 3. Coba cepat (tanpa unduhan)

```bash
hybrid-ir demo                                   # bangun indeks mini + 2 pencarian contoh
hybrid-ir ui --config configs/demo.yaml          # UI di http://127.0.0.1:8501
```

Demo memakai encoder `hashing`. Itu **bukan model semantik** (hanya mencocokkan kata), jadi demo hanya membuktikan
alur dan UI berjalan. Keyphrase demo dibaca dari `data/demo/keyphrases_demo.jsonl`: itu **contoh tulis tangan**
(bukan keluaran BART) untuk 24 dokumen dan 8 query demo. Demo tidak memuat model BART (`query_fallback: none`), jadi
hanya 8 query di `data/demo/queries.jsonl` yang bisa dicari; query lain ditolak dengan pesan galat yang jelas.
Hasil bermakna butuh model nyata (bagian 4).

## 4. Pakai data dan model nyata (SciFact)

SciFact: 5.183 abstrak ilmiah, 300 query uji, ada split `train` (809 query) untuk menala alpha.

```bash
hybrid-ir download scifact                                       # ke data/scifact (md5 diverifikasi)
hybrid-ir build    --config configs/scifact_bart.yaml            # encode + baca keyphrase dari artifacts/
hybrid-ir search   --config configs/scifact_bart.yaml "mice lacking vitamin D receptor develop rickets"
hybrid-ir evaluate --config configs/scifact_bart.yaml --out results/scifact_bart
hybrid-ir ui       --config configs/scifact_bart.yaml
```

Unduhan pertama model `all-MiniLM-L6-v2` terjadi saat `build`. Keyphrase dokumen dan query evaluasi dibaca dari
`artifacts/keyphrases_scifact.jsonl`; query bebas (UI/CLI) yang tidak ada di berkas itu dibuat lokal oleh model BART
yang sama (`query_fallback: seq2seq`).

### Mode pencarian

| Mode | Isi skor | Catatan |
|---|---|---|
| `hybrid` | `α·semantik + (1−α)·keyphrase` | hasil utama; `retrieval.alpha` atau `--alpha` |
| `semantic` | semantik saja | |
| `keyphrase` | keyphrase saja | hanya dokumen yang punya keyphrase cocok |
| `bm25` | BM25 saja | pembanding; hanya dokumen yang punya kata cocok |

```bash
hybrid-ir search --config configs/scifact_bart.yaml "vitamin D receptor rickets" --mode hybrid --alpha 0.9
```

Mode bawaan diatur di `retrieval.mode` pada YAML. Index BM25 tidak perlu `build` ulang: dibangun di memori saat
pertama dipakai, dan hanya dihitung di mode `bm25`. Di UI, mode dan α diatur di sidebar; tab "Bandingkan mode"
menampilkan keempat mode berdampingan.

Dataset:

| Dataset | Dokumen | Query tuning | Query uji | Catatan |
|---|---|---|---|---|
| `scifact` | 5.183 | 809 (`train`) | 300 | utama; relevansi jarang (~1,1 dok/query) |
| `nfcorpus` | 3.633 | 324 (`dev`) | 323 | banyak dokumen relevan per query, metrik lebih stabil |

Periksa lisensi masing-masing dataset di halaman BEIR sebelum dipakai di luar tugas kuliah.
Jika server BEIR tidak bisa diakses dari jaringanmu, unduh manual `scifact.zip` dari halaman BEIR, ekstrak ke
`data/scifact/` sehingga berisi `corpus.jsonl`, `queries.jsonl`, `qrels/`.

### Korpus sendiri (mis. berbahasa Indonesia)

Buat folder dengan format yang sama:

```
mydata/corpus.jsonl      {"_id": "p1", "title": "Judul", "text": "Abstrak ..."}   satu objek per baris
mydata/queries.jsonl     {"_id": "q1", "text": "pertanyaan ..."}
mydata/qrels/test.tsv    query-id <TAB> corpus-id <TAB> skor   (baris pertama header; skor >0 = relevan)
```

Salin `configs/scifact_bart.yaml`, ubah `data.path`, `index.dir`, dan `keyphrase.language: id`. Generate keyphrase
korpusnya dengan `kaggle/generate_keyphrases.py` (atau `keyphrase.method: seq2seq` bila punya GPU lokal). Catatan:
BART-KPG dilatih pada paper berbahasa Inggris; untuk bahasa Indonesia ganti juga encoder ke model multibahasa
(mis. `intfloat/multilingual-e5-base` dengan prefix `query: ` / `passage: `), dan **buat qrels sendiri**: tidak ada
dataset Indonesia yang saya verifikasi untuk tugas ini.

## 5. Membaca hasil evaluasi

`results/<dataset>/metrics.md` berisi tabel empat skenario, uji signifikansi, dan sapuan alpha. Cara kerjanya:

- alpha (grid 0,0–1,0 langkah 0,1) ditala **hanya** pada query tuning (split `train`/`dev`; bila tak ada, 30% query
  uji), lalu dilaporkan pada query evaluasi yang tidak dipakai untuk menala. Sapuan alpha pada query evaluasi hanya
  untuk analisis.
- Uji signifikansi membandingkan `hybrid_weighted` dengan semantic, keyphrase, dan BM25.
- Query tanpa dokumen relevan di korpus dilewati.
- "Signifikan" berarti CI 95% paired bootstrap selisih nDCG@10 tidak melewati nol. Hibrida **belum tentu** mengalahkan
  semantic-only atau BM25; hasil seri atau kalah tetap temuan yang sah dan harus dilaporkan apa adanya.

## 6. Keyphrase generatif (BART) dan Kaggle

Keyphrase dibuat oleh checkpoint publik `taln-ls2n/bart-base-kp20k` (BART-base, fine-tune pada KP20k, lisensi
Apache-2.0). Model ini **generatif**: bisa menghasilkan keyphrase yang tidak muncul di teks. Melatih BART dari nol
**tidak diperlukan**. Bagian yang berat adalah men-generate keyphrase untuk seluruh korpus; itu dikerjakan di Kaggle
(2×T4), dan hasilnya dibaca lokal lewat `keyphrase.method: precomputed`. Lihat `kaggle/README.md`.

- `kaggle/generate_keyphrases.py`: generate keyphrase dokumen + query, satu proses per GPU. **Bagian dari pipeline.**
- `kaggle/train_encoder.py`: fine-tune encoder dengan DDP di 2 GPU. **Bukan bagian pipeline yang dilaporkan**; hanya
  eksperimen terpisah (tidak meningkatkan nDCG@10: 0,6497 → 0,6506, tidak signifikan). Skripnya tetap disimpan.

Pakai hasilnya: taruh berkas JSONL di `artifacts/keyphrases_<dataset>.jsonl`, lalu
`hybrid-ir build --config configs/<dataset>_bart.yaml` dan `hybrid-ir evaluate ...`.

Sudah dijalankan di Kaggle 2×T4 (Oktober 2026; torch 2.11, transformers 5.16, sentence-transformers 5.7):

| Job | Hasil | Waktu GPU |
|---|---|---|
| keyphrase SciFact | 6.288 teks unik | ~130 detik per GPU |
| keyphrase NFCorpus | 6.809 teks unik | ~100 detik per GPU |

Revisi model BART dipin ke commit `674e4b03753c70d23b0c5be504a81043dcb783a7` di `configs/*_bart.yaml`.

## 7. Hasil evaluasi (query uji)

| Skenario | SciFact nDCG@10 | NFCorpus nDCG@10 |
|---|---|---|
| BM25 (pembanding) | 0,6834 | 0,3278 |
| Semantic-only | 0,6497 | 0,3174 |
| Keyphrase-only | 0,4078 | 0,2321 |
| **Hybrid weighted** (α=0,9 di kedua dataset) | **0,6671** | **0,3219** |

Uji signifikansi (selisih nDCG@10 hybrid dikurangi pembanding, CI 95%, p):

| Pembanding | SciFact | NFCorpus |
|---|---|---|
| semantic | +0,0174 [+0,0048, +0,0307], p=0,006 | +0,0046 [+0,0003, +0,0092], p=0,035 |
| keyphrase | +0,2593 [+0,2141, +0,3035], p=0,0001 | +0,0899 [+0,0680, +0,1121], p=0,0001 |
| bm25 | −0,0163 [−0,0507, +0,0173], p=0,354 | −0,0058 [−0,0280, +0,0154], p=0,590 |

Ringkasan (rincian di `results/*/metrics.md` dan `docs/PENJELASAN_PROYEK.md`):

- Hybrid signifikan lebih baik daripada semantic-only dan keyphrase-only di kedua dataset.
- Hybrid **tidak** mengalahkan BM25: selisihnya negatif kecil dan tidak signifikan.
- α terpilih 0,9, artinya sinyal semantik dominan dan keyphrase berperan sebagai koreksi kecil.

Hasil dari desain penelitian sebelumnya disimpan di `results/_arsip/` (tidak dilaporkan lagi).

## 8. Keamanan

- Tidak ada `pickle`: indeks disimpan sebagai `.npy`/`.npz`/`.json`; `np.load(allow_pickle=False)`.
- YAML dibaca dengan `safe_load`; konfigurasi divalidasi ketat (kunci salah ketik ditolak, nilai di luar rentang ditolak,
  kunci dari desain lama ditolak dengan pesan yang menjelaskan sebabnya).
- Model: `trust_remote_code=False` selalu; generator keyphrase memuat `safetensors` saja (`require_safetensors: true`).
  Pin `revision` (commit hash) model di YAML agar versi tidak berubah diam-diam.
- Indeks memuat sidik jari encoder; indeks yang dibangun dengan encoder berbeda **ditolak** saat dimuat.
- `save` menolak menimpa folder yang bukan indeks hybrid-ir (mencegah salah isi `index.dir` menghapus file lain).
- Pengunduh dataset: daftar putih dataset, satu host HTTPS, redirect hanya ke host yang sama, batas ukuran,
  verifikasi MD5, ekstraksi ZIP menolak path berbahaya/symlink/arsip membengkak (zip-slip).
- UI: hanya `127.0.0.1`, tanpa unggah berkas, tanpa input path, tanpa `unsafe_allow_html`, semua teks data di-escape;
  panjang query dibatasi.
- Rahasia: `.gitignore` memblokir `kaggle.json`, `.env`, `access_token`. Jangan pernah menempel token Kaggle di chat,
  commit, notebook, atau log. Simpan di `~/.kaggle/` (macOS/Linux: `chmod 600`) atau environment variable sesi terminal.
  Bila token pernah bocor, cabut di pengaturan akun Kaggle dan buat baru.

## 9. Pemetaan ke pipeline (diagram V2) dan artikel

| Kotak di diagram | Kode | Rujukan |
|---|---|---|
| Pra-pemrosesan | `text.py` (`normalize_text`, `tokenize`, stemming) | A1, A2 |
| Embedding encoder + Vector index | `encoders.py`, `indexes.DenseIndex` | [R] (A1/A2 memakai embedding kata; di sini embedding kalimat) |
| Generasi keyphrase | `keyphrases.py` (seq2seq BART-KPG, precomputed) | A2 (keyphrase + bobot), [R] BART-KPG |
| Keyphrase index (keyphrase + bobot) | `indexes.KeyphraseIndex` | A2 |
| Query diproses seperti dokumen | `retriever.score_matrices` | A1 (Bab 7), A2 |
| Semantic retrieval / Keyphrase matching | `DenseIndex.scores`, `KeyphraseIndex.scores` (cosine, Jaccard) | A2 |
| Pembanding BM25 | `bm25.py`, `HybridRetriever.bm25` / `bm25_scores` | [R] |
| Normalisasi skor & fusi | `fusion.py` (`minmax`, `candidate_set`, `fuse_weighted`) | terinspirasi Relwt (A1) |
| Ranking top-k | `fusion.rank_documents` → CLI `search`, `ui_app.py` | A2 |
| Skenario, metrik, tuning alpha, analisis | `evaluation.py` | A1 (P@k, MAP), A2 (NDCG) |
| Opsional: Kaggle GPU | `kaggle/` | [R] |

[R] = rancangan kelompok, bukan diambil dari artikel. Tidak ada klaim bahwa fusi atau encoder ini berasal dari A1/A2.

## 10. Batasan yang perlu kamu tahu

- 156 tes otomatis memakai encoder pengganti (hashing), keyphrase contoh, dan data mini. Pipeline lengkap juga sudah
  dijalankan dengan model nyata pada SciFact dan NFCorpus, dan skrip Kaggle sudah dijalankan di 2×T4 (bagian 6 dan 7).
  Pustaka AI berubah cepat; perbedaan versi mungkin perlu penyesuaian kecil.
- Hasil hanya untuk dua dataset biomedis/nutrisi berbahasa Inggris dan satu encoder kecil (MiniLM). Kesimpulan bisa
  berbeda pada domain atau encoder lain.
- Keyphrase dari query pendek (mis. kalimat klaim di SciFact) sedikit jumlahnya; sinyal keyphrase pada query bisa lemah.
- Pra-pemrosesan: huruf kecil dan pembuangan tanda baca diterapkan untuk BM25 dan pencocokan keyphrase; teks untuk
  encoder hanya dinormalkan (Unicode, spasi), karena tokenizer encoder menangani sendiri.
- Perbandingan cosine vs Jaccard untuk keyphrase belum dijalankan otomatis; semua hasil di atas memakai cosine
  (Jaccard tersedia lewat `keyphrase.similarity: jaccard`).
- UI di Apple Silicon: pencarian di-serialisasi dengan lock karena pemakaian model bersamaan di MPS bisa membuat proses
  crash. Bila tetap crash, set `encoder.device: cpu`.
