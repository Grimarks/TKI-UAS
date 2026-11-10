# Skrip demo presentasi

Penanggung jawab: **Surya Andika** (cadangan: Darrell Satriano). Durasi sekitar 4 menit.

Semua langkah memakai konfigurasi **`configs/scifact_bart.yaml`** dan **`configs/nfcorpus_bart.yaml`** (bukan
`demo.yaml`), dengan kueri dari studi kasus laporan (Tabel 6). Semua perintah sudah diuji, dan keluaran yang
diharapkan di bawah disalin dari hasil uji itu (`presentasi/cadangan/langkah*.txt`).

## Persiapan (H−1 dan 15 menit sebelum tampil)

```bash
cd "Program/hybrid-ir"
../.venv/bin/python -m pytest -q                    # 157 passed
ls indexes/scifact_bart indexes/nfcorpus_bart       # kedua indeks harus ada (kalau belum: hybrid-ir build)
bash presentasi/demo.sh                             # jalankan semua langkah sekali; memanaskan cache model
```

- Pakai **`HF_HUB_OFFLINE=1`** (sudah di dalam `demo.sh`) supaya tidak ada akses internet ke Hugging Face.
- Perbesar ukuran huruf terminal dan siapkan juga UI: `HF_HUB_OFFLINE=1 ../.venv/bin/hybrid-ir ui --config
  configs/nfcorpus_bart.yaml`, lalu buka http://127.0.0.1:8501.
- Ambil tangkapan layar UI (tab "Bandingkan mode" untuk "betel nuts") saat gladi dan simpan ke `presentasi/cadangan/`.

## Langkah demo

Jalankan satu per satu: `bash presentasi/demo.sh 1`, `bash presentasi/demo.sh 2`, dan seterusnya.

### Langkah 1: hybrid menolong, NFCorpus "betel nuts"

```bash
HF_HUB_OFFLINE=1 ../.venv/bin/hybrid-ir search --config configs/nfcorpus_bart.yaml "betel nuts" --mode semantic --top-k 3
HF_HUB_OFFLINE=1 ../.venv/bin/hybrid-ir search --config configs/nfcorpus_bart.yaml "betel nuts" --mode hybrid   --top-k 3
```

Yang diharapkan:

- semantic: #1 `[MED-2383] The role of nuts in the optimal diet…` (tidak relevan), #2 `[MED-5017] Association between
  betel-nut chewing and chronic kidney disease in men.` (relevan);
- hybrid: #1 `[MED-5017]` dengan `keyphrase=0.3989` dan `cocok: betel-nut chewing`.

Yang diucapkan: "Embedding menangkap 'nuts' secara umum. Keyphrase dokumen relevan 'betel-nut chewing' berbagi kata
'betel' dan 'nuts' dengan kueri, sehingga dokumen yang benar naik ke peringkat 1."

### Langkah 2: hybrid menolong, SciFact 1185 (kidney paired donation)

```bash
bash presentasi/demo.sh 2
```

Yang diharapkan:

- semantic: #1 `[21203899] Managing finances of shipping living donor kidneys…`, dan dokumen relevan `[16737210]` baru
  di #3;
- hybrid: #1 `[16737210] Kidney paired donation and optimizing the use of live donor organs.`, `skor=0.9868`,
  `cocok: kidney paired donation, optimization, renal transplantation`.

Yang diucapkan: "Skor semantik tiga dokumen teratas hampir sama (0.6013, 0.6002, 0.5918). Keyphrase 'kidney paired
donation' menjadi penentu, padahal bobotnya hanya 0.1."

### Langkah 3: hybrid kalah dari BM25, SciFact 1024 (CTCF)

```bash
bash presentasi/demo.sh 3
```

Yang diharapkan:

- bm25: #1 `[5373138] 3D Chromosome Regulatory Landscape of Human Pluripotent Cells.` (relevan), `bm25=28.8398`;
- hybrid: #1 `[8494570] Characterization of constitutive CTCF/cohesin loci…` (tidak relevan); dokumen relevan ada di
  peringkat 49 (Tabel 6).

Yang diucapkan: "Kata langka 'CTCF' dan 'anchor' cocok persis untuk BM25. Embedding dan keyphrase tidak menangkapnya.
Ini sejalan dengan Chen dkk. [6] tentang istilah langka."

### Langkah 4: hybrid kalah dari BM25, NFCorpus "kohlrabi"

```bash
bash presentasi/demo.sh 4
```

Yang diharapkan:

- bm25: #1 `[MED-4455] Glucoraphanin and 4-hydroxyglucobrassicin contents in seeds of 59 cultivars of broccoli, raab,
  kohlrabi…` (relevan);
- keyphrase: **tidak ada hasil** (hanya baris Query/Mode/Keyphrase), karena tidak ada dokumen yang keyphrase-nya memuat
  "kohlrabi";
- hybrid: #1 `[MED-3947] Hibiscus anthocyanins…` dengan `keyphrase=0.0000`.

Yang diucapkan: "BART hanya membangkitkan tiga keyphrase untuk dokumen ini, dan tidak ada yang memuat 'kohlrabi'.
Keyphrase yang dibangkitkan tidak selalu mencakup semua istilah penting."

### Langkah 5: kueri bebas (opsional, bila waktu cukup)

```bash
bash presentasi/demo.sh 5
```

Kueri "does coffee raise blood pressure" tidak ada di berkas precomputed, jadi keyphrase-nya dibangkitkan **BART-KPG
lokal**: `Keyphrase : coffee (1.00), blood pressure (0.80)`. Pemuatan BART pertama kali butuh beberapa detik.

> **Catatan dokumen ganda.** NFCorpus memuat dokumen ganda: `[MED-2155]` dan `[MED-5244]` berjudul sama persis
> (*Effects of habitual coffee consumption on cardiometabolic disease…*) dengan skor sama (0.9860), sehingga muncul
> sebagai #1 dan #2 pada langkah 5. Itu berasal dari data NFCorpus, bukan bug sistem. Kalau ditanya, jawab begitu.

## Rencana cadangan

| Masalah | Tindakan |
|---|---|
| Model gagal dimuat / terminal error | buka `presentasi/cadangan/langkah<N>.txt` (keluaran asli hasil uji) dan bacakan |
| UI tidak terbuka | pakai demo CLI (langkah 1–4), atau tunjukkan tangkapan layar UI di `presentasi/cadangan/` |
| Waktu habis | lewati langkah 3–5; langkah 1 dan 2 sudah cukup menunjukkan peran keyphrase |
| Laptop presentasi tidak punya indeks | jalankan `hybrid-ir build` untuk kedua konfigurasi H−1 (sekitar 1 menit per dataset), atau pakai keluaran cadangan |
| Hasil berbeda di digit terakhir | wajar (urutan `set`, lihat README); peringkat dan metrik tidak berubah |
