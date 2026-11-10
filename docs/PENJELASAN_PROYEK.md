# Penjelasan Proyek: Hybrid Semantic-Keyphrase Retrieval

Dokumen ini menjelaskan proyek secara sederhana: apa yang dibuat, bagaimana cara kerjanya, dan apa hasilnya.
Diagram alurnya ada di `../Pipeline_Hybrid_Semantic_Keyphrase_Retrieval_V2.xml` (folder `Program`, di luar
`hybrid-ir`; buka di app.diagrams.net).
Rincian kode per berkas ada di `docs/PETA_KODE.md`.

---

## 1. Proyek ini tentang apa?

Kita membuat **mesin pencari untuk dokumen ilmiah** (judul + abstrak paper). Pengguna mengetik query, misalnya
"mice lacking vitamin D receptor develop rickets", lalu sistem mengembalikan dokumen yang paling relevan.

Pertanyaan penelitiannya: **apakah menggabungkan pencarian semantik dan pencocokan keyphrase menghasilkan
pencarian yang lebih baik daripada salah satunya saja?** BM25 dipakai sebagai pembanding (baseline).

## 2. Tiga cara menilai kecocokan dokumen

| Jalur | Cara kerja | Kelebihan | Kekurangan |
|---|---|---|---|
| **A. Semantik** | Teks diubah menjadi vektor angka oleh model SBERT (`all-MiniLM-L6-v2`). Dokumen yang maknanya mirip punya vektor yang berdekatan (cosine similarity). | Paham sinonim dan parafrase ("car" ≈ "automobile"). | Kurang tajam untuk istilah teknis yang langka. |
| **B. Keyphrase** | Dari tiap dokumen diambil 10 frasa kunci berbobot, lalu dicocokkan dengan frasa kunci query. | Fokus ke topik inti dokumen; mudah dijelaskan ("cocok karena frasa X"). | Hanya cocok bila frasanya sama. |
| **C. BM25** (pembanding) | Metode klasik: menghitung kata query yang muncul di dokumen; kata langka diberi bobot lebih besar. | Sangat kuat untuk istilah persis (nama gen, obat). | Tidak paham makna. |

Semantik (A) dan keyphrase (B) adalah bahan utama. BM25 (C) **tidak ikut digabung**; ia hanya dijalankan sendiri
sebagai pembanding.

Keyphrase dibuat oleh satu model, **BART-KPG** (`taln-ls2n/bart-base-kp20k`): model AI yang dilatih di ~500 ribu
paper. Bisa *menghasilkan* frasa, tidak hanya menyalin dari teks. Butuh GPU, jadi dijalankan di
**Kaggle (2× GPU T4)**, lalu hasilnya disimpan ke `artifacts/keyphrases_*.jsonl`.

## 3. Bagaimana skor digabung (fusi)?

Jalur semantik dan keyphrase masing-masing menghasilkan skor untuk tiap dokumen. Kandidatnya adalah gabungan
100 dokumen teratas dari tiap jalur. Skor kandidat dinormalkan ke rentang 0..1 (min-max), lalu digabung:

`skor hybrid = α·semantik + (1−α)·keyphrase`

Bobot α **tidak ditebak**: dicoba 0,0; 0,1; …; 1,0 di query latih (train/dev), yang terbaik dipilih, lalu hasil
dilaporkan di query uji (test) yang tidak pernah dipakai untuk memilih α. Ini supaya hasilnya jujur.

## 4. Alur kerja (3 fase)

1. **Indexing (offline)**, perintah `hybrid-ir build`: semua dokumen dibersihkan, diubah jadi vektor, dan diambil
   keyphrase-nya. Hasilnya disimpan sebagai index di folder `indexes/`.
2. **Pencarian (online)**, perintah `hybrid-ir search` atau UI: query diproses dengan cara yang sama, dihitung skornya
   di tiap jalur, digabung, lalu diurutkan (top-k).
3. **Evaluasi**, perintah `hybrid-ir evaluate`: empat skenario (BM25, semantik saja, keyphrase saja, hybrid)
   dibandingkan memakai query uji yang sudah punya jawaban benar (qrels), dan hasilnya disimpan di `results/`.

## 5. Cara mengukur "lebih baik"

- **nDCG@10** (metrik utama): seberapa baik dokumen relevan ditaruh di 10 peringkat teratas. 1,0 = sempurna.
- Metrik lain: MAP@100, Recall@10/100, Precision@10, MRR@10.
- **Uji signifikansi (paired bootstrap, 10.000 resample)**: memastikan selisih bukan kebetulan. Hybrid dibandingkan
  dengan semantik saja, keyphrase saja, dan BM25. Selisih dianggap nyata bila p < 0,05 dan rentang kepercayaan
  (CI 95%) tidak melewati nol.

## 6. Data uji

| Dataset | Isi | Dokumen | Query latih (untuk α) | Query uji |
|---|---|---|---|---|
| **SciFact** | klaim ilmiah vs abstrak (biomedis) | 5.183 | 809 (`train`) | 300 |
| **NFCorpus** | pertanyaan nutrisi/medis; banyak dokumen relevan per query | 3.633 | 324 (`dev`) | 323 |

## 7. Hasil (nDCG@10 pada query uji)

| Mode | SciFact | NFCorpus |
|---|---|---|
| BM25 (pembanding) | 0,6834 | 0,3278 |
| Semantik saja | 0,6497 | 0,3174 |
| Keyphrase saja | 0,4078 | 0,2321 |
| **Hybrid (semantik + keyphrase), α = 0,9** | **0,6671** | **0,3219** |

Eksperimen terpisah (bukan bagian pipeline): fine-tune encoder dengan data KP20k di Kaggle **tidak** meningkatkan
hasil (SciFact semantik 0,6497 → 0,6506, tidak signifikan), jadi tidak dipakai.

## 8. Kesimpulan (apa adanya)

1. **Hybrid lebih baik daripada semantik saja** dan perbedaannya signifikan di kedua dataset
   (SciFact +0,017, p = 0,006; NFCorpus +0,005, p = 0,035). Kenaikannya kecil.
2. **Hybrid jauh lebih baik daripada keyphrase saja** (SciFact +0,259; NFCorpus +0,090; signifikan).
3. **Hybrid belum mengalahkan BM25**: selisihnya negatif kecil (SciFact −0,016; NFCorpus −0,006) dan tidak
   signifikan, jadi keduanya bisa dianggap setara secara statistik.
4. α terpilih 0,9 di kedua dataset: sinyal semantik dominan, keyphrase berperan sebagai koreksi kecil.
5. Kenapa BM25 kuat? Query di SciFact/NFCorpus penuh istilah teknis yang sama persis dengan dokumen, dan model
   SBERT yang dipakai kecil dan bersifat umum (bukan khusus biomedis).

## 9. Cara menjalankan

```bash
cd hybrid-ir
source ../.venv/bin/activate
hybrid-ir search --config configs/scifact_bart.yaml "vitamin D receptor rickets" --mode hybrid
hybrid-ir ui     --config configs/scifact_bart.yaml          # buka http://127.0.0.1:8501
hybrid-ir evaluate --config configs/scifact_bart.yaml --out results/scifact_bart
```

Di UI, pilih mode di sidebar (Hybrid, Semantic saja, Keyphrase saja, BM25 saja) dan atur α untuk mode Hybrid.
Tab "Bandingkan mode" menampilkan hasil keempat mode berdampingan.

## 10. Peta folder

| Folder | Isi |
|---|---|
| `src/hybrid_ir/` | kode utama (encoder, keyphrase, index, fusi, evaluasi, CLI, UI) |
| `configs/` | pengaturan per dataset (`scifact_bart.yaml`, `nfcorpus_bart.yaml`, `demo.yaml`); `_arsip/` = desain lama |
| `kaggle/` | skrip GPU: generate keyphrase BART (pipeline) dan fine-tune encoder (eksperimen, bukan pipeline) |
| `artifacts/` | keyphrase BART hasil Kaggle |
| `indexes/` | index hasil `build` |
| `results/` | hasil evaluasi (`metrics.md`, `metrics.json`, skor per query) |
| `tests/` | tes otomatis (`pytest`) |

## 11. Istilah singkat

- **Embedding / vektor**: deretan angka yang mewakili makna teks.
- **Cosine similarity**: ukuran kemiripan dua vektor (1 = sangat mirip).
- **Keyphrase**: frasa kunci yang meringkas topik dokumen.
- **Index**: struktur data yang disiapkan di awal supaya pencarian cepat.
- **Qrels**: daftar jawaban benar (dokumen relevan) untuk tiap query uji.
- **Fusi**: menggabungkan skor semantik dan keyphrase menjadi satu skor.
- **[R]**: bagian rancangan kelompok, bukan diambil dari artikel rujukan (A1 = Sharma & Kumar 2023,
  A2 = Sarwar, Noor & Miah 2022).
