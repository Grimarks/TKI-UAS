# Alur presentasi

Rencana untuk presentasi sekitar **20 menit + tanya jawab**. Kalau dosen menetapkan durasi lain, kurangi atau tambah
secara proporsional; urutan dan pembicara tetap. Semua gambar diambil dari `analisis/gambar/` (sama dengan laporan) dan
`modul/gambar/gambar1_pipeline.png` (Gambar 1). Semua angka berasal dari `analisis/README.md`.

| No | Slide | Pembicara | Durasi | Isi pokok | Gambar / bahan |
|---|---|---|---|---|---|
| 1 | Judul dan anggota | Darrell | 0:30 | judul, nama, NIM | — |
| 2 | Masalah: *vocabulary gap* | Darrell | 1:30 | "heart attack" vs "myocardial infarction": BM25 = 0, cosine embedding = 0.5936; kelemahan embedding pada istilah langka | modul Bab 01 |
| 3 | Pertanyaan penelitian dan kontribusi | Darrell | 1:00 | RQ1–RQ3; BM25 hanya pembanding | — |
| 4 | Arsitektur | Darrell | 2:00 | tiga fase, dua jalur utama, BM25 putus-putus | `modul/gambar/gambar1_pipeline.png` |
| 5 | Data dan evaluasi | Jordi | 1:30 | SciFact vs NFCorpus (Tabel 2), tune vs test, nDCG@10 dan metrik lain, paired bootstrap | Tabel 2 laporan |
| 6 | Jalur A: semantik | Rabyndra | 1:30 | MiniLM 384 dimensi, normalisasi L2, cosine = perkalian matriks; persamaan (1) | modul Bab 04 |
| 7 | Jalur B: keyphrase | Leonardo | 2:00 | BART-KPG di Kaggle 2×T4, bobot urutan, unit frasa + kata, IDF, cosine; persamaan (2)–(4) | modul Bab 05 |
| 8 | Fusi | Darrell | 1:30 | kandidat C = 100, min-max, α·sem + (1−α)·kp; contoh 1185: 0.9868 vs 0.9143 | modul Bab 07 |
| 9 | Hasil utama | Jordi | 2:00 | Tabel 3: hybrid 0.6671 / 0.3219 > semantic > keyphrase; BM25 0.6834 / 0.3278 | `analisis/gambar/g1_ndcg10_skenario.png` |
| 10 | Signifikansi dan per kueri | Jordi | 1:30 | Tabel 4: vs semantic p = 0.006 / 0.035; vs BM25 p = 0.354 / 0.590 (tidak signifikan ≠ setara) | `analisis/gambar/g3_selisih_per_query.png` |
| 11 | Pengaruh α | Darrell | 1:00 | α = 0.9 di kedua koleksi; kurva test hanya analisis | `analisis/gambar/g2_pengaruh_alpha.png` |
| 12 | Analisis keyphrase | Leonardo | 1:30 | Tabel 5: *absent* sekitar 5%; cakupan 80.5% / 14.0% (qrels padat) vs 81.7% / 71.2% per kueri; yang tidak diukur | Tabel 5 laporan |
| 13 | Demo dan studi kasus | Surya | 4:00 | `presentasi/SKRIP_DEMO.md` langkah 1–4 | terminal / UI; Tabel 6 laporan |
| 14 | Kesimpulan, keterbatasan, rencana | Darrell | 1:00 | jawaban RQ1–RQ3; zero-shot, satu encoder kecil, dua koleksi biomedis, keyphrase domain ilmu komputer | Bab 5 laporan |
| — | Tanya jawab | semua | — | penjawab utama sesuai `PEMBAGIAN_TUGAS.md` (P1–P30) | modul Bab 12 |

Total: 22:30. Untuk tepat 20 menit, persingkat slide 2, 6, dan 7 masing-masing 30 detik dan demo menjadi 3 menit (langkah 1–3).

## Catatan untuk penyusun slide

- Pakai gambar PNG 300 dpi dari `analisis/gambar/` apa adanya, karena warnanya sudah konsisten: Hybrid biru, skenario
  lain abu-abu.
- Satu slide = satu pesan. Tulis angkanya besar, dan letakkan rincian di laporan.
- Di slide 10, tulis eksplisit: "tidak signifikan **bukan** berarti setara".
- Di slide 12, tulis eksplisit: "manfaat keyphrase *absent* untuk *vocabulary gap* belum diukur".
- Desain lama (YAKE, RRF, fine-tune) tidak dibahas di slide.

## Transisi antarpembicara

| Dari → ke | Kalimat jembatan |
|---|---|
| Darrell → Jordi (slide 4 → 5) | "Untuk menguji pipeline ini, kami memakai dua koleksi BEIR; Jordi akan menjelaskan datanya." |
| Jordi → Rabyndra (5 → 6) | "Sekarang kita lihat jalur pertama, yaitu jalur semantik." |
| Rabyndra → Leonardo (6 → 7) | "Embedding kuat untuk makna, tetapi lemah untuk istilah spesifik; di sinilah jalur keyphrase masuk." |
| Leonardo → Darrell (7 → 8) | "Kedua skor ini lalu digabung." |
| Darrell → Jordi (8 → 9) | "Bagaimana hasilnya? Jordi akan memaparkan." |
| Jordi → Darrell (10 → 11) | "Hasil ini bergantung pada bobot α." |
| Darrell → Leonardo (11 → 12) | "Kenapa bobot keyphrase kecil? Leonardo menjelaskan dari sisi keyphrase." |
| Leonardo → Surya (12 → 13) | "Mari lihat langsung di sistem." |
| Surya → Darrell (13 → 14) | "Dari demo ini, kita bisa menyimpulkan…" |
