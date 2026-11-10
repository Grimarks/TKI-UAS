# Bab 00 — Cara memakai modul ini

Modul ini ditulis supaya **setiap anggota Kelompok 1** paham sistem *Hybrid Semantic-Keyphrase Retrieval* sampai
bisa menjelaskannya sendiri di depan dosen, walaupun tidak ikut menulis kodenya. Isinya santai, tapi setiap angka dan
potongan kode diambil dari proyek yang sebenarnya:

- **Potongan kode adalah kode asli.** Setiap blok diberi keterangan "Kode asli: `berkas`, baris a–b" dan dibangkitkan
  otomatis dari `src/` oleh `scripts/modul.py`. Kalau kodenya berubah, `scripts/modul.py --cek` akan gagal, jadi
  nomor baris di modul tidak akan basi diam-diam.
- **Angka hasil diambil dari `analisis/` dan `results/`.** Angka contoh hitungan (misalnya cosine "heart attack" vs
  "myocardial infarction") dihitung dengan fungsi proyek dan disimpan di `modul/data/contoh_angka.json`.
- **Gambar** berasal dari `analisis/gambar/` (sama dengan yang dipakai di laporan dan slide) dan
  `modul/gambar/gambar1_pipeline.png` (Gambar 1 laporan).

## Urutan belajar

| Bab | Isi | Waktu baca |
|---|---|---|
| 01 | Masalah IR, *vocabulary gap*, leksikal vs semantik | 20 menit |
| 02 | Arsitektur: Gambar 1 kotak demi kotak | 25 menit |
| 03 | Data BEIR: SciFact dan NFCorpus | 20 menit |
| 04 | Jalur A: embedding semantik | 25 menit |
| 05 | Jalur B: keyphrase BART-KPG | 35 menit |
| 06 | BM25 sebagai pembanding | 25 menit |
| 07 | Fusi hybrid: min-max dan α | 30 menit |
| 08 | Evaluasi: nDCG, MAP, Recall, P, MRR, bootstrap | 40 menit |
| 09 | Hasil dan cara membaca tabel/grafik | 30 menit |
| 10 | Tur kode berkas per berkas | 30 menit |
| 11 | Menjalankan program dan demo | 20 menit |
| 12 | Persiapan tanya jawab | 30 menit |
| — | Glosarium | rujukan |

## Siapa membaca bab mana

Semua anggota **wajib** membaca 01, 02, dan 09 (gambaran besar dan hasil), lalu bab tanggung jawabnya sendiri.
Bab 12 dibaca semua orang, terutama pertanyaan yang ditandai dengan namanya. Pembagian lengkap ada di
dokumen pembagian tugas kelompok (dibagikan terpisah oleh ketua).

| Anggota | Bab wajib |
|---|---|
| Darrell Satriano (ketua) | 01, 02, 07, 09, 10, 12 |
| Jordi Abrar Satria | 01, 02, 03, 08, 09, 12 |
| M. Rabyndra Janitra Binello | 01, 02, 04, 09, 12 |
| Leonardo Gunawan | 01, 02, 05, 09, 12 |
| Surya Andika | 01, 02, 06, 09, 11, 12 |

## Cara menjalankan perintah di modul

Semua perintah dijalankan dari folder `hybrid-ir/`. Python yang dipakai ada di folder induk (`../.venv`):

```bash
cd "Program/hybrid-ir"
../.venv/bin/python -m pytest                     # 157 tes harus lulus
../.venv/bin/hybrid-ir search --config configs/scifact_bart.yaml "betel nuts" --top-k 3
../.venv/bin/python scripts/analisis_uas.py --tanpa-waktu   # bangkitkan ulang tabel & grafik di analisis/
../.venv/bin/python scripts/modul.py --cek        # pastikan modul masih cocok dengan kode
```

Tambahkan `HF_HUB_OFFLINE=1` di depan perintah kalau tidak ada internet. Model sudah tersimpan di cache, jadi tidak
perlu diunduh ulang. Instalasi dari nol dibahas di Bab 11.

## Cara belajar yang disarankan

1. Baca bagian **Tujuan belajar** dulu, supaya tahu apa yang harus bisa kamu jelaskan.
2. Buka berkas kode yang disebut di samping modul, lalu cocokkan baris demi baris.
3. Kerjakan **Soal latihan** tanpa melihat kunci, baru cocokkan dengan **Kunci jawaban** di akhir bab.
4. Coba ubah satu hal kecil (misalnya α di perintah `search`) dan amati efeknya. Jangan menimpa folder `results/`,
   karena isinya adalah sumber angka laporan.
