# Bab 01 — Masalah dan konsep dasar temu kembali informasi

## Tujuan belajar

Setelah bab ini kamu bisa:

1. menjelaskan apa itu *information retrieval* (IR) dan apa keluaran sebuah sistem IR;
2. menjelaskan *vocabulary gap* dengan contoh "heart attack" vs "myocardial infarction";
3. membedakan pencarian **leksikal** (BM25) dan **semantik** (embedding), termasuk kelebihan dan kelemahannya;
4. menjelaskan kenapa proyek ini menambahkan **keyphrase** sebagai sinyal kedua.

## 1. Apa itu temu kembali informasi?

Temu kembali informasi adalah tugas mencari dokumen yang relevan dengan kebutuhan informasi pengguna dari koleksi yang
besar. Masukannya **kueri** (teks pendek), dan keluarannya **daftar dokumen terurut** (peringkat). Sistem dinilai dari
seberapa tinggi dokumen relevan muncul di daftar itu, bukan dari jawaban tunggal.

Di proyek ini dokumennya adalah artikel ilmiah (judul + abstrak) dari dua koleksi biomedis, SciFact dan NFCorpus
(Bab 03). Contoh kueri SciFact berbentuk klaim ilmiah, misalnya *"0-dimensional biomaterials show inductive
properties."*, sedangkan kueri NFCorpus berbahasa awam, misalnya *"betel nuts"*.

## 2. *Vocabulary gap*

Pengguna dan penulis dokumen sering memakai kata yang berbeda untuk hal yang sama. Orang awam mengetik
**heart attack**, sedangkan makalah kedokteran menulis **myocardial infarction**. Kalau sistem hanya mencocokkan kata,
dokumen itu tidak akan ditemukan. Inilah *vocabulary gap* yang menjadi motivasi laporan (Sharma dan Kumar [2]).

Contoh nyata dari kode proyek:

| Representasi | "heart attack" | "myocardial infarction" | Ada yang sama? |
|---|---|---|---|
| Token BM25 (setelah stopword + stemming) | `heart`, `attack` | `myocardi`, `infarct` | **tidak ada**, jadi skor BM25 = 0 |
| Embedding MiniLM (384 dimensi) | vektor $e_1$ | vektor $e_2$ | cosine $e_1 \cdot e_2$ = **0.5936** |

Sebagai pembanding, cosine "heart attack" dengan "kidney transplant" hanya **0.2623**. Jadi embedding "tahu" bahwa
heart attack lebih dekat ke myocardial infarction daripada ke transplantasi ginjal, walaupun tidak ada satu kata pun
yang sama. Angka ini dihitung oleh `scripts/modul.py --contoh` dengan encoder proyek.

## 3. Leksikal vs semantik

| | Leksikal (BM25) | Semantik (embedding) |
|---|---|---|
| Cara kerja | hitung kata yang sama, beri bobot kata langka (Bab 06) | ubah teks jadi vektor, ukur kemiripan vektor (Bab 04) |
| Kelebihan | cepat, tanpa pelatihan, kuat untuk istilah langka atau nama spesifik | menangkap sinonim dan parafrase |
| Kelemahan | buta sinonim (*vocabulary gap*) | bisa "kabur" untuk istilah spesifik; model *zero-shot* sering kalah dari BM25 (Ren dkk. [5], Chen dkk. [6]) |

Kelemahan semantik itu terlihat nyata di proyek ini. Pada kueri SciFact 1024 tentang *CTCF anchor sites*, BM25
menaruh dokumen relevan di peringkat 1, sedangkan embedding menaruhnya di peringkat 41 (Bab 09, studi kasus).

## 4. Ide proyek: tambahkan keyphrase

**Keyphrase** adalah frasa kunci yang merangkum isi dokumen, misalnya "kidney paired donation". Proyek ini membangkitkan
keyphrase dengan model **BART-KPG**, lalu mencocokkan keyphrase kueri dengan keyphrase dokumen (Bab 05). Skornya
digabung dengan skor embedding:

$$s_{hyb} = \alpha \cdot \tilde{s}_{sem} + (1-\alpha) \cdot \tilde{s}_{kp}$$

Ide besarnya: embedding menangani makna umum, keyphrase menegaskan istilah inti. BM25 **tidak** ikut digabung. Ia
hanya menjadi **pembanding** yang kuat untuk mengukur seberapa baik sistem kita (Bab 06).

Tiga pertanyaan penelitian di laporan:

1. Apakah menambah keyphrase meningkatkan kualitas peringkat dibanding tiap jalur sendiri-sendiri?
2. Bagaimana pengaruh bobot fusi α?
3. Seberapa dekat hybrid dengan BM25 sebagai pembanding leksikal yang kuat?

Jawabannya ada di Bab 09: (1) ya, kecil tetapi signifikan; (2) α terbaik 0.9; (3) BM25 sedikit lebih tinggi, tetapi
selisihnya tidak signifikan.

## Ringkasan

- IR menghasilkan **peringkat** dokumen untuk sebuah kueri.
- *Vocabulary gap*: kata berbeda untuk makna sama. BM25 gagal di sini (skor 0 untuk heart attack vs myocardial
  infarction), embedding berhasil (cosine 0.5936).
- Embedding lemah pada istilah langka. Itulah alasan menambah keyphrase.
- Sistem final menggabungkan **semantik + keyphrase**; BM25 hanya pembanding.

## Soal latihan

**Soal 1.** Apa keluaran sebuah sistem IR, dan kenapa evaluasinya memakai peringkat, bukan benar/salah?

**Soal 2.** Jelaskan dengan contoh dari proyek kenapa BM25 gagal mencocokkan "heart attack" dengan "myocardial
infarction".

**Soal 3.** Sebutkan satu kelemahan pencarian semantik dan satu contoh kasus nyata dari proyek ini.

**Soal 4.** Kenapa BM25 tidak ikut digabung ke skor hybrid, padahal BM25 kuat?

## Kunci jawaban

**Jawaban 1.** Keluarannya daftar dokumen terurut. Relevansi bersifat bertingkat dan pengguna membaca dari atas, jadi
yang penting adalah **posisi** dokumen relevan. Karena itu dipakai metrik peringkat seperti nDCG@10.

**Jawaban 2.** BM25 hanya menghitung token yang sama. Setelah stopword dibuang dan di-stem, "heart attack" menjadi
`heart`, `attack`, sedangkan "myocardial infarction" menjadi `myocardi`, `infarct`. Tidak ada token yang sama, sehingga
skornya 0.

**Jawaban 3.** Embedding bisa kurang tajam untuk istilah spesifik atau langka. Contohnya kueri SciFact 1024 (*CTCF
anchor sites*): BM25 menaruh dokumen relevan di peringkat 1, embedding di peringkat 41.

**Jawaban 4.** Desain penelitian ini sengaja menguji sinyal **keyphrase** sebagai pelengkap embedding. BM25 dipakai
sebagai pembanding independen yang kuat, supaya kita bisa menilai apakah hybrid semantik + keyphrase mendekati kualitas
metode leksikal klasik. Kalau BM25 ikut digabung, pertanyaan itu tidak bisa dijawab dengan bersih.
