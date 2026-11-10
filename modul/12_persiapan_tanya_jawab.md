# Bab 12 — Persiapan tanya jawab

Berikut 30 pertanyaan yang mungkin diajukan dosen, dikelompokkan menurut anggota yang menjadi penjawab utama. Semua
anggota sebaiknya membaca semuanya. Setiap jawaban hanya memakai angka yang ada di `analisis/` dan `results/`. Kalau
ditanya sesuatu yang tidak diukur, jawab jujur bahwa hal itu tidak diukur.

Tips umum:

- Jawab dengan pola **klaim → angka → alasan**. Contoh: "Hybrid lebih baik dari semantic (0.6671 vs 0.6497, p = 0.006)
  karena keyphrase menegaskan istilah inti."
- Bedakan "tidak signifikan" (data tidak cukup untuk menyimpulkan beda) dari "setara" (tidak kita uji).
- Kalau pertanyaannya di luar bagianmu, oper ke anggota yang bertanggung jawab.

## Darrell Satriano: arsitektur, fusi, kesimpulan

**P1. Apa kontribusi utama penelitian ini?**
Pipeline yang menggabungkan embedding (all-MiniLM-L6-v2) dan keyphrase hasil pembangkitan (BART-KPG) dengan fusi linear,
implementasi lengkap dari indeks sampai UI, dan evaluasi empat skenario pada dua koleksi BEIR dengan uji signifikansi.
Hasilnya, hybrid unggul signifikan atas semantic (+0.0174 dan +0.0046) dan tidak berbeda signifikan dari BM25.

**P2. Kenapa BM25 tidak ikut digabung, padahal BM25 paling tinggi?**
Pertanyaan penelitian kami adalah apakah keyphrase bisa melengkapi embedding. BM25 dijadikan pembanding independen
supaya kami bisa mengukur seberapa dekat hybrid semantik + keyphrase dengan metode leksikal klasik. Menggabungkan BM25
adalah pengembangan yang menarik, tetapi itu pertanyaan yang berbeda.

**P3. Kenapa α = 0.9? Bukankah itu berarti keyphrase hampir tidak dipakai?**
α dipilih dari data penalaan (nDCG@10 tertinggi: 0.6699 SciFact, 0.3076 NFCorpus). Bobot 0.1 kecil, tetapi berpengaruh
karena skor semantik kandidat teratas sering sangat berdekatan setelah min-max. Contohnya kueri 1185: semantik
ternormalisasi 0.9855 vs 1.0, lalu keyphrase (0.9988 vs 0.1427) membalik urutan, dan dokumen relevan naik dari peringkat
3 ke 1. Tanpa keyphrase (α = 1) nDCG@10 penalaan turun menjadi 0.6593 dan 0.3028.

**P4. Kenapa kombinasi linear, bukan RRF?**
Bruch dkk. [18] menunjukkan bahwa kombinasi konveks atas skor ternormalisasi mengungguli RRF dan hanya butuh sedikit data
untuk menala parameternya; Sun dkk. [19] mendukung hal serupa. Kombinasi linear juga lebih mudah dijelaskan: satu
parameter α dengan arti yang jelas.

**P5. Apakah α dipilih dengan melihat data test?**
Tidak. α dipilih di kueri penalaan (SciFact train, NFCorpus dev). Sapuan α di test baru dihitung setelahnya dan hanya
untuk analisis. Kebetulan α terbaik di test juga 0.9, yang menunjukkan bahwa pilihan α berpindah dengan baik ke data uji.

**P6. Apa keterbatasan terbesar penelitian ini?**
*Zero-shot* (tidak ada pelatihan ulang), satu encoder kecil, dua koleksi biomedis berbahasa Inggris, dan keyphrase dari
model yang dilatih di domain ilmu komputer dengan kualitas yang tidak diukur langsung. Kenaikan atas semantic juga kecil.

## Jordi Abrar Satria: data dan evaluasi

**P7. Kenapa metrik utamanya nDCG@10?**
nDCG memperhitungkan posisi dan relevansi bertingkat (NFCorpus punya tingkat 1 dan 2), dan @10 mewakili halaman pertama
hasil yang benar-benar dilihat pengguna. Metrik ini juga dipakai di penelitian acuan Sarwar dkk. [7].

**P8. Hitung nDCG@10 untuk kueri dengan satu dokumen relevan di peringkat 3.**
DCG = 1/log₂(4) = 0.5, IDCG = 1/log₂(2) = 1, sehingga nDCG@10 = 0.5. Ini persis nDCG semantic pada kueri SciFact 1185.

**P9. Kenapa nDCG NFCorpus jauh lebih rendah?**
Rata-rata ada 38.2 dokumen relevan per kueri test NFCorpus (SciFact 1.1), sehingga urutan ideal sangat sulit didekati.
Kuerinya juga pendek dan berbahasa awam. Angka antar-koleksi tidak dibandingkan langsung; yang dibandingkan adalah
skenario di dalam satu koleksi.

**P10. Bagaimana paired bootstrap bekerja?**
Hitung selisih nDCG@10 per kueri, ambil sampel ulang kueri dengan pengembalian 10.000 kali, lalu hitung rata-rata selisih
tiap sampel. CI 95% = persentil 2.5–97.5, dan p dua sisi = 2 × min(proporsi ≤ 0, proporsi ≥ 0). Seed tetap 13.

**P11. Hybrid vs BM25 p = 0.354. Artinya hybrid sama bagusnya dengan BM25?**
Tidak persis. Artinya selisihnya (−0.0163) tidak signifikan: CI 95% [−0.0507, +0.0173] mencakup nol. Kami tidak menguji
kesetaraan, jadi tidak boleh bilang "setara". Yang benar: tidak berbeda signifikan. Per kueri pun imbang (65 menang, 64
kalah).

**P12. Kenapa tidak memakai uji t berpasangan?**
Paired bootstrap tidak mengasumsikan distribusi normal. Selisih nDCG per kueri jelas tidak normal: banyak nol (seri) dan
nilai ekstrem ±1. Bootstrap juga langsung memberi CI.

## M. Rabyndra Janitra Binello: jalur semantik

**P13. Kenapa memilih all-MiniLM-L6-v2?**
Ringan (6 lapis, 384 dimensi) dan bisa dijalankan di laptop: pencarian semantik sekitar 4 ms per kueri. Pilihan ini
sesuai cakupan tugas. Encoder yang lebih besar atau khusus biomedis menjadi rencana pengembangan.

**P14. Kenapa vektor dinormalisasi L2?**
Supaya cosine sama dengan perkalian dalam, sehingga skor semua dokumen bisa dihitung dengan satu perkalian matriks (Q ×
384)·(384 × N).

**P15. Apa beda bi-encoder dan cross-encoder?**
Bi-encoder mengenkode kueri dan dokumen terpisah, jadi dokumen cukup dienkode sekali. Cross-encoder mengenkode pasangan
kueri–dokumen bersama-sama: lebih akurat tetapi jauh lebih mahal, karena setiap dokumen harus diproses ulang untuk setiap
kueri. Proyek ini memakai bi-encoder.

**P16. Kenapa semantic-only kalah dari BM25?**
Pada pengujian *zero-shot*, *dense retriever* sering belum mampu mengungguli BM25 (Ren dkk. [5]) dan lemah pada istilah
langka (Chen dkk. [6]). Contoh nyata: kueri SciFact 1024 tentang CTCF, dengan BM25 di peringkat 1 dan semantic di
peringkat 41.

**P17. Apakah pencarian vektornya pakai ANN (FAISS, dan sebagainya)?**
Tidak. Pencarian eksak dengan perkalian matriks penuh. Untuk 5.183 dan 3.633 dokumen ini cepat dan bebas galat
pendekatan.

**P18. Apa yang terjadi pada abstrak yang lebih panjang dari 256 token?**
Encoder memotong masukan di 256 token (`max_seq_length`), jadi bagian akhir abstrak yang sangat panjang tidak ikut
dienkode. Kebanyakan abstrak cukup pendek; dampaknya terhadap hasil tidak kami ukur terpisah.

## Leonardo Gunawan: jalur keyphrase

**P19. Kenapa keyphrase dibangkitkan, bukan diekstraksi?**
Model pembangkit secara prinsip bisa menghasilkan keyphrase yang tidak tertulis di teks (Kulkarni dkk. [10], Boudin dkk.
[11]). Tetapi kami jujur: di data kami hanya 5.4% (SciFact) dan 4.5% (NFCorpus) keyphrase dokumen yang *absent*, jadi
model ini hampir selalu menyalin frasa dari teks.

**P20. Apakah keyphrase absent membantu mengatasi vocabulary gap?**
Belum bisa disimpulkan. Kami mengukur proporsinya (kueri: 14.7% dan 28.5%), tetapi tidak mengukur dampaknya terhadap
kualitas peringkat. Itu masuk rencana pengembangan.

**P21. Modelnya dilatih di ilmu komputer (KP20k), datanya biomedis. Apakah itu masalah?**
Boudin dkk. [11] mencatat perpindahan domain sebagai kesulitan. Kami tidak punya keyphrase acuan untuk mengukur kualitas
secara langsung. Bukti tidak langsungnya: rata-rata hanya 4.44 dan 4.33 keyphrase per dokumen (batas 10), dan ada kasus
seperti "kohlrabi" ketika istilah penting tidak terangkut.

**P22. Kenapa pembangkitan dilakukan di Kaggle?**
BART-KPG berat untuk sekitar 6.300–6.800 teks per koleksi. Di 2× T4, setiap proses selesai dalam 91–132 detik.
Hasilnya disimpan ke JSONL dengan kunci SHA-1 teks, lalu dibaca lokal. Kueri bebas di UI tetap dibangkitkan lokal oleh
model yang sama (sekitar 83–116 ms per kueri).

**P23. Kenapa unit pencocokan frasa dan kata sekaligus?**
Frasa utuh memberi kecocokan yang tepat ("kidney paired donation"); kata memberi kecocokan sebagian ("betel" dan "nuts"
dengan "betel-nut chewing"). Bobot dikalikan IDF supaya unit langka lebih berarti: frasa "kidney paired donation" punya
idf 8.1670, kata "cell" hanya 2.6843.

**P24. NFCorpus: hanya 14.0% pasangan relevan yang punya skor keyphrase. Bukankah keyphrase gagal?**
Angka per pasangan itu dipengaruhi qrels yang padat (rata-rata 38.2 dokumen relevan per kueri). Per kueri, 71.2% kueri
NFCorpus punya minimal satu dokumen relevan yang mendapat skor keyphrase. Jadi itu bukan kegagalan keyphrase semata.
Tetapi memang benar sinyalnya jarang: 9.9% kueri NFCorpus tidak mendapat skor keyphrase sama sekali.

## Surya Andika: BM25, UI, studi kasus

**P25. Jelaskan rumus BM25 dan parameternya.**
Jumlah atas term kueri dari idf × frekuensi tersaturasi yang dinormalkan panjang dokumen. $k_1 = 1.2$ mengatur saturasi
frekuensi, $b = 0.75$ mengatur normalisasi panjang. Contoh: korpus tiga dokumen dengan kueri "heart attack" memberi skor
0.8843 untuk dua dokumen yang memuat kedua kata, sama dengan hasil kode.

**P26. Apakah BM25 kalian benar? Kenapa tidak memakai pustaka seperti Pyserini?**
Implementasinya kecil dan transparan (`bm25.py`), memakai tokenisasi dan stemming yang sama dengan jalur keyphrase supaya
adil, dan bisa dicek dengan hitungan tangan (Bab 06). Nilai nDCG@10-nya (0.6834 SciFact) wajar untuk BM25. Kami tidak
membandingkannya dengan pustaka lain, jadi itu tidak kami klaim.

**P27. Ceritakan satu kasus ketika hybrid menolong.**
NFCorpus "betel nuts": semantic menaruh artikel tentang kacang secara umum di peringkat 1. Keyphrase dokumen relevan
"betel-nut chewing" memuat kata "betel" dan "nuts", sehingga dokumen relevan naik dari peringkat 2 ke 1.

**P28. Ceritakan satu kasus ketika hybrid kalah.**
SciFact 1024 (CTCF anchor sites): BM25 menemukan dokumen relevan di peringkat 1 berkat kata langka "CTCF" dan "anchor".
Semantic menaruhnya di peringkat 41 dan hybrid di 49, karena keyphrase dokumen tidak berbagi kata dengan keyphrase kueri.

**P29. Bagaimana keamanan antarmuka?**
Streamlit hanya mendengarkan 127.0.0.1, tidak ada unggah berkas atau input path, teks dokumen di-escape (tanpa
`unsafe_allow_html`), panjang kueri dibatasi, dan pemakaian model diserialisasi dengan lock supaya tidak crash di GPU
Apple.

**P30. Bagaimana kasus studi dipilih? Apakah dipilih yang bagus-bagus saja?**
Dipilih otomatis oleh `scripts/analisis_uas.py`: dua kenaikan terbesar dari semantic ke hybrid, dan dua penurunan
terbesar dari BM25 ke hybrid, satu dari tiap koleksi. Jadi kasus gagal sengaja ikut ditampilkan, dan kandidat lainnya ada
di `analisis/studi_kasus.md`.
