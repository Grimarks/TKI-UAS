# Glosarium

| Istilah (Indonesia) | Istilah (Inggris) | Arti singkat |
|---|---|---|
| temu kembali informasi | information retrieval (IR) | mencari dokumen relevan untuk sebuah kueri dari koleksi besar |
| kueri | query | teks kebutuhan informasi yang diketik pengguna |
| korpus | corpus | kumpulan dokumen yang dicari |
| penilaian relevansi | relevance judgments (qrels) | daftar pasangan kueri–dokumen beserta tingkat relevansinya |
| kesenjangan kosakata | vocabulary gap | kata kueri dan dokumen berbeda padahal maknanya sama |
| pencarian leksikal | lexical / sparse retrieval | pencocokan kata yang sama persis, mis. BM25 |
| pencarian semantik | semantic / dense retrieval | pencocokan makna lewat vektor embedding |
| embedding | embedding | vektor angka yang mewakili makna teks |
| bi-encoder | bi-encoder | kueri dan dokumen dienkode terpisah oleh model yang sama |
| normalisasi L2 | L2 normalization | membagi vektor dengan panjangnya sehingga panjangnya 1 |
| kemiripan kosinus | cosine similarity | ukuran kemiripan arah dua vektor |
| frasa kunci | keyphrase | frasa singkat yang merangkum isi dokumen |
| pembangkitan keyphrase | keyphrase generation (KPG) | membuat keyphrase dengan model seq2seq, mis. BART-KPG |
| ekstraksi keyphrase | keyphrase extraction | memilih keyphrase dari kata yang ada di teks |
| keyphrase hadir / tidak hadir | present / absent keyphrase | keyphrase yang muncul / tidak muncul di teks |
| unit pencocokan | matching unit | frasa ternormalisasi atau kata bermakna dari sebuah keyphrase |
| pemotongan imbuhan | stemming | mengubah kata ke bentuk dasar kasar ("networks" → "network") |
| kata henti | stopword | kata umum yang dibuang, mis. "the", "of" |
| frekuensi dokumen invers | inverse document frequency (IDF) | bobot tinggi untuk unit/kata yang jarang |
| BM25 | Best Matching 25 | fungsi skor leksikal klasik; di proyek ini hanya pembanding |
| fusi skor | score fusion | menggabungkan skor beberapa jalur menjadi satu |
| normalisasi min-max | min-max normalization | memetakan skor ke rentang [0, 1] |
| kombinasi linear / konveks | linear / convex combination | α·a + (1−α)·b dengan 0 ≤ α ≤ 1 |
| fusi peringkat resiprokal | reciprocal rank fusion (RRF) | fusi berbasis peringkat; tidak dipakai di proyek ini |
| himpunan kandidat | candidate set | dokumen yang ikut difusikan (top-C tiap jalur) |
| penalaan | tuning | memilih nilai parameter (α) pada data penalaan |
| data penalaan | tune split (train/dev) | kueri untuk memilih α; tidak dipakai untuk pelaporan |
| data uji | test split | kueri untuk melaporkan hasil akhir |
| zero-shot | zero-shot | model dipakai tanpa dilatih ulang pada koleksi uji |
| nDCG@10 | normalized discounted cumulative gain | metrik peringkat utama: posisi + relevansi bertingkat |
| MAP@100 | mean average precision | rata-rata presisi di posisi dokumen relevan, dirata-ratakan antar-kueri |
| Recall@k | recall | proporsi dokumen relevan yang masuk k teratas |
| P@10 | precision at 10 | proporsi dokumen relevan di 10 teratas |
| MRR@10 | mean reciprocal rank | 1 / posisi dokumen relevan pertama, dirata-ratakan |
| bootstrap berpasangan | paired bootstrap | uji signifikansi dengan sampel ulang selisih per kueri |
| selang kepercayaan | confidence interval (CI) | rentang nilai yang mungkin untuk selisih rata-rata |
| nilai p | p-value | peluang melihat selisih sebesar itu bila sebenarnya tidak ada beda |
| signifikan | statistically significant | p < 0.05; selisihnya kecil kemungkinan kebetulan |
| indeks | index | struktur data yang disiapkan saat `build` agar pencarian cepat |
| matriks jarang | sparse matrix | matriks yang sebagian besar isinya nol, disimpan efisien |
| antarmuka | user interface (UI) | aplikasi Streamlit untuk mencoba pencarian |
