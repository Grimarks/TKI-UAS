# Peta kode: cara membaca proyek ini

Tujuannya supaya kamu paham tiap bagian, bukan hanya menjalankannya. Urutan baca yang disarankan (dari paling
sederhana ke paling kompleks). Tiap berkas dimulai dengan docstring yang menjelaskan tujuan dan keputusan desainnya.

| Urutan | Berkas | Yang dipelajari |
|---|---|---|
| 1 | `text.py` | normalisasi, token, stemming, `phrase_units` (satuan pencocokan keyphrase), `text_fingerprint` |
| 2 | `config.py` | konfigurasi bertipe + validasi ketat |
| 3 | `data.py` | format BEIR, qrels, pembagian query tuning/evaluasi |
| 4 | `encoders.py` | encoder embedding; kenapa vektor dinormalkan (cosine = dot product) |
| 5 | `keyphrases.py` | generator seq2seq (BART-KPG), cache `precomputed`, `FallbackExtractor` |
| 6 | `indexes.py` | `DenseIndex` (matriks padat) dan `KeyphraseIndex` (matriks jarang, TF-IDF atas bobot keyphrase) |
| 7 | `fusion.py` | min-max, kandidat (union top-C), fusi linear `alpha`, `rank_documents` untuk 4 mode |
| 8 | `retriever.py` | merangkai semuanya: `build`, `save/load`, `search` |
| 9 | `evaluation.py`, `bm25.py` | 4 skenario, metrik, tuning alpha yang adil, paired bootstrap; BM25 sebagai pembanding |
| 10 | `cli.py`, `ui_app.py` | lapisan tipis di atas `HybridRetriever` |

## Apa yang terjadi saat `hybrid-ir search`

1. `cli._cmd_search` memuat konfigurasi lalu `HybridRetriever.load` (memeriksa encoder/pengaturan keyphrase cocok dengan indeks).
2. `search_many` memvalidasi query (kosong/terlalu panjang ditolak) lewat `validate_query`.
3. `score_matrices`:
   - semantik: `encoder.encode_queries` → `DenseIndex.scores` = `Q @ matriks.T` (cosine).
   - keyphrase: `query_extractor.extract` → `KeyphraseIndex.scores` (cosine atas vektor `bobot × IDF`).
4. `fusion.rank_documents` menyatukan (mode `hybrid`): kandidat = top-C tiap sinyal, normalisasi min-max,
   `alpha*sem + (1-alpha)*kp`. Mode `semantic`/`keyphrase`/`bm25` memperingkat satu sinyal saja; skor BM25 hanya
   dihitung di mode `bm25`.
5. `_make_hit` menyusun hasil, termasuk keyphrase yang cocok dengan query (penjelasan untuk pengguna).

## Ide latihan (mengubah perilaku sambil belajar)

1. **Ubah satuan pencocokan**: di YAML ganti `keyphrase.unit` menjadi `phrase`, `token`, `both`, bangun ulang, bandingkan
   nDCG@10. Baca `text.phrase_units` untuk melihat apa yang berubah.
2. **Tambah ekstraktor**: buat kelas dengan `extract(items, top_n)` dan `fingerprint()` seperti
   `Seq2SeqKeyphraseGenerator`, daftarkan di `build_extractor`, tambah nilai di `config.KeyphraseConfig`.
3. **Ganti fusi**: tambahkan z-score di `fusion.py` dan bandingkan dengan min-max pada `evaluate`.
4. **Tambah metrik**: tulis `success_at_k` di `evaluation.py` dan tes tangan-hitung di `tests/test_evaluation.py`.
5. **Uji ketahanan**: ubah `candidate_k` (10, 100, 1000) dan lihat pengaruhnya pada recall@100.

## Konvensi

- Skor selalu `float32`; indeks dokumen = urutan di `docs.jsonl`; pengurutan seri deterministik (indeks kecil dulu).
- Fungsi murni diutamakan supaya mudah diuji; efek samping (disk, jaringan, model) dipisahkan.
- Setiap perilaku penting punya tes di `tests/`; jalankan `pytest` sebelum dan sesudah mengubah sesuatu.
