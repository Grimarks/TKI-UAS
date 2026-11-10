# Peta kode: kotak di Gambar 1 → berkas → fungsi → baris

Dokumen ini dipakai saat presentasi untuk menunjuk kode yang menjalankan setiap kotak di diagram pipeline
(Gambar 1 laporan). Setiap modul di `src/hybrid_ir/` juga dibuka dengan docstring yang menyebut kotaknya,
masukan/keluarannya, dan nomor persamaan yang diimplementasikan.

> **Tabel di bawah dibangkitkan otomatis** dari kode oleh `scripts/peta_kode.py`, jadi nomor barisnya selalu cocok.
> Setelah mengubah kode, jalankan `../.venv/bin/python scripts/peta_kode.py` (atau `--cek` untuk memeriksa saja).

Nama kotak mengikuti Gambar 1: dua jalur utama, **A (semantik)** dan **B (keyphrase)**, yang difusikan, ditambah
**BM25 sebagai pembanding** yang diperingkat sendiri dan **tidak** ikut difusikan.

<!-- PETA:MULAI -->
### Fase 1 — Indexing dokumen (offline), `hybrid-ir build`

| Kotak di Gambar 1 | Berkas | Fungsi | Baris |
|---|---|---|---|
| Korpus dokumen | `src/hybrid_ir/data.py` | `load_beir_dir` | 134–152 |
|  | `src/hybrid_ir/data.py` | `load_corpus` | 79–99 |
|  | `src/hybrid_ir/data.py` | `Document.full_text` | 40–43 |
| Pra-pemrosesan | `src/hybrid_ir/text.py` | `normalize_text` | 47–53 |
|  | `src/hybrid_ir/text.py` | `make_full_text` | 56–67 |
| A. Encoder SBERT | `src/hybrid_ir/encoders.py` | `SentenceTransformerEncoder.encode_documents` | 145–147 |
|  | `src/hybrid_ir/encoders.py` | `SentenceTransformerEncoder._encode` | 132–143 |
|  | `src/hybrid_ir/encoders.py` | `_l2_normalize` | 55–59 |
| Vector index | `src/hybrid_ir/indexes.py` | `DenseIndex.__init__` | 41–46 |
|  | `src/hybrid_ir/indexes.py` | `DenseIndex.save` | 60–63 |
| B. Keyphrase + bobot (BART-KPG) — di Kaggle | `kaggle/generate_keyphrases.py` | `load_generator` | 168–196 |
|  | `kaggle/generate_keyphrases.py` | `run_worker` | 129–165 |
|  | `kaggle/generate_keyphrases.py` | `parse_generated_keyphrases` | 60–73 |
|  | `kaggle/generate_keyphrases.py` | `rank_weights` | 56–57 |
| B. Keyphrase + bobot (BART-KPG) — dibaca lokal | `src/hybrid_ir/keyphrases.py` | `PrecomputedKeyphrases.extract` | 210–227 |
|  | `src/hybrid_ir/keyphrases.py` | `Seq2SeqKeyphraseGenerator.extract` | 138–154 |
|  | `src/hybrid_ir/keyphrases.py` | `rank_weights` | 76–78 |
| Keyphrase index | `src/hybrid_ir/text.py` | `phrase_units` | 136–153 |
|  | `src/hybrid_ir/text.py` | `normalize_phrase` | 122–133 |
|  | `src/hybrid_ir/indexes.py` | `unit_weights` | 74–85 |
|  | `src/hybrid_ir/indexes.py` | `keyphrase_idf` | 88–90 |
|  | `src/hybrid_ir/indexes.py` | `KeyphraseIndex.build` | 112–137 |
| Pembanding: Token + stemming → BM25 index | `src/hybrid_ir/text.py` | `content_tokens` | 107–118 |
|  | `src/hybrid_ir/bm25.py` | `bm25_idf` | 29–31 |
|  | `src/hybrid_ir/bm25.py` | `BM25.fit` | 44–66 |
| (perakit Fase 1) | `src/hybrid_ir/retriever.py` | `HybridRetriever.build` | 126–147 |
|  | `src/hybrid_ir/retriever.py` | `HybridRetriever.save` | 169–185 |
| (perintah) | `src/hybrid_ir/cli.py` | `_cmd_build` | 88–95 |

### Fase 2 — Query & retrieval (online), `hybrid-ir search` / `hybrid-ir ui`

| Kotak di Gambar 1 | Berkas | Fungsi | Baris |
|---|---|---|---|
| Query pengguna | `src/hybrid_ir/cli.py` | `_cmd_search` | 98–108 |
|  | `src/hybrid_ir/ui_app.py` | `main` | 76–144 |
| Pra-pemrosesan | `src/hybrid_ir/retriever.py` | `HybridRetriever.validate_query` | 221–228 |
| A. Encode query | `src/hybrid_ir/encoders.py` | `SentenceTransformerEncoder.encode_queries` | 149–151 |
| Skor semantik | `src/hybrid_ir/indexes.py` | `DenseIndex.scores` | 53–58 |
| B. Keyphrase query | `src/hybrid_ir/keyphrases.py` | `build_extractor` | 269–279 |
|  | `src/hybrid_ir/keyphrases.py` | `FallbackExtractor.extract` | 247–262 |
| Skor keyphrase | `src/hybrid_ir/indexes.py` | `KeyphraseIndex.query_units` | 140–142 |
|  | `src/hybrid_ir/indexes.py` | `KeyphraseIndex.scores` | 144–161 |
|  | `src/hybrid_ir/indexes.py` | `KeyphraseIndex._cosine_row` | 163–172 |
| Pembanding: Token query → Skor BM25 | `src/hybrid_ir/retriever.py` | `HybridRetriever.bm25` | 253–258 |
|  | `src/hybrid_ir/retriever.py` | `HybridRetriever.bm25_scores` | 260–262 |
|  | `src/hybrid_ir/bm25.py` | `BM25.scores` | 68–80 |
| (perakit Fase 2) | `src/hybrid_ir/retriever.py` | `HybridRetriever.search_many` | 265–276 |
|  | `src/hybrid_ir/retriever.py` | `HybridRetriever.score_matrices` | 230–251 |
|  | `src/hybrid_ir/retriever.py` | `HybridRetriever._scores_for_mode` | 282–288 |
| Normalisasi + Fusi | `src/hybrid_ir/fusion.py` | `candidate_set` | 56–59 |
|  | `src/hybrid_ir/fusion.py` | `minmax` | 46–53 |
|  | `src/hybrid_ir/fusion.py` | `fuse_weighted` | 62–67 |
| Ranking top-k | `src/hybrid_ir/fusion.py` | `rank_documents` | 77–98 |
|  | `src/hybrid_ir/fusion.py` | `topk_indices` | 25–33 |
|  | `src/hybrid_ir/fusion.py` | `positive_topk` | 36–42 |
|  | `src/hybrid_ir/retriever.py` | `HybridRetriever._hits_for_query` | 290–298 |
|  | `src/hybrid_ir/retriever.py` | `HybridRetriever._make_hit` | 300–308 |
| Ranking top-k (tampilan) | `src/hybrid_ir/cli.py` | `_print_result` | 69–78 |
|  | `src/hybrid_ir/ui_app.py` | `render_hit` | 64–73 |

### Fase 3 — Evaluasi, `hybrid-ir evaluate`

| Kotak di Gambar 1 | Berkas | Fungsi | Baris |
|---|---|---|---|
| Data uji BEIR | `src/hybrid_ir/data.py` | `usable_queries` | 155–162 |
|  | `src/hybrid_ir/data.py` | `load_qrels_file` | 116–131 |
|  | `src/hybrid_ir/evaluation.py` | `_split_queries` | 187–203 |
| Skenario | `src/hybrid_ir/evaluation.py` | `ScoreTable.compute` | 160–168 |
|  | `src/hybrid_ir/evaluation.py` | `ScoreTable.single_signal_rankers` | 174–180 |
|  | `src/hybrid_ir/evaluation.py` | `ScoreTable.hybrid_ranker` | 182–184 |
|  | `src/hybrid_ir/evaluation.py` | `_evaluate_scenarios` | 227–239 |
| Tuning α | `src/hybrid_ir/evaluation.py` | `_alpha_sweep` | 211–218 |
|  | `src/hybrid_ir/evaluation.py` | `_select_alpha` | 221–224 |
| Metrik | `src/hybrid_ir/evaluation.py` | `_dcg` | 46–48 |
|  | `src/hybrid_ir/evaluation.py` | `ndcg_at_k` | 51–56 |
|  | `src/hybrid_ir/evaluation.py` | `average_precision` | 70–80 |
|  | `src/hybrid_ir/evaluation.py` | `recall_at_k` | 64–67 |
|  | `src/hybrid_ir/evaluation.py` | `precision_at_k` | 59–61 |
|  | `src/hybrid_ir/evaluation.py` | `mrr_at_k` | 83–88 |
|  | `src/hybrid_ir/evaluation.py` | `evaluate_run` | 103–111 |
| Analisis | `src/hybrid_ir/evaluation.py` | `paired_bootstrap` | 115–134 |
|  | `src/hybrid_ir/evaluation.py` | `_significance_tests` | 242–250 |
|  | `src/hybrid_ir/evaluation.py` | `write_outputs` | 337–344 |
| (perakit Fase 3) | `src/hybrid_ir/evaluation.py` | `run_experiment` | 253–282 |
| (perintah) | `src/hybrid_ir/cli.py` | `_cmd_evaluate` | 111–122 |

### Persamaan di laporan → kode

| Persamaan | Isi | Berkas | Fungsi | Baris |
|---|---|---|---|---|
| (1) | skor semantik, cosine = perkalian dalam | `src/hybrid_ir/indexes.py` | `DenseIndex.scores` | 53–58 |
| (2) | idf unit keyphrase | `src/hybrid_ir/indexes.py` | `keyphrase_idf` | 88–90 |
| (3) | bobot unit w = b·idf | `src/hybrid_ir/indexes.py` | `unit_weights` | 74–85 |
|  |  | `src/hybrid_ir/indexes.py` | `KeyphraseIndex.build` | 112–137 |
| (4) | skor keyphrase, cosine | `src/hybrid_ir/indexes.py` | `KeyphraseIndex._cosine_row` | 163–172 |
| (5) | BM25 (pembanding) | `src/hybrid_ir/bm25.py` | `bm25_idf` | 29–31 |
|  |  | `src/hybrid_ir/bm25.py` | `BM25.fit` | 44–66 |
|  |  | `src/hybrid_ir/bm25.py` | `BM25.scores` | 68–80 |
| (6) | normalisasi min-max | `src/hybrid_ir/fusion.py` | `minmax` | 46–53 |
| (7) | fusi linear α | `src/hybrid_ir/fusion.py` | `fuse_weighted` | 62–67 |
| (8) | DCG@k | `src/hybrid_ir/evaluation.py` | `_dcg` | 46–48 |
| (9) | nDCG@k | `src/hybrid_ir/evaluation.py` | `ndcg_at_k` | 51–56 |
<!-- PETA:SELESAI -->

## Urutan pemanggilan saat `hybrid-ir search` (mode hybrid)

```
cli._cmd_search
├─ config.load_config                                  baca YAML (alpha = 0.9, C = 100, top_k = 10)
├─ HybridRetriever.load                                muat indeks, cek encoder & pengaturan keyphrase cocok
└─ HybridRetriever.search → search_many
   ├─ _check_search_args                               top_k, mode, alpha dalam rentang
   ├─ _scores_for_mode
   │  └─ score_matrices
   │     ├─ validate_query                             Pra-pemrosesan: normalize_text, cek panjang
   │     ├─ encoder.encode_queries → DenseIndex.scores                 Jalur A: persamaan (1)
   │     └─ query_extractor.extract → KeyphraseIndex.scores            Jalur B: unit_weights → (4)
   └─ _hits_for_query
      ├─ fusion.rank_documents → fuse_weighted
      │  ├─ candidate_set                              R = top-C(sem) ∪ top-C(kp > 0)
      │  └─ minmax (6), lalu α·sem + (1−α)·kp (7)
      └─ _make_hit                                     judul, cuplikan, keyphrase yang cocok dengan query
```

Pada mode `bm25`, `_scores_for_mode` memanggil `bm25_scores` (persamaan (5)) dan `rank_documents` hanya
memperingkat skor BM25. Jalur A dan B tidak dihitung, dan BM25 tidak pernah dicampur dengan skor hybrid.

## Urutan pemanggilan saat `hybrid-ir evaluate`

```
cli._cmd_evaluate → evaluation.run_experiment
├─ _split_queries          tune = SciFact train / NFCorpus dev; eval = test
├─ ScoreTable.compute      semua query di-encode/di-ekstrak SEKALI (sem, kp, bm25: Q × N)
├─ _alpha_sweep (tune)     α = 0, 0.1, …, 1 → nDCG@10 pada query tune
├─ _select_alpha           α terbaik (seri → paling dekat 0.5)  → 0.9 di kedua dataset
├─ _evaluate_scenarios     bm25 | semantic | keyphrase | hybrid_weighted pada query test, enam metrik
├─ _alpha_sweep (eval)     hanya untuk analisis (tidak dipakai memilih α)
├─ _significance_tests     paired bootstrap 10.000×, hybrid vs semantic / keyphrase / bm25
└─ write_outputs           results/<dataset>/metrics.json, metrics.md, per_query_ndcg10.csv
```

## Ide latihan (mengubah perilaku sambil belajar)

1. **Ubah satuan pencocokan**: di YAML ganti `keyphrase.unit` menjadi `phrase` atau `token`, bangun ulang ke
   `index.dir` lain, lalu bandingkan nDCG@10. Baca `text.phrase_units` untuk melihat apa yang berubah.
2. **Cosine vs Jaccard**: ganti `keyphrase.similarity: jaccard`; lihat `KeyphraseIndex._jaccard_row`.
3. **Ganti normalisasi**: tambahkan normalisasi z-score di `fusion.py` dan bandingkan dengan min-max.
4. **Tambah metrik**: tulis `success_at_k` di `evaluation.py` beserta tes hitung tangan di `tests/test_evaluation.py`.
5. **Uji ketahanan**: ubah `retrieval.candidate_k` (10, 100, 1000) dan lihat pengaruhnya pada recall@100.

Untuk latihan, simpan hasil ke folder lain (`--out results/latihan_...`). Jangan menimpa `results/scifact_bart` dan
`results/nfcorpus_bart`, karena keduanya adalah sumber angka laporan.

## Konvensi

- Nama sinyal seragam di seluruh kode: `sem` (Jalur A), `kp` (Jalur B), `bm25` (pembanding), dan `alpha` (α).
- Skor selalu `float32`; indeks dokumen = urutan di `docs.jsonl`; pengurutan seri deterministik (indeks kecil dulu).
- Fungsi murni diutamakan supaya mudah diuji; efek samping (disk, jaringan, model) dipisahkan.
- Setiap perilaku penting punya tes di `tests/`; jalankan `../.venv/bin/python -m pytest` sebelum dan sesudah
  mengubah sesuatu.
