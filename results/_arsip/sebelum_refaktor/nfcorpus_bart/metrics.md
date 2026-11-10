# Hasil evaluasi: nfcorpus

- Dokumen: 3633; query tuning: 324 (split:dev); query evaluasi: 323 (split `test`)
- alpha dipilih pada query tuning: **0.9**
- bobot tiga sinyal dipilih pada query tuning: **sem=0.4, kp=0.2, bm25=0.4**

| Skenario | ndcg@10 | map@100 | recall@10 | recall@100 | precision@10 | mrr@10 |
|---|---|---|---|---|---|---|
| BM25 (leksikal) | 0.3278 | 0.1474 | 0.1520 | 0.2586 | 0.2387 | 0.5314 |
| Semantic-only | 0.3174 | 0.1431 | 0.1569 | 0.3111 | 0.2437 | 0.5083 |
| Keyphrase-only | 0.2321 | 0.0857 | 0.1004 | 0.1705 | 0.1724 | 0.4200 |
| Hybrid weighted (alpha=0.9) | 0.3219 | 0.1466 | 0.1575 | 0.3137 | 0.2430 | 0.5225 |
| Hybrid RRF | 0.3019 | 0.1351 | 0.1456 | 0.3099 | 0.2245 | 0.4990 |
| Hybrid+BM25 weighted (sem=0.4, kp=0.2, bm25=0.4) | 0.3527 | 0.1657 | 0.1710 | 0.3215 | 0.2607 | 0.5546 |
| Hybrid+BM25 RRF | 0.3346 | 0.1547 | 0.1635 | 0.3220 | 0.2455 | 0.5449 |

## Uji signifikansi (paired bootstrap, nDCG@10, model dikurangi pembanding)

| Model | Pembanding | selisih rata-rata | CI 95% | p |
|---|---|---|---|---|
| hybrid_weighted | semantic | +0.0046 | [+0.0003, +0.0092] | 0.0350 |
| hybrid_weighted | keyphrase | +0.0899 | [+0.0680, +0.1121] | 0.0001 |
| hybrid_weighted | bm25 | -0.0058 | [-0.0280, +0.0154] | 0.5902 |
| hybrid_bm25_weighted | bm25 | +0.0249 | [+0.0127, +0.0377] | 0.0002 |
| hybrid_bm25_weighted | hybrid_weighted | +0.0307 | [+0.0158, +0.0469] | 0.0001 |
| hybrid_bm25_weighted | semantic | +0.0353 | [+0.0194, +0.0525] | 0.0001 |

## Sapuan alpha (query evaluasi; hanya untuk analisis, jangan dipakai memilih alpha)

| alpha | nDCG@10 | recall@100 |
|---|---|---|
| 0.0 | 0.2341 | 0.2634 |
| 0.1 | 0.2682 | 0.2812 |
| 0.2 | 0.2760 | 0.2877 |
| 0.3 | 0.2839 | 0.2979 |
| 0.4 | 0.2910 | 0.3059 |
| 0.5 | 0.2996 | 0.3106 |
| 0.6 | 0.3073 | 0.3133 |
| 0.7 | 0.3171 | 0.3164 |
| 0.8 | 0.3215 | 0.3156 |
| 0.9 | 0.3219 | 0.3137 |
| 1.0 | 0.3174 | 0.3111 |
