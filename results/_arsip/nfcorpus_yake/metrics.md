# Hasil evaluasi: nfcorpus

- Dokumen: 3633; query tuning: 324 (split:dev); query evaluasi: 323 (split `test`)
- alpha dipilih pada query tuning: **0.7**
- bobot tiga sinyal dipilih pada query tuning: **sem=0.4, kp=0.0, bm25=0.6**

| Skenario | ndcg@10 | map@100 | recall@10 | recall@100 | precision@10 | mrr@10 |
|---|---|---|---|---|---|---|
| BM25 (leksikal) | 0.3278 | 0.1474 | 0.1520 | 0.2586 | 0.2387 | 0.5314 |
| Semantic-only | 0.3174 | 0.1431 | 0.1569 | 0.3111 | 0.2437 | 0.5083 |
| Keyphrase-only | 0.2064 | 0.0779 | 0.0882 | 0.1674 | 0.1557 | 0.3683 |
| Hybrid weighted (alpha=0.7) | 0.3139 | 0.1436 | 0.1551 | 0.3118 | 0.2350 | 0.5084 |
| Hybrid RRF | 0.2980 | 0.1316 | 0.1462 | 0.3049 | 0.2266 | 0.4812 |
| Hybrid+BM25 weighted (sem=0.4, kp=0.0, bm25=0.6) | 0.3575 | 0.1689 | 0.1710 | 0.3164 | 0.2632 | 0.5627 |
| Hybrid+BM25 RRF | 0.3280 | 0.1497 | 0.1638 | 0.3207 | 0.2477 | 0.5199 |

## Uji signifikansi (paired bootstrap, nDCG@10, model dikurangi pembanding)

| Model | Pembanding | selisih rata-rata | CI 95% | p |
|---|---|---|---|---|
| hybrid_weighted | semantic | -0.0035 | [-0.0127, +0.0060] | 0.4794 |
| hybrid_weighted | keyphrase | +0.1075 | [+0.0869, +0.1282] | 0.0001 |
| hybrid_weighted | bm25 | -0.0139 | [-0.0351, +0.0071] | 0.1928 |
| hybrid_bm25_weighted | bm25 | +0.0297 | [+0.0195, +0.0405] | 0.0001 |
| hybrid_bm25_weighted | hybrid_weighted | +0.0436 | [+0.0277, +0.0607] | 0.0001 |
| hybrid_bm25_weighted | semantic | +0.0401 | [+0.0230, +0.0584] | 0.0001 |

## Sapuan alpha (query evaluasi; hanya untuk analisis, jangan dipakai memilih alpha)

| alpha | nDCG@10 | recall@100 |
|---|---|---|
| 0.0 | 0.2076 | 0.2565 |
| 0.1 | 0.2465 | 0.2786 |
| 0.2 | 0.2551 | 0.2896 |
| 0.3 | 0.2646 | 0.2938 |
| 0.4 | 0.2767 | 0.3008 |
| 0.5 | 0.2914 | 0.3070 |
| 0.6 | 0.3026 | 0.3108 |
| 0.7 | 0.3139 | 0.3118 |
| 0.8 | 0.3195 | 0.3132 |
| 0.9 | 0.3179 | 0.3122 |
| 1.0 | 0.3174 | 0.3111 |
