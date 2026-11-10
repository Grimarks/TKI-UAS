# Hasil evaluasi: scifact

- Dokumen: 5183; query tuning: 809 (split:train); query evaluasi: 300 (split `test`)
- alpha dipilih pada query tuning: **0.9**
- bobot tiga sinyal dipilih pada query tuning: **sem=0.6, kp=0.0, bm25=0.4**

| Skenario | ndcg@10 | map@100 | recall@10 | recall@100 | precision@10 | mrr@10 |
|---|---|---|---|---|---|---|
| BM25 (leksikal) | 0.6834 | 0.6357 | 0.8312 | 0.9193 | 0.0917 | 0.6416 |
| Semantic-only | 0.6497 | 0.6072 | 0.7883 | 0.9250 | 0.0890 | 0.6097 |
| Keyphrase-only | 0.2921 | 0.2660 | 0.4017 | 0.6481 | 0.0437 | 0.2624 |
| Hybrid weighted (alpha=0.9) | 0.6595 | 0.6120 | 0.8117 | 0.9350 | 0.0913 | 0.6146 |
| Hybrid RRF | 0.4901 | 0.4445 | 0.6597 | 0.9327 | 0.0740 | 0.4433 |
| Hybrid+BM25 weighted (sem=0.6, kp=0.0, bm25=0.4) | 0.7363 | 0.6964 | 0.8600 | 0.9583 | 0.0967 | 0.7019 |
| Hybrid+BM25 RRF | 0.5905 | 0.5343 | 0.7814 | 0.9543 | 0.0873 | 0.5357 |

## Uji signifikansi (paired bootstrap, nDCG@10, model dikurangi pembanding)

| Model | Pembanding | selisih rata-rata | CI 95% | p |
|---|---|---|---|---|
| hybrid_weighted | semantic | +0.0098 | [-0.0012, +0.0213] | 0.0834 |
| hybrid_weighted | keyphrase | +0.3674 | [+0.3221, +0.4125] | 0.0001 |
| hybrid_weighted | bm25 | -0.0238 | [-0.0583, +0.0096] | 0.1682 |
| hybrid_bm25_weighted | bm25 | +0.0529 | [+0.0345, +0.0719] | 0.0001 |
| hybrid_bm25_weighted | hybrid_weighted | +0.0767 | [+0.0507, +0.1033] | 0.0001 |
| hybrid_bm25_weighted | semantic | +0.0866 | [+0.0592, +0.1151] | 0.0001 |

## Sapuan alpha (query evaluasi; hanya untuk analisis, jangan dipakai memilih alpha)

| alpha | nDCG@10 | recall@100 |
|---|---|---|
| 0.0 | 0.2932 | 0.6714 |
| 0.1 | 0.3116 | 0.7381 |
| 0.2 | 0.3332 | 0.8131 |
| 0.3 | 0.3742 | 0.8649 |
| 0.4 | 0.4146 | 0.9067 |
| 0.5 | 0.4705 | 0.9350 |
| 0.6 | 0.5288 | 0.9483 |
| 0.7 | 0.5948 | 0.9450 |
| 0.8 | 0.6356 | 0.9383 |
| 0.9 | 0.6595 | 0.9350 |
| 1.0 | 0.6497 | 0.9250 |
