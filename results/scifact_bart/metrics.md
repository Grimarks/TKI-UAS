# Hasil evaluasi: scifact

- Dokumen: 5183; query tuning: 809 (split:train); query evaluasi: 300 (split `test`)
- alpha dipilih pada query tuning: **0.9**

| Skenario | ndcg@10 | map@100 | recall@10 | recall@100 | precision@10 | mrr@10 |
|---|---|---|---|---|---|---|
| BM25 (pembanding) | 0.6834 | 0.6357 | 0.8312 | 0.9193 | 0.0917 | 0.6416 |
| Semantic-only | 0.6497 | 0.6072 | 0.7883 | 0.9250 | 0.0890 | 0.6097 |
| Keyphrase-only | 0.4078 | 0.3727 | 0.5346 | 0.7662 | 0.0590 | 0.3749 |
| Hybrid weighted (alpha=0.9) | 0.6671 | 0.6243 | 0.8073 | 0.9317 | 0.0913 | 0.6241 |

## Uji signifikansi (paired bootstrap, nDCG@10, model dikurangi pembanding)

| Model | Pembanding | selisih rata-rata | CI 95% | p |
|---|---|---|---|---|
| hybrid_weighted | semantic | +0.0174 | [+0.0048, +0.0307] | 0.0060 |
| hybrid_weighted | keyphrase | +0.2593 | [+0.2141, +0.3035] | 0.0001 |
| hybrid_weighted | bm25 | -0.0163 | [-0.0507, +0.0173] | 0.3538 |

## Sapuan alpha (query evaluasi; hanya untuk analisis, jangan dipakai memilih alpha)

| alpha | nDCG@10 | recall@100 |
|---|---|---|
| 0.0 | 0.4091 | 0.8201 |
| 0.1 | 0.4242 | 0.8459 |
| 0.2 | 0.4560 | 0.8793 |
| 0.3 | 0.4906 | 0.9027 |
| 0.4 | 0.5288 | 0.9293 |
| 0.5 | 0.5540 | 0.9327 |
| 0.6 | 0.5983 | 0.9393 |
| 0.7 | 0.6445 | 0.9450 |
| 0.8 | 0.6653 | 0.9417 |
| 0.9 | 0.6671 | 0.9317 |
| 1.0 | 0.6497 | 0.9250 |
