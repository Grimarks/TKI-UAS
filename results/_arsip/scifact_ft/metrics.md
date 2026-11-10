# Hasil evaluasi: scifact

- Dokumen: 5183; query tuning: 809 (split:train); query evaluasi: 300 (split `test`)
- alpha dipilih pada query tuning: **1.0**

| Skenario | ndcg@10 | map@100 | recall@10 | recall@100 | precision@10 | mrr@10 |
|---|---|---|---|---|---|---|
| BM25 (leksikal) | 0.6834 | 0.6357 | 0.8312 | 0.9193 | 0.0917 | 0.6416 |
| Semantic-only | 0.6506 | 0.6116 | 0.7810 | 0.9333 | 0.0880 | 0.6132 |
| Keyphrase-only | 0.2921 | 0.2660 | 0.4017 | 0.6481 | 0.0437 | 0.2624 |
| Hybrid weighted (alpha=1.0) | 0.6506 | 0.6116 | 0.7810 | 0.9333 | 0.0880 | 0.6132 |
| Hybrid RRF | 0.4877 | 0.4438 | 0.6540 | 0.9167 | 0.0730 | 0.4426 |

## Uji signifikansi (paired bootstrap, nDCG@10, hybrid weighted dikurangi pembanding)

| Pembanding | selisih rata-rata | CI 95% | p |
|---|---|---|---|
| semantic | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| keyphrase | +0.3585 | [+0.3088, +0.4075] | 0.0001 |
| bm25 | -0.0327 | [-0.0714, +0.0049] | 0.0900 |

## Sapuan alpha (query evaluasi; hanya untuk analisis, jangan dipakai memilih alpha)

| alpha | nDCG@10 | recall@100 |
|---|---|---|
| 0.0 | 0.2921 | 0.6614 |
| 0.1 | 0.3120 | 0.7214 |
| 0.2 | 0.3336 | 0.7864 |
| 0.3 | 0.3692 | 0.8582 |
| 0.4 | 0.4078 | 0.8983 |
| 0.5 | 0.4621 | 0.9183 |
| 0.6 | 0.5113 | 0.9267 |
| 0.7 | 0.5792 | 0.9317 |
| 0.8 | 0.6190 | 0.9333 |
| 0.9 | 0.6473 | 0.9333 |
| 1.0 | 0.6506 | 0.9333 |
