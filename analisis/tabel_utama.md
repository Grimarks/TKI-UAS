## Tabel utama (split test)

| Dataset | Skenario | nDCG@10 | MAP@100 | Recall@10 | Recall@100 | P@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| SciFact | BM25 (pembanding) | 0.6834 | 0.6357 | 0.8312 | 0.9193 | 0.0917 | 0.6416 |
| SciFact | Semantic-only | 0.6497 | 0.6072 | 0.7883 | 0.9250 | 0.0890 | 0.6097 |
| SciFact | Keyphrase-only | 0.4078 | 0.3727 | 0.5346 | 0.7662 | 0.0590 | 0.3749 |
| SciFact | Hybrid (α = 0.9) | 0.6671 | 0.6243 | 0.8073 | 0.9317 | 0.0913 | 0.6241 |
| NFCorpus | BM25 (pembanding) | 0.3278 | 0.1463 | 0.1520 | 0.2484 | 0.2387 | 0.5314 |
| NFCorpus | Semantic-only | 0.3174 | 0.1431 | 0.1569 | 0.3111 | 0.2437 | 0.5083 |
| NFCorpus | Keyphrase-only | 0.2321 | 0.0857 | 0.1004 | 0.1705 | 0.1724 | 0.4200 |
| NFCorpus | Hybrid (α = 0.9) | 0.3219 | 0.1466 | 0.1575 | 0.3137 | 0.2430 | 0.5225 |
