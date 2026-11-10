## Menang / seri / kalah per query (nDCG@10, split test)

| Dataset | Perbandingan | Query | Menang | Seri | Kalah | Seri, keduanya nDCG@10 = 0 | Rata-rata selisih |
|---|---|---|---|---|---|---|---|
| SciFact | Hybrid vs Semantic | 300 | 41 | 231 | 28 | 52 | +0.0174 |
| SciFact | Hybrid vs BM25 | 300 | 65 | 171 | 64 | 29 | -0.0163 |
| SciFact | Hybrid vs Keyphrase | 300 | 151 | 127 | 22 | 50 | +0.2593 |
| NFCorpus | Hybrid vs Semantic | 323 | 75 | 191 | 57 | 97 | +0.0046 |
| NFCorpus | Hybrid vs BM25 | 323 | 107 | 109 | 107 | 75 | -0.0058 |
| NFCorpus | Hybrid vs Keyphrase | 323 | 156 | 111 | 56 | 85 | +0.0899 |

Menang = nDCG@10 Hybrid lebih tinggi pada query itu; seri = sama persis (6 desimal).
