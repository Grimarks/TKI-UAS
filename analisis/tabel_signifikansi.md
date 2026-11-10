## Uji signifikansi (paired bootstrap 10.000×, nDCG@10, Hybrid dikurangi pembanding)

| Dataset | Perbandingan | Selisih rata-rata nDCG@10 | CI 95% | p | n | Signifikan (p < 0.05) |
|---|---|---|---|---|---|---|
| SciFact | Hybrid − Semantic | +0.0174 | [+0.0048, +0.0307] | 0.0060 | 300 | ya |
| SciFact | Hybrid − Keyphrase | +0.2593 | [+0.2141, +0.3035] | 0.0001 | 300 | ya |
| SciFact | Hybrid − BM25 | -0.0163 | [-0.0507, +0.0173] | 0.3538 | 300 | tidak |
| NFCorpus | Hybrid − Semantic | +0.0046 | [+0.0003, +0.0092] | 0.0350 | 323 | ya |
| NFCorpus | Hybrid − Keyphrase | +0.0899 | [+0.0680, +0.1121] | 0.0001 | 323 | ya |
| NFCorpus | Hybrid − BM25 | -0.0058 | [-0.0280, +0.0154] | 0.5902 | 323 | tidak |

Catatan: p = 0.0001 adalah batas bawah (1/10.000), artinya tidak ada satu pun sampel bootstrap yang selisihnya ≤ 0.
