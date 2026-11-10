## Lingkungan

### Komputer lokal (pengindeksan, pencarian, evaluasi)

| Komponen | Nilai |
|---|---|
| Model Name | MacBook Pro |
| Chip | Apple M4 Pro |
| Total Number of Cores | 12 (8 Performance and 4 Efficiency) |
| Memory | 24 GB |
| macOS | 26.5.1 |
| RAM (sysctl, GB) | 24 |
| CPU (sysctl) | Apple M4 Pro |
| device torch | mps |

| Pustaka | Versi |
|---|---|
| python | 3.14.4 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| torch | 2.14.1 |
| transformers | 5.18.0 |
| sentence-transformers | 6.1.0 |
| snowballstemmer | 3.1.1 |
| streamlit | 1.65.0 |
| matplotlib | 3.11.2 |

### Kaggle (pembangkitan keyphrase BART-KPG)

| Dataset | Worker | GPU | torch | transformers | Commit model | Teks | Detik |
|---|---|---|---|---|---|---|---|
| SciFact | 0 | Tesla T4 | 2.11.0+cu128 | 5.16.1 | 674e4b0 | 3144 | 130.1 |
| SciFact | 1 | Tesla T4 | 2.11.0+cu128 | 5.16.1 | 674e4b0 | 3144 | 132.0 |
| NFCorpus | 0 | Tesla T4 | 2.11.0+cu128 | 5.16.1 | 674e4b0 | 3405 | 91.4 |
| NFCorpus | 1 | Tesla T4 | 2.11.0+cu128 | 5.16.1 | 674e4b0 | 3404 | 107.6 |

Model taln-ls2n/bart-base-kp20k, beam 4, top-n 10; entri unik (dokumen + query): SciFact 6288, NFCorpus 6809.
