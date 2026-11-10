## Waktu pencarian per query (komputer lokal)

| Dataset | Skenario / langkah | Query | Rata-rata (ms) | Median (ms) |
|---|---|---|---|---|
| SciFact | Bangun indeks BM25 di memori (5183 dokumen, sekali) | 1 | 692.3 | - |
| SciFact | BM25 | 300 | 0.6 | 0.6 |
| SciFact | Semantic | 300 | 3.8 | 3.6 |
| SciFact | Keyphrase | 300 | 0.4 | 0.4 |
| SciFact | Hybrid | 300 | 4.2 | 3.9 |
| SciFact | BART-KPG query bebas (mps) | 20 | 116.4 | 121.7 |
| NFCorpus | Bangun indeks BM25 di memori (3633 dokumen, sekali) | 1 | 527.2 | - |
| NFCorpus | BM25 | 323 | 0.4 | 0.4 |
| NFCorpus | Semantic | 323 | 4.1 | 3.7 |
| NFCorpus | Keyphrase | 323 | 0.3 | 0.3 |
| NFCorpus | Hybrid | 323 | 4.3 | 4.0 |
| NFCorpus | BART-KPG query bebas (mps) | 20 | 83.1 | 86.9 |

Diukur dengan `HybridRetriever.search(top_k=10)` atas seluruh query test, 3 putaran, setelah 5 query pemanasan. Keyphrase query diambil dari berkas precomputed; untuk query bebas di UI ditambah waktu generasi BART-KPG (baris terakhir tiap dataset). Waktu bergantung pada mesin dan beban saat pengukuran.
