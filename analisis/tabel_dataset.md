## Statistik koleksi

| Koleksi | Dokumen | Split | Query di qrels | Query terpakai* | Pasangan qrels | Rata-rata dok. relevan/query | Peran |
|---|---|---|---|---|---|---|---|
| SciFact | 5183 | test | 300 | 300 | 339 | 1.1 | Evaluasi akhir |
| SciFact | 5183 | train | 809 | 809 | 919 | 1.1 | Penalaan α |
| NFCorpus | 3633 | dev | 324 | 324 | 11385 | 35.1 | Penalaan α |
| NFCorpus | 3633 | test | 323 | 323 | 12334 | 38.2 | Evaluasi akhir |
| NFCorpus | 3633 | train | 2590 | 2590 | 110575 | 42.7 | Tidak dipakai |

*Query terpakai = query yang punya minimal satu dokumen relevan di korpus (`data.usable_queries`). Untuk split penalaan, query yang juga ada di test dikeluarkan.
