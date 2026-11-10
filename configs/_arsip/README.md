# Konfigurasi arsip (desain lama)

Konfigurasi di folder ini berasal dari desain penelitian sebelumnya dan **tidak bisa dijalankan lagi**
dengan kode sekarang:

| Berkas | Isi | Kenapa tidak jalan |
|---|---|---|
| `scifact.yaml` | SciFact + keyphrase YAKE | YAKE sudah dihapus dari pipeline |
| `nfcorpus.yaml` | NFCorpus + keyphrase YAKE | YAKE sudah dihapus dari pipeline |
| `scifact_ft.yaml` | SciFact + encoder fine-tune KP20k + YAKE | YAKE dihapus; fine-tune encoder bukan bagian pipeline yang dilaporkan |

Ketiganya juga memakai `retrieval.fusion`, yang sekarang ditolak (fusi RRF dihapus).
Hasil dan indeks yang dibuat dengan konfigurasi ini ada di `results/_arsip/` dan `indexes/_arsip/`.
Konfigurasi aktif: `configs/scifact_bart.yaml`, `configs/nfcorpus_bart.yaml`, `configs/demo.yaml`.
