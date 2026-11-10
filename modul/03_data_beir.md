# Bab 03 — Data BEIR: SciFact dan NFCorpus

## Tujuan belajar

1. Menjelaskan tiga berkas format BEIR: `corpus.jsonl`, `queries.jsonl`, dan `qrels/<split>.tsv`.
2. Membedakan SciFact dan NFCorpus (jenis kueri, tingkat relevansi, kepadatan qrels).
3. Menjelaskan pembagian split: kueri **tune** untuk memilih α, kueri **test** untuk melaporkan hasil.
4. Membaca Tabel 2 laporan dan tahu dari mana angkanya.

## 1. Format BEIR

BEIR adalah kumpulan koleksi uji IR dengan format seragam. Setiap koleksi berisi tiga jenis berkas.

**`corpus.jsonl`**: satu dokumen per baris (contoh asli SciFact, dipotong):

```json
{"_id": "4983", "title": "Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.", "text": "Alterations of the architecture of cerebral white matter ..."}
```

**`queries.jsonl`**: satu kueri per baris:

```json
{"_id": "0", "text": "0-dimensional biomaterials lack inductive properties.", "metadata": {}}
{"_id": "PLAIN-681", "text": "betel nuts", "metadata": {"url": "http://nutritionfacts.org/topics/betel-nuts/"}}
```

(Baris kedua dari NFCorpus.)

**`qrels/test.tsv`**: penilaian relevansi, satu pasangan kueri–dokumen per baris:

```text
query-id	corpus-id	score
1	31715818	1
PLAIN-2	MED-2427	2
```

Skor relevansi SciFact selalu 1 (relevan). NFCorpus punya tingkat 1 dan 2 (dokumen yang lebih relevan diberi 2).
Dokumen yang tidak tercantum dianggap tidak relevan (skor 0).

Kode pembacanya ada di `data.py`. Dokumen tanpa judul dan isi dilewati, dan `_id` ganda ditolak supaya data rusak
tidak lolos diam-diam.

<!-- kode: src/hybrid_ir/data.py#load_qrels_file -->
*Kode asli: `src/hybrid_ir/data.py`, baris 116–131.*

```python
def load_qrels_file(path: Path) -> dict[str, dict[str, int]]:
    """Muat satu berkas qrels TSV; baris header dilewati."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0].strip().lower() in ("query-id", "query_id", "qid"):
                continue
            if len(row) < 3:
                raise DataError(f"{path.name}: baris qrels harus punya 3 kolom, ditemukan {row}")
            qid, did, score = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                rel = int(float(score))
            except ValueError as exc:
                raise DataError(f"{path.name}: skor relevansi bukan angka: {score!r}") from exc
            qrels.setdefault(qid, {})[did] = rel
    return qrels
```
<!-- /kode -->

## 2. Dua koleksi

| | SciFact | NFCorpus |
|---|---|---|
| Dokumen | 5.183 abstrak ilmiah | 3.633 artikel PubMed |
| Kueri | klaim ilmiah yang ditulis pakar (Wadden dkk. [21]) | pertanyaan kesehatan berbahasa awam dari NutritionFacts.org (Boteva dkk. [22]) |
| Tingkat relevansi | 1 saja | 1 dan 2 |
| Dokumen relevan per kueri test | rata-rata 1.1 | rata-rata 38.2 |

Perbedaan kepadatan qrels ini penting. Di SciFact biasanya ada **satu** dokumen yang benar, jadi nDCG@10 bisa tinggi
(0.6–0.7). Di NFCorpus ada puluhan dokumen relevan per kueri, sehingga mustahil memasukkan semuanya ke 10 teratas;
nDCG@10 pun lebih rendah (sekitar 0.3) dan Recall@10 kecil.

## 3. Split dan Tabel 2

Statistik ini dibangkitkan oleh `scripts/analisis_uas.py` ke `analisis/tabel_dataset.md`:

| Koleksi | Dokumen | Split | Kueri | Pasangan qrels | Peran |
|---|---|---|---|---|---|
| SciFact | 5.183 | train | 809 | 919 | penalaan α |
| | | test | 300 | 339 | evaluasi akhir |
| NFCorpus | 3.633 | train | 2.590 | 110.575 | tidak dipakai |
| | | dev | 324 | 11.385 | penalaan α |
| | | test | 323 | 12.334 | evaluasi akhir |

SciFact tidak punya split dev, jadi α ditala di train. NFCorpus punya dev, jadi train tidak dipakai (kita tidak
melatih model apa pun). Kueri yang dipakai hanyalah kueri yang punya minimal satu dokumen relevan di korpus:

<!-- kode: src/hybrid_ir/data.py#usable_queries -->
*Kode asli: `src/hybrid_ir/data.py`, baris 155–162.*

```python
def usable_queries(dataset: Dataset, split: str) -> dict[str, str]:
    """Query pada `split` yang punya minimal satu dokumen relevan yang ada di korpus."""
    doc_ids = {d.id for d in dataset.documents}
    out: dict[str, str] = {}
    for qid, rels in dataset.qrels_for(split).items():
        if qid in dataset.queries and any(rel > 0 and did in doc_ids for did, rel in rels.items()):
            out[qid] = dataset.queries[qid]
    return out
```
<!-- /kode -->

Pemilihan kueri tune dan eval dilakukan di `evaluation._split_queries`. Kalau ada kueri yang muncul di split tune
**dan** test, kueri itu hanya dipakai sebagai test, supaya tidak ada kueri test yang ikut menala α.

<!-- kode: src/hybrid_ir/evaluation.py#_split_queries -->
*Kode asli: `src/hybrid_ir/evaluation.py`, baris 187–203.*

```python
def _split_queries(dataset: Dataset, eval_split: str, tune_split: str | None, dev_fraction: float,
                   seed: int) -> QuerySplit:
    """Pilih query tune dan eval. Query yang ada di keduanya hanya dipakai sebagai eval."""
    eval_q = usable_queries(dataset, eval_split)
    if tune_split:
        tune_q = {q: t for q, t in usable_queries(dataset, tune_split).items() if q not in eval_q}
        tune_qrels = dataset.qrels_for(tune_split)
        source = f"split:{tune_split}"
    else:
        dev_ids, eval_ids = split_query_ids(list(eval_q), dev_fraction, seed)
        tune_q = {q: eval_q[q] for q in dev_ids}
        tune_qrels = dataset.qrels_for(eval_split)
        eval_q = {q: eval_q[q] for q in eval_ids}
        source = f"dev-fraction:{dev_fraction}"
    if not tune_q or not eval_q:
        raise ValueError("query untuk tuning/evaluasi kosong; periksa split atau perbesar dataset")
    return QuerySplit(tune_q, tune_qrels, eval_q, dataset.qrels_for(eval_split), source)
```
<!-- /kode -->

## 4. Zero-shot

Model embedding (MiniLM) dan model keyphrase (BART-KPG) **tidak dilatih ulang** pada SciFact maupun NFCorpus. Satu-satunya
yang "dipelajari" dari data adalah satu angka, α. Pengujian seperti ini disebut *zero-shot* (Ren dkk. [5]): model
dipakai pada koleksi yang berbeda dari data latihnya.

## Ringkasan

- BEIR = `corpus.jsonl` + `queries.jsonl` + `qrels/<split>.tsv`.
- SciFact: kueri klaim, sekitar 1 dokumen relevan per kueri. NFCorpus: kueri awam, sekitar 38 dokumen relevan per
  kueri dengan dua tingkat relevansi.
- α ditala di SciFact train (809 kueri) dan NFCorpus dev (324 kueri), lalu dilaporkan di test (300 dan 323 kueri).
- Tidak ada model yang dilatih ulang: *zero-shot*.

## Soal latihan

**Soal 1.** Apa arti baris `PLAIN-2	MED-2427	2` di qrels NFCorpus?

**Soal 2.** Kenapa nDCG@10 NFCorpus jauh lebih rendah daripada SciFact, padahal sistemnya sama?

**Soal 3.** Kenapa split train NFCorpus tidak dipakai?

**Soal 4.** Apa yang dilakukan `_split_queries` bila sebuah kueri muncul di split tune dan test sekaligus? Kenapa?

## Kunci jawaban

**Jawaban 1.** Untuk kueri `PLAIN-2`, dokumen `MED-2427` relevan dengan tingkat 2 (sangat relevan).

**Jawaban 2.** Koleksinya berbeda. NFCorpus punya rata-rata 38.2 dokumen relevan per kueri, jadi IDCG@10 (urutan ideal)
sangat tinggi dan sulit didekati. Kuerinya juga pendek dan berbahasa awam, sehingga lebih sulit dicocokkan dengan
artikel ilmiah. Angka antar-koleksi tidak boleh dibandingkan langsung; yang dibandingkan adalah skenario di dalam satu
koleksi.

**Jawaban 3.** Proyek ini tidak melatih model, dan untuk menala satu parameter (α) split dev sudah cukup. Split train
yang besar (2.590 kueri) tidak dibutuhkan.

**Jawaban 4.** Kueri itu dikeluarkan dari himpunan tune (`if q not in eval_q`). Tujuannya agar tidak ada kueri test yang
ikut memengaruhi pemilihan α, sehingga hasil test tetap adil.
