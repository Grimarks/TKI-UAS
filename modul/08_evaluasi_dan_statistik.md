# Bab 08 — Evaluasi dan uji statistik

## Tujuan belajar

1. Menghitung nDCG@10 manual untuk satu kueri dengan relevansi bertingkat (persamaan (8) dan (9)).
2. Menjelaskan MAP@100, Recall@k, Precision@10, dan MRR@10, serta kapan masing-masing berguna.
3. Menjelaskan pemisahan kueri tune dan test serta kenapa itu penting.
4. Menjelaskan *paired bootstrap*: selisih rata-rata, CI 95%, dan nilai p.

## 1. nDCG@10, dihitung tangan

$$DCG@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}, \qquad nDCG@k = \frac{DCG@k}{IDCG@k}$$

$rel_i$ adalah tingkat relevansi dokumen di posisi $i$. Pembagi $\log_2(i+1)$ membuat posisi bawah bernilai lebih kecil.
IDCG adalah DCG dari urutan **ideal** (semua dokumen relevan diurutkan dari relevansi tertinggi), sehingga nDCG berada
di antara 0 dan 1.

**Contoh nyata: kueri NFCorpus PLAIN-2600 "Eggs and Arterial Function", skenario hybrid.** Kueri ini punya 10 dokumen
relevan (4 bertingkat 2, 6 bertingkat 1). Relevansi 10 dokumen teratas hasil hybrid:

| Posisi $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| $rel_i$ | 2 | 2 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 |
| $rel_i / \log_2(i+1)$ | 2.0 | 1.2619 | 0 | 0.8614 | 0 | 0 | 0.3333 | 0 | 0 | 0 |

$DCG@10 = 2 + 1.2619 + 0.8614 + 0.3333 = 4.4565$.

Urutan ideal: 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, sehingga $IDCG@10 = 7.1052$ (jumlah sepuluh suku yang sama caranya).

$nDCG@10 = 4.4565 / 7.1052 = 0.6272$. Angka ini sama persis dengan baris `PLAIN-2600` di
`results/nfcorpus_bart/per_query_ndcg10.csv` (kolom hybrid).

Contoh yang lebih sederhana dari SciFact (satu dokumen relevan): pada kueri 1185, semantik saja menaruh dokumen relevan
di peringkat 3, sehingga $nDCG@10 = (1/\log_2 4)/(1/\log_2 2) = 0.5$; hybrid menaruhnya di peringkat 1, sehingga
nDCG@10 = 1.

<!-- kode: src/hybrid_ir/evaluation.py#ndcg_at_k -->
*Kode asli: `src/hybrid_ir/evaluation.py`, baris 51–56.*

```python
def ndcg_at_k(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    """Persamaan (9): DCG@k peringkat sistem dibagi DCG@k urutan ideal (relevansi bertingkat)."""
    gains = [max(qrel.get(d, 0), 0) for d in ranked[:k]]
    ideal = sorted((max(v, 0) for v in qrel.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0
```
<!-- /kode -->

<!-- kode: src/hybrid_ir/evaluation.py#_dcg -->
*Kode asli: `src/hybrid_ir/evaluation.py`, baris 46–48.*

```python
def _dcg(gains: Sequence[float]) -> float:
    """Persamaan (8). i dihitung dari 0 di kode, sehingga penyebutnya log2(i + 2)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))
```
<!-- /kode -->

Di kode, `i` dihitung dari 0, sehingga penyebutnya `log2(i + 2)`; ini sama dengan $\log_2(i+1)$ untuk $i$ yang dihitung
dari 1.

## 2. Metrik lain

Untuk kueri PLAIN-2600 yang sama:

| Metrik | Arti | Nilai |
|---|---|---|
| P@10 | proporsi dokumen relevan di 10 teratas (4 dari 10) | 0.4 |
| Recall@10 | proporsi dokumen relevan yang masuk 10 teratas (4 dari 10 relevan) | 0.4 |
| MRR@10 | 1 / posisi dokumen relevan pertama (posisi 1) | 1.0 |
| AP@100 | rata-rata presisi di setiap posisi dokumen relevan, dibagi jumlah **semua** dokumen relevan | 0.3539 |

- **nDCG@10** adalah metrik utama karena memperhitungkan posisi dan relevansi bertingkat. Metrik ini juga dipakai di
  penelitian acuan kedua [7].
- **MAP@100** dan **Recall@100** melihat daftar yang lebih panjang (cocok untuk melihat apakah dokumen relevan "masuk"
  sama sekali).
- **MRR@10** berguna untuk SciFact yang umumnya hanya punya satu dokumen relevan.
- Setiap metrik dirata-ratakan atas semua kueri test yang punya dokumen relevan.

<!-- kode: src/hybrid_ir/evaluation.py#average_precision -->
*Kode asli: `src/hybrid_ir/evaluation.py`, baris 70–80.*

```python
def average_precision(ranked: Sequence[str], qrel: dict[str, int], k: int) -> float:
    """AP@k dinormalkan oleh jumlah SEMUA dokumen relevan (konvensi trec_eval/BEIR)."""
    n_rel = sum(1 for v in qrel.values() if v > 0)
    if not n_rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if qrel.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / n_rel
```
<!-- /kode -->

## 3. Tune vs test

Satu-satunya parameter yang ditala adalah α. Aturannya:

1. Pilih α di kueri **tune** (SciFact train 809, NFCorpus dev 324).
2. Laporkan semua metrik di kueri **test** (300 dan 323) dengan α itu.
3. Sapuan α di kueri test boleh dihitung, tetapi hanya untuk analisis (`alpha_sweep_eval_posthoc`).

Kalau α dipilih langsung di test, hasilnya menjadi terlalu optimis karena kita "mengintip" jawaban. Di proyek ini α
terbaik di test kebetulan juga 0.9, tetapi itu dilihat **setelah** α dipilih.

## 4. Paired bootstrap

Pertanyaannya: apakah selisih rata-rata nDCG@10 hybrid vs pembanding sungguh ada, atau hanya kebetulan dari kueri yang
kebetulan terpilih?

Langkah `paired_bootstrap`:

1. Hitung selisih per kueri $\delta_q = nDCG_{hyb}(q) - nDCG_{lain}(q)$. Ini **berpasangan**: kueri yang sama
   dibandingkan dengan dirinya sendiri.
2. Ambil sampel ulang $n$ kueri **dengan pengembalian** sebanyak 10.000 kali, lalu hitung rata-rata selisih tiap sampel.
3. CI 95% = persentil 2.5 dan 97.5 dari 10.000 rata-rata itu.
4. Nilai p (dua sisi) = 2 × min(proporsi rata-rata ≤ 0, proporsi rata-rata ≥ 0).

<!-- kode: src/hybrid_ir/evaluation.py#paired_bootstrap -->
*Kode asli: `src/hybrid_ir/evaluation.py`, baris 115–134.*

```python
def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 13) -> dict:
    """Uji bootstrap berpasangan untuk selisih rata-rata a − b (dua sisi).

    Langkah: hitung selisih per query; ambil sampel ulang n query (dengan pengembalian) sebanyak n_boot kali;
    CI 95% = persentil 2.5 dan 97.5 dari rata-rata sampel; p = 2·min(P(rata-rata ≤ 0), P(rata-rata ≥ 0)).
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = diff.size
    if n < 2:
        return {"mean_diff": float("nan"), "ci95": [float("nan"), float("nan")], "p_value": float("nan"), "n": int(n)}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    step = max(1, 2_000_000 // n)          # batasi memori
    for s in range(0, n_boot, step):
        e = min(n_boot, s + step)
        means[s:e] = diff[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / n_boot)))
    return {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "p_value": p, "n": int(n)}
```
<!-- /kode -->

Seed tetap (13), jadi hasilnya selalu sama. Batas bawah p adalah 1/10.000 = 0.0001: kalau tidak ada satu pun sampel
yang selisihnya ≤ 0, p ditulis 0.0001, dan di laporan ditulis "p < 0.001".

Hasil proyek (Tabel 4 laporan):

| Koleksi | Perbandingan | Selisih | CI 95% | p |
|---|---|---|---|---|
| SciFact | hybrid − semantic | +0.0174 | [+0.0048, +0.0307] | 0.006 |
| SciFact | hybrid − BM25 | −0.0163 | [−0.0507, +0.0173] | 0.354 |
| NFCorpus | hybrid − semantic | +0.0046 | [+0.0003, +0.0092] | 0.035 |
| NFCorpus | hybrid − BM25 | −0.0058 | [−0.0280, +0.0154] | 0.590 |

Cara membaca: kalau CI 95% **tidak** mencakup nol (misalnya [+0.0048, +0.0307]), selisihnya signifikan. Kalau mencakup
nol (misalnya [−0.0507, +0.0173]), selisihnya tidak signifikan.

## Ringkasan

- nDCG@10 PLAIN-2600 = 4.4565 / 7.1052 = 0.6272, cocok dengan hasil resmi.
- Metrik pendukung: MAP@100, Recall@10/100, P@10, MRR@10.
- α dipilih di tune, hasil dilaporkan di test.
- Paired bootstrap 10.000×: hybrid signifikan lebih baik dari semantic dan keyphrase; tidak signifikan berbeda dari BM25.

## Soal latihan

**Soal 1.** Sebuah kueri SciFact punya satu dokumen relevan, dan sistem menaruhnya di peringkat 2. Berapa nDCG@10-nya?

**Soal 2.** Kenapa nDCG memakai IDCG sebagai pembagi?

**Soal 3.** Untuk PLAIN-2600, kenapa Recall@10 (0.4) jauh di bawah MRR@10 (1.0)?

**Soal 4.** Apa arti "berpasangan" pada paired bootstrap, dan kenapa itu lebih tepat daripada membandingkan dua rata-rata
secara terpisah?

**Soal 5.** Hybrid − semantic di NFCorpus: selisih +0.0046 dengan CI [+0.0003, +0.0092]. Signifikan atau tidak? Jelaskan.

## Kunci jawaban

**Jawaban 1.** $DCG = 1/\log_2 3 = 0.631$, $IDCG = 1/\log_2 2 = 1$, sehingga nDCG@10 = 0.631.

**Jawaban 2.** Supaya nilainya berada di rentang 0–1 dan bisa dibandingkan antar-kueri. Kueri dengan banyak dokumen
relevan punya DCG maksimum yang lebih besar; tanpa IDCG, kueri seperti itu akan mendominasi rata-rata.

**Jawaban 3.** MRR hanya melihat dokumen relevan **pertama**, yang ada di posisi 1. Recall@10 melihat berapa dari
**semua** 10 dokumen relevan yang masuk 10 teratas, yaitu hanya 4.

**Jawaban 4.** Setiap selisih dihitung pada kueri yang sama, sehingga perbedaan tingkat kesulitan antar-kueri saling
meniadakan. Variasi yang tersisa hanya karena perbedaan sistem, sehingga uji lebih peka dan adil.

**Jawaban 5.** Signifikan (p = 0.035 < 0.05), karena seluruh CI 95% berada di atas nol. Namun selisihnya sangat kecil,
jadi signifikan secara statistik tidak sama dengan besar secara praktis.
