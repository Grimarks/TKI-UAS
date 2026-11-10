# Bab 07 — Fusi hybrid: kandidat, min-max, dan α

## Tujuan belajar

1. Menjelaskan langkah fusi: kandidat union top-C, normalisasi min-max (6), kombinasi linear (7).
2. Menghitung skor fusi dua dokumen secara manual dengan angka nyata dari kueri SciFact 1185.
3. Menjelaskan arti α = 0.9 dan cara α dipilih.
4. Menjelaskan alasan memakai kombinasi linear, bukan RRF.

## 1. Kenapa perlu normalisasi?

Skor semantik (cosine embedding) dan skor keyphrase (cosine bobot unit) punya rentang yang berbeda. Pada kueri 1185,
skor semantik kandidat berkisar −0.0557 sampai 0.6013, sedangkan skor keyphrase 0 sampai 0.6682. Kalau langsung
dijumlahkan, jalur yang rentangnya lebih lebar akan mendominasi tanpa disengaja. Karena itu setiap jalur dinormalkan
min-max ke [0, 1] **di dalam himpunan kandidat**, persamaan (6):

$$\tilde{s}(d) = \frac{s(d) - s_{min}}{s_{max} - s_{min}}$$

<!-- kode: src/hybrid_ir/fusion.py#minmax -->
*Kode asli: `src/hybrid_ir/fusion.py`, baris 46–53.*

```python
def minmax(values: np.ndarray) -> np.ndarray:
    """Persamaan (6): normalisasi min-max ke [0, 1]. Bila semua nilai sama (tak ada informasi), hasilnya nol."""
    if values.size == 0:
        return values.astype(np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)
```
<!-- /kode -->

Kalau semua skor sama (misalnya semua keyphrase bernilai 0), hasil normalisasinya nol, sehingga jalur itu tidak
memberi informasi dan tidak mengubah urutan.

## 2. Himpunan kandidat

Kandidat = gabungan **C = 100** dokumen teratas menurut semantik dan C dokumen teratas menurut keyphrase (hanya yang
skornya > 0):

<!-- kode: src/hybrid_ir/fusion.py#candidate_set -->
*Kode asli: `src/hybrid_ir/fusion.py`, baris 56–59.*

```python
def candidate_set(sem: np.ndarray, kp: np.ndarray, candidate_k: int) -> np.ndarray:
    """Kandidat R = gabungan top-C semantik dan top-C keyphrase yang berskor positif (terurut menurut indeks)."""
    parts = [topk_indices(sem, candidate_k), positive_topk(kp, candidate_k)]
    return np.unique(np.concatenate(parts))
```
<!-- /kode -->

Pada kueri 1185, himpunan kandidatnya berisi 160 dokumen (100 dari semantik, 100 dari keyphrase, dan sebagian
beririsan). Dokumen di luar kandidat tidak diperingkat oleh hybrid.

## 3. Kombinasi linear

Persamaan (7):

$$s_{hyb}(q,d) = \alpha \cdot \tilde{s}_{sem}(q,d) + (1-\alpha) \cdot \tilde{s}_{kp}(q,d), \qquad 0 \le \alpha \le 1$$

<!-- kode: src/hybrid_ir/fusion.py#fuse_weighted -->
*Kode asli: `src/hybrid_ir/fusion.py`, baris 62–67.*

```python
def fuse_weighted(sem: np.ndarray, kp: np.ndarray, alpha: float, candidate_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Persamaan (7) pada kandidat R. Kembalikan (indeks_kandidat, skor_fusi) terurut menurun."""
    cand = candidate_set(sem, kp, candidate_k)
    fused = alpha * minmax(sem[cand]) + (1.0 - alpha) * minmax(kp[cand])
    order = np.lexsort((cand, -fused))
    return cand[order], fused[order]
```
<!-- /kode -->

$\alpha = 1$ berarti hanya semantik, dan $\alpha = 0$ berarti hanya keyphrase (pada kandidat). Seri diputus menurut
indeks dokumen (`np.lexsort`), sehingga hasilnya deterministik.

## 4. Contoh hitungan dua dokumen (kueri SciFact 1185, α = 0.9)

Kueri: *"The US health care system can save up to $750 million if 7% of patients waiting for kidney transplants
participate in the optimized national kidney paired donation program."* Keyphrase kuerinya adalah "kidney paired
donation" (bobot 1.0) dan "optimized national kidney transplantation program" (bobot 0.8).

Batas min-max di himpunan kandidat: semantik −0.0557 sampai 0.6013, keyphrase 0 sampai 0.6682.

| | Dokumen relevan 16737210 *Kidney paired donation and optimizing the use of live donor organs* | Dokumen 21203899 *Managing finances of shipping living donor kidneys…* (tidak relevan) |
|---|---|---|
| $s_{sem}$ | 0.5918 | 0.6013 (tertinggi) |
| $s_{kp}$ | 0.6675 | 0.0953 |
| $\tilde{s}_{sem}$ | (0.5918 + 0.0557) / (0.6013 + 0.0557) = 0.9855 | 1.0 |
| $\tilde{s}_{kp}$ | 0.6675 / 0.6682 = 0.9988 | 0.0953 / 0.6682 = 0.1427 |
| $0.9\,\tilde{s}_{sem} + 0.1\,\tilde{s}_{kp}$ | **0.9868** | 0.9143 |

Semantik saja menaruh dokumen 21203899 di atas (0.6013 > 0.5918), dan dokumen relevan ada di peringkat 3. Setelah fusi,
dokumen relevan naik ke **peringkat 1** dengan skor 0.9868. Alasannya, keyphrase-nya ("kidney paired donation",
"optimization", "renal transplantation") cocok dengan keyphrase kueri, sedangkan dokumen tidak relevan hanya punya
keyphrase umum seperti "kidney" dan "donor exchange".

Bobot keyphrase cuma 0.1, tetapi cukup karena skor semantik keduanya **sangat dekat** setelah normalisasi (0.9855 vs
1.0). Keyphrase menjadi "penentu" saat embedding ragu. Kalau α = 0.5, skor menjadi 0.9922 dan 0.5713, dan keyphrase
mendominasi. Itu bagus untuk kueri ini, tetapi buruk secara rata-rata (Bab 09).

## 5. Memilih α

α tidak ditebak. `evaluate` mencoba α = 0, 0.1, …, 1 pada **kueri penalaan** (SciFact train, NFCorpus dev), lalu memilih
nDCG@10 tertinggi. Kalau ada nilai yang sama, dipilih yang paling dekat ke 0.5:

<!-- kode: src/hybrid_ir/evaluation.py#_select_alpha -->
*Kode asli: `src/hybrid_ir/evaluation.py`, baris 221–224.*

```python
def _select_alpha(tune_sweep: list[dict]) -> float:
    """Alpha dengan nDCG@10 tertinggi pada query tune; bila seri, pilih yang paling dekat ke 0.5."""
    best = max(tune_sweep, key=lambda r: (r["ndcg@10"], -abs(r["alpha"] - 0.5)))
    return float(best["alpha"])
```
<!-- /kode -->

Di kedua koleksi hasilnya **α = 0.9**, dengan nDCG@10 penalaan 0.6699 (SciFact) dan 0.3076 (NFCorpus). Nilai ini
dicatat di `configs/*_bart.yaml` (`retrieval.alpha: 0.9`), sehingga `search` dan UI memakai α yang sama dengan laporan.

## 6. Kenapa bukan RRF?

*Reciprocal Rank Fusion* (RRF) hanya memakai peringkat, bukan skor. Bruch dkk. [18] menunjukkan bahwa kombinasi konveks
(linear) atas skor yang dinormalisasi mengungguli RRF dan hanya butuh sedikit data untuk menala satu-satunya parameternya,
sedangkan RRF sensitif terhadap parameternya. Sun dkk. [19] juga menunjukkan bahwa bobot linear yang mendekati optimal
bisa didapat dari sebagian kecil penilaian relevansi. Karena itu proyek ini memakai kombinasi linear dengan satu
parameter, α.

## Ringkasan

- Fusi = kandidat (top-100 semantik ∪ top-100 keyphrase positif) → min-max per jalur → α·sem + (1−α)·kp.
- Contoh 1185: dokumen relevan naik dari peringkat 3 ke 1 (0.9868 vs 0.9143) berkat kecocokan "kidney paired donation".
- α = 0.9 dipilih pada data penalaan: semantik dominan, keyphrase sebagai penentu kecil.
- Kombinasi linear dipilih berdasarkan [18] dan [19].

## Soal latihan

**Soal 1.** Kenapa normalisasi min-max dilakukan pada himpunan kandidat, bukan pada seluruh korpus?

**Soal 2.** Dengan angka di tabel contoh, hitung skor fusi dokumen 21203899 untuk α = 1. Apa artinya?

**Soal 3.** Apa yang terjadi pada fusi kalau sebuah kueri tidak punya satu pun unit keyphrase yang cocok dengan dokumen
mana pun?

**Soal 4.** Kenapa α = 0.9, bukan α = 0.5 yang terlihat "adil"?

## Kunci jawaban

**Jawaban 1.** Yang dibandingkan hanyalah dokumen kandidat, jadi rentang skor yang relevan adalah rentang di antara
kandidat. Normalisasi di seluruh korpus akan didominasi ribuan dokumen tidak relevan, sehingga perbedaan di antara
kandidat teratas menjadi sangat kecil.

**Jawaban 2.** α = 1 berarti $\tilde{s}_{sem}$ saja, yaitu 1.0. Artinya fusi menjadi sama dengan semantik saja (pada
kandidat), dan dokumen tidak relevan ini kembali berada di atas dokumen relevan.

**Jawaban 3.** Skor keyphrase semua dokumen nol, sehingga `positive_topk` tidak menambah kandidat dan `minmax`
menghasilkan nol untuk jalur keyphrase. Skor fusi = α × semantik ternormalisasi, dan urutannya sama dengan semantik saja.

**Jawaban 4.** α dipilih dari data, bukan dari intuisi. Pada kueri penalaan, nDCG@10 tertinggi ada di α = 0.9 untuk kedua
koleksi. Pada α = 0.5, nDCG@10 penalaan hanya 0.5450 (SciFact) dan 0.2933 (NFCorpus), karena sinyal keyphrase jarang dan
kasar sehingga bobot besar justru mengacaukan urutan.
