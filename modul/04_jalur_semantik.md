# Bab 04 — Jalur A: pencarian semantik dengan embedding

## Tujuan belajar

1. Menjelaskan apa itu *embedding* dan *bi-encoder*.
2. Menjelaskan model all-MiniLM-L6-v2 (dimensi 384, panjang masukan 256 token).
3. Menjelaskan kenapa vektor dinormalkan L2, sehingga cosine sama dengan perkalian dalam.
4. Menjelaskan pencarian sebagai **perkalian matriks** dan menunjuk kodenya.

## 1. Embedding dan bi-encoder

*Embedding* adalah vektor angka yang mewakili makna teks. Teks yang maknanya mirip menghasilkan vektor yang
arahnya mirip. Model yang dipakai adalah **all-MiniLM-L6-v2** dari pustaka Sentence-Transformers (keluarga SBERT):

- keluarannya 384 angka per teks;
- masukan dipotong maksimal 256 token (`encoder.max_seq_length` di konfigurasi);
- ringan (6 lapis transformer), sehingga cukup cepat di laptop.

Arsitekturnya **bi-encoder**: kueri dan dokumen dienkode **terpisah** oleh model yang sama. Keuntungannya, dokumen cukup
dienkode sekali saat `build`; saat pencarian hanya kueri yang perlu dienkode (Zhao dkk. [4]).

## 2. Normalisasi L2 dan cosine

Kemiripan kosinus, persamaan (1) di laporan:

$$s_{sem}(q,d) = \cos(e_q, e_d) = \frac{e_q \cdot e_d}{\lVert e_q \rVert \, \lVert e_d \rVert}$$

Kalau setiap vektor dibagi dengan panjangnya sendiri (normalisasi L2), maka $\lVert e \rVert = 1$, dan cosine tinggal
$e_q \cdot e_d$ (perkalian dalam biasa). Contoh nyata: ketiga vektor "heart attack", "myocardial infarction", dan
"kidney transplant" punya norma 1.0 setelah dienkode, dengan 384 dimensi.

<!-- kode: src/hybrid_ir/encoders.py#_l2_normalize -->
*Kode asli: `src/hybrid_ir/encoders.py`, baris 55–59.*

```python
def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Bagi tiap baris dengan normanya (batas bawah 1e-12 agar vektor nol tidak membuat pembagian nol)."""
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)
```
<!-- /kode -->

Batas bawah `1e-12` mencegah pembagian dengan nol kalau ada vektor nol.

<!-- kode: src/hybrid_ir/encoders.py#SentenceTransformerEncoder._encode -->
*Kode asli: `src/hybrid_ir/encoders.py`, baris 132–143.*

```python
    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, getattr(self, "dim", 1)), dtype=np.float32)
        batch = [prefix + t for t in texts]
        emb = self.model.encode(
            batch,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(batch) > 512,
        )
        return _l2_normalize(emb)
```
<!-- /kode -->

## 3. Pencarian = perkalian matriks

Semua vektor dokumen disusun menjadi matriks $V$ berukuran N × 384 (SciFact: 5.183 × 384). Untuk Q kueri dengan
matriks $E_q$ (Q × 384), skor semua pasangan dihitung sekaligus:

$$S = E_q V^\top \quad \text{(Q × N)}$$

<!-- kode: src/hybrid_ir/indexes.py#DenseIndex.scores -->
*Kode asli: `src/hybrid_ir/indexes.py`, baris 53–58.*

```python
    def scores(self, query_vecs: np.ndarray) -> np.ndarray:
        """Persamaan (1) untuk semua pasangan sekaligus: (Q × d) @ (d × N) = skor cosine (Q, N)."""
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"dimensi query {q.shape} tidak cocok dengan indeks (dim={self.dim})")
        return q @ self.matrix.T
```
<!-- /kode -->

Pencarian ini **eksak**: setiap dokumen dihitung skornya, tanpa pendekatan tetangga terdekat (ANN). Untuk beberapa
ribu dokumen ini cepat sekali. Rata-rata waktu pencarian semantik per kueri di MacBook M4 Pro adalah sekitar 4 ms
(`analisis/tabel_waktu.md`), sebagian besar untuk mengenkode kueri.

## 4. Contoh angka dari proyek

| Pasangan | Cosine |
|---|---|
| "heart attack" vs "myocardial infarction" | 0.5936 |
| "heart attack" vs "kidney transplant" | 0.2623 |

Pada kueri SciFact 1185 (*kidney paired donation*), dokumen relevan 16737210 mendapat skor semantik 0.5918, sedikit di
bawah dokumen 21203899 (0.6013) yang membahas pendanaan pengiriman ginjal donor. Semantik saja menaruh dokumen relevan
di peringkat 3. Bab 07 menunjukkan bagaimana keyphrase memperbaikinya.

## 5. Hasil jalur semantik

| | SciFact | NFCorpus |
|---|---|---|
| nDCG@10 Semantic-only | 0.6497 | 0.3174 |
| nDCG@10 BM25 (pembanding) | 0.6834 | 0.3278 |

Semantik saja masih di bawah BM25 di kedua koleksi. Ini sejalan dengan temuan Ren dkk. [5] bahwa *dense retriever*
*zero-shot* sering kalah dari BM25. Recall@100-nya justru lebih tinggi daripada BM25 (0.9250 vs 0.9193 di SciFact;
0.3111 vs 0.2484 di NFCorpus): embedding pandai "mengumpulkan" dokumen yang mirip makna, tetapi kurang tajam menaruh
dokumen yang tepat di urutan paling atas.

## Ringkasan

- MiniLM mengubah teks menjadi vektor 384 dimensi; kueri dan dokumen dienkode terpisah (bi-encoder).
- Setelah normalisasi L2, cosine = perkalian dalam, sehingga skor semua dokumen cukup dihitung dengan satu perkalian
  matriks.
- Semantik saja: nDCG@10 0.6497 (SciFact) dan 0.3174 (NFCorpus), masih di bawah BM25 tetapi dengan Recall@100 lebih
  tinggi.

## Soal latihan

**Soal 1.** Kenapa vektor perlu dinormalkan L2 sebelum disimpan?

**Soal 2.** Berapa ukuran matriks hasil `DenseIndex.scores` untuk 300 kueri test SciFact?

**Soal 3.** Apa keuntungan bi-encoder dibanding mengenkode pasangan kueri–dokumen bersama-sama?

**Soal 4.** Semantik saja punya Recall@100 lebih tinggi dari BM25 tetapi nDCG@10 lebih rendah. Apa artinya?

## Kunci jawaban

**Jawaban 1.** Supaya cosine cukup dihitung sebagai perkalian dalam ($e_q \cdot e_d$), yang bisa dilakukan sekaligus
untuk semua dokumen dengan perkalian matriks. Normalisasi juga membuat panjang teks tidak memengaruhi skor.

**Jawaban 2.** (300 × 384) dikali (384 × 5.183) menghasilkan **300 × 5.183**: satu baris skor untuk tiap kueri.

**Jawaban 3.** Dokumen hanya dienkode sekali saat `build`. Saat pencarian, hanya kueri yang dienkode, lalu skor dihitung
dengan perkalian matriks. Mengenkode setiap pasangan kueri–dokumen akan butuh 5.183 kali jalan model per kueri.

**Jawaban 4.** Embedding berhasil membawa banyak dokumen relevan ke 100 besar, tetapi urutan di 10 teratas kurang tepat.
Dokumen yang mirip makna tetapi tidak relevan sering berada di atas dokumen yang benar.
