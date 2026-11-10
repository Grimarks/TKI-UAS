# Bab 06 — BM25 sebagai pembanding

## Tujuan belajar

1. Menuliskan dan menjelaskan rumus BM25 (persamaan (5) laporan), termasuk peran $k_1$ dan $b$.
2. Menghitung skor BM25 tangan untuk korpus kecil, lalu mencocokkannya dengan keluaran kode.
3. Menjelaskan cara implementasi proyek: bobot dihitung sekali, skor kueri = jumlah kolom.
4. Menjelaskan kenapa BM25 dipakai **hanya sebagai pembanding** dan apa artinya hasil BM25 sedikit lebih tinggi.

## 1. Rumus

$$s_{bm25}(q,d) = \sum_{t \in q} idf(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\frac{|d|}{avgdl}\right)}, \qquad idf(t) = \ln\left(1 + \frac{N - df(t) + 0.5}{df(t) + 0.5}\right)$$

- $f(t,d)$: berapa kali term $t$ muncul di dokumen $d$.
- $|d|$: panjang dokumen (jumlah token bermakna); $avgdl$: rata-rata panjang dokumen.
- $k_1 = 1.2$: mengatur **saturasi**. Kemunculan ke-10 sebuah kata tidak bernilai 10 kali kemunculan pertama.
- $b = 0.75$: mengatur **normalisasi panjang**. Dokumen panjang tidak otomatis menang hanya karena memuat banyak kata.
- $idf$: kata langka bernilai tinggi. Versi "+1" di dalam log menjamin idf selalu positif.

## 2. Contoh hitungan kecil (dicek dengan kode)

Korpus tiga dokumen, kueri **"heart attack"**:

| Dok | Teks | Token setelah stopword + stemming | $\lvert d \rvert$ |
|---|---|---|---|
| d1 | heart attack risk in young adults | heart, attack, risk, young, adult | 5 |
| d2 | myocardial infarction treatment after heart attack | myocardi, infarct, treatment, heart, attack | 5 |
| d3 | kidney transplant outcomes | kidney, transplant, outcom | 3 |

<!-- contoh dihitung ulang oleh scripts/modul.py --contoh (bm25_kecil) -->

Langkah 1, panjang rata-rata: $avgdl = (5+5+3)/3 = 4.333$.

Langkah 2, idf "heart" (muncul di 2 dari 3 dokumen):
$idf = \ln(1 + (3-2+0.5)/(2+0.5)) = \ln(1.6) = 0.4700$. Nilai idf "attack" sama.

Langkah 3, penyebut normalisasi untuk d1:
$k_1(1 - b + b \cdot 5/4.333) = 1.2 \times (0.25 + 0.8654) = 1.3385$.

Langkah 4, bobot satu term di d1 ($f = 1$):
$0.4700 \times \frac{1 \times 2.2}{1 + 1.3385} = 0.4422$.

Langkah 5, skor = jumlah dua term = $2 \times 0.4422 = 0.8843$.

`BM25("en").fit(...).scores(["heart attack"])` di kode menghasilkan **[0.8843, 0.8843, 0]**: d1 dan d2 sama (sama
panjang, sama-sama memuat kedua kata), dan d3 nol. Perhatikan d2: dokumen itu juga membahas *myocardial infarction*,
tetapi BM25 menilainya hanya karena kata "heart" dan "attack". Kalau d2 hanya menulis "myocardial infarction", skornya
akan nol (*vocabulary gap*, Bab 01).

## 3. Implementasi di proyek

Bobot setiap pasangan (dokumen, term) dihitung **sekali** saat BM25 pertama dibutuhkan, lalu disimpan di memori sebagai
matriks jarang N × V. Skor kueri tinggal menjumlahkan kolom-kolom term kueri.

<!-- kode: src/hybrid_ir/bm25.py#BM25.fit -->
*Kode asli: `src/hybrid_ir/bm25.py`, baris 44–66.*

```python
    def fit(self, texts: Sequence[str]) -> "BM25":
        """Hitung bobot BM25 untuk setiap (dokumen, term) dari seluruh korpus."""
        docs = [Counter(content_tokens(t, self.language)) for t in texts]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        vocab = sorted(df)
        self.term_index = {t: i for i, t in enumerate(vocab)}
        n = len(docs)
        lengths = np.array([sum(d.values()) for d in docs], dtype=np.float64)   # |d|
        avgdl = max(float(lengths.mean()) if n else 1.0, 1e-9)
        idf = bm25_idf(np.array([df[t] for t in vocab]), n)
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            # penyebut bagian kanan (5) yang hanya bergantung pada panjang dokumen
            norm = self.k1 * (1.0 - self.b + self.b * lengths[r] / avgdl)
            for term, tf in d.items():
                c = self.term_index[term]
                rows.append(r)
                cols.append(c)
                vals.append(float(idf[c]) * tf * (self.k1 + 1.0) / (tf + norm))
        self.matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
        return self
```
<!-- /kode -->

<!-- kode: src/hybrid_ir/bm25.py#BM25.scores -->
*Kode asli: `src/hybrid_ir/bm25.py`, baris 68–80.*

```python
    def scores(self, queries: Sequence[str]) -> np.ndarray:
        """Skor (Q, N) = jumlah kolom term query. Term query yang tidak ada di korpus diabaikan.

        Term yang muncul dua kali di query dihitung sekali (himpunan term), sesuai Σ_{t ∈ q}.
        """
        if self.matrix is None:
            raise RuntimeError("panggil fit() lebih dulu")
        out = np.zeros((len(queries), self.matrix.shape[0]), dtype=np.float32)
        for qi, q in enumerate(queries):
            cols = sorted({self.term_index[t] for t in content_tokens(q, self.language) if t in self.term_index})
            if cols:
                out[qi] = np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()
        return out
```
<!-- /kode -->

Beberapa detail penting:

- Tokenisasi, stopword, dan stemming sama dengan unit kata keyphrase (`text.content_tokens`), supaya perbandingannya
  adil.
- Term kueri dihitung sebagai **himpunan**: kata yang diulang di kueri tidak dihitung dua kali.
- Indeks BM25 tidak disimpan ke disk; membangunnya hanya sekitar 0.5–0.7 detik (`analisis/tabel_waktu.md`).
- Di mode `bm25` dan di evaluasi, dokumen dengan skor 0 (tidak ada kata yang sama) **tidak** ikut diperingkat
  (`positive_topk`).

## 4. Kenapa hanya pembanding?

Desain final sengaja menguji **semantik + keyphrase**. BM25 adalah metode leksikal klasik yang tidak butuh pelatihan
dan dikenal sebagai pembanding yang sulit dikalahkan (Ren dkk. [5]). Dengan menjadikannya pembanding terpisah, kita bisa
menjawab pertanyaan penelitian ketiga: seberapa dekat hybrid dengan BM25?

Hasilnya (Bab 09):

| | SciFact | NFCorpus |
|---|---|---|
| BM25 nDCG@10 | 0.6834 | 0.3278 |
| Hybrid nDCG@10 | 0.6671 | 0.3219 |
| Selisih hybrid − BM25 | −0.0163 (p = 0.354) | −0.0058 (p = 0.590) |

BM25 sedikit lebih tinggi, tetapi selisihnya **tidak signifikan**. Per kueri pun imbang: hybrid menang 65 dan kalah 64
di SciFact, serta menang 107 dan kalah 107 di NFCorpus. Hati-hati: "tidak signifikan" **bukan** berarti "terbukti
setara". Artinya, data kita tidak cukup kuat untuk menyimpulkan bahwa keduanya berbeda.

## Ringkasan

- BM25 = jumlah idf × tf tersaturasi, dengan normalisasi panjang; $k_1 = 1.2$, $b = 0.75$.
- Contoh tangan "heart attack" menghasilkan 0.8843, sama dengan kode.
- Implementasinya: matriks bobot di memori, skor = jumlah kolom term kueri.
- BM25 hanya pembanding. Ia sedikit lebih tinggi daripada hybrid, tetapi selisihnya tidak signifikan.

## Soal latihan

**Soal 1.** Apa fungsi $k_1$ dan $b$ dalam BM25?

**Soal 2.** Pada contoh tiga dokumen di atas, hitung idf "kidney" (muncul di 1 dari 3 dokumen).

**Soal 3.** Kenapa d1 dan d2 mendapat skor sama persis untuk kueri "heart attack"?

**Soal 4.** Apa maksud "BM25 lebih tinggi tetapi tidak signifikan"? Bolehkah kita bilang hybrid "setara" dengan BM25?

## Kunci jawaban

**Jawaban 1.** $k_1$ membatasi pengaruh frekuensi kata (saturasi), sedangkan $b$ mengatur seberapa kuat skor
dinormalkan terhadap panjang dokumen ($b = 0$ berarti tanpa normalisasi panjang, $b = 1$ berarti normalisasi penuh).

**Jawaban 2.** $idf = \ln(1 + (3-1+0.5)/(1+0.5)) = \ln(1 + 1.667) = \ln(2.667) = 0.9808$.

**Jawaban 3.** Keduanya memuat "heart" dan "attack" masing-masing sekali, dan panjangnya sama (5 token), sehingga setiap
suku rumusnya identik.

**Jawaban 4.** Rata-rata nDCG@10 BM25 sedikit lebih tinggi, tetapi uji *paired bootstrap* memberi p > 0.05 dan selang
kepercayaan 95% yang mencakup nol. Jadi perbedaannya bisa saja kebetulan. Kita **tidak** boleh menyebutnya "setara",
karena itu butuh uji kesetaraan tersendiri; yang benar adalah "tidak berbeda signifikan".
