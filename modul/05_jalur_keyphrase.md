# Bab 05 — Jalur B: keyphrase BART-KPG

## Tujuan belajar

1. Menjelaskan apa itu keyphrase, beda *ekstraksi* dan *pembangkitan*, serta keyphrase *present* dan *absent*.
2. Menjelaskan alur BART-KPG: masukan "judul`<s>`abstrak", keluaran "kp1;kp2;…", dijalankan di Kaggle 2×T4.
3. Menghitung bobot urutan $1/(1+0.25i)$, unit pencocokan, IDF (2), dan bobot unit (3).
4. Menjelaskan skor keyphrase sebagai cosine (4).
5. Membaca statistik keyphrase proyek dan menjelaskan keterbatasannya dengan jujur.

## 1. Keyphrase: diekstraksi atau dibangkitkan

Keyphrase adalah frasa pendek yang merangkum isi dokumen (Sarwar dkk. [7]).

- **Ekstraksi** (misalnya KEA, YAKE, TopicRank) hanya bisa memilih frasa yang tertulis di teks.
- **Pembangkitan** memakai model *sequence-to-sequence* seperti BART (Lewis dkk. [9], Kulkarni dkk. [10]), sehingga
  secara prinsip bisa menghasilkan frasa yang **tidak** tertulis di teks.

Keyphrase yang muncul di teks disebut **present**; yang tidak muncul disebut **absent**.

Model proyek ini adalah checkpoint publik `taln-ls2n/bart-base-kp20k`: BART-base yang dilatih pada **KP20k**, yaitu
judul dan abstrak makalah **ilmu komputer**. Sementara itu, koleksi uji kita berdomain biomedis. Perpindahan domain
seperti ini dicatat Boudin dkk. [11] sebagai salah satu kesulitan.

## 2. Pembangkitan di Kaggle

Membangkitkan keyphrase untuk sekitar 6.300–6.800 teks (dokumen + kueri) itu berat, jadi dijalankan di Kaggle dengan
2× GPU T4, satu proses per GPU (`kaggle/generate_keyphrases.py`). Setiap proses selesai dalam 91–132 detik. Hasilnya:

```json
{"key": "000bfc9380a7ded117acf00f492d21ad5b929ec0", "keyphrases": [["amyloidosis", 1.0], ["magnetic resonance imaging", 0.8]]}
```

`key` adalah SHA-1 dari teks "judul. abstrak" yang sudah dinormalisasi (`text.text_fingerprint`). Dengan cara ini,
saat `build` di laptop, setiap dokumen dicari keyphrase-nya lewat SHA-1 teksnya. Kalau aturan normalisasi berubah,
kuncinya tidak cocok lagi, dan `tests/test_kaggle_scripts.py` menjaga agar skrip Kaggle dan kode lokal tetap sama.

Keluaran mentah model berbentuk `"kidney paired donation;optimization;matching algorithms"`. Teks ini dipecah dan
dibersihkan (duplikat dan frasa terlalu panjang dibuang), lalu diberi bobot menurut urutan:

<!-- kode: src/hybrid_ir/keyphrases.py#rank_weights -->
*Kode asli: `src/hybrid_ir/keyphrases.py`, baris 76–78.*

```python
def rank_weights(n: int) -> list[float]:
    """Bobot berdasarkan urutan keluaran generator: 1.0, 0.8, 0.67, 0.57, ... (1 / (1 + 0.25 i))."""
    return [1.0 / (1.0 + 0.25 * i) for i in range(n)]
```
<!-- /kode -->

<!-- kode: src/hybrid_ir/keyphrases.py#parse_generated_keyphrases -->
*Kode asli: `src/hybrid_ir/keyphrases.py`, baris 81–100.*

```python
def parse_generated_keyphrases(text: str, max_phrases: int, sep: str = ";",
                               max_words: int = 8, max_chars: int = 80) -> list[Keyphrase]:
    """Ubah keluaran generator 'a;b;c' menjadi [(a, w1), (b, w2), ...].

    Frasa kosong, terlalu panjang, atau duplikat (tanpa memandang huruf besar/kecil) dibuang.
    """
    seen: set[str] = set()
    phrases: list[str] = []
    for part in text.split(sep):
        phrase = normalize_text(part)
        if not phrase or len(phrase) > max_chars or len(phrase.split()) > max_words:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return list(zip(phrases, rank_weights(len(phrases))))
```
<!-- /kode -->

Bobot untuk urutan ke-0 sampai ke-5: 1.0, 0.8, 0.667, 0.571, 0.5, 0.444. Frasa pertama dianggap paling penting.
Dokumen dibatasi paling banyak 10 frasa, kueri 8.

## 3. Dari keyphrase ke unit pencocokan

Keyphrase tidak dicocokkan sebagai string mentah. Setiap keyphrase dipecah menjadi **unit**: frasa utuh yang sudah
dinormalisasi, ditambah setiap kata bermaknanya. Konfigurasi final memakai `unit: both`.

<!-- kode: src/hybrid_ir/text.py#phrase_units -->
*Kode asli: `src/hybrid_ir/text.py`, baris 136–153.*

```python
def phrase_units(phrase: str, lang: str, unit: str) -> set[str]:
    """Unit pencocokan sebuah keyphrase (dasar dari u pada persamaan (2) dan (3)).

    unit='phrase' -> {'p:<frasa ternormalisasi>'}
    unit='token'  -> {'t:<kata>' ...} untuk tiap kata bermakna
    unit='both'   -> gabungan keduanya (konfigurasi final)

    Contoh (unit='both'): 'heart attacks' -> {'p:heart attack', 't:heart', 't:attack'}.
    """
    units: set[str] = set()
    if unit in ("phrase", "both"):
        norm = normalize_phrase(phrase, lang)
        if norm:
            units.add("p:" + norm)
    if unit in ("token", "both"):
        for tok in content_tokens(phrase, lang):
            units.add("t:" + tok)
    return units
```
<!-- /kode -->

Contoh dari fungsi itu:

| Keyphrase | Frasa ternormalisasi | Kata |
|---|---|---|
| Kidney Paired Donation | kidney pair donat | kidney, pair, donat |
| heart attacks | heart attack | heart, attack |
| the neural networks | neural network | neural, network |
| betel-nut chewing | betel nut chew | betel, nut, chew |

Normalisasi frasa memakai huruf kecil, membuang stopword di **tepi**, lalu melakukan stemming Snowball. Karena itu
"Kidney Paired Donation" dan "kidney paired donation" dianggap sama. Kenapa memakai frasa **dan** kata? Frasa utuh
memberi kecocokan yang tepat, sedangkan kata memberi kecocokan sebagian. Pada kueri NFCorpus "betel nuts", tidak ada
frasa yang sama utuh dengan keyphrase dokumen "betel-nut chewing", tetapi kata "betel" dan "nuts" tetap cocok.

Kalau satu unit muncul di beberapa keyphrase, bobotnya adalah bobot keyphrase **tertinggi** yang memuatnya ($b_{d,u}$).
Aturan ini dipakai sama persis untuk dokumen dan kueri:

<!-- kode: src/hybrid_ir/indexes.py#unit_weights -->
*Kode asli: `src/hybrid_ir/indexes.py`, baris 74–85.*

```python
def unit_weights(keyphrases: Sequence[Keyphrase], language: str, unit: str) -> dict[str, float]:
    """Pecah keyphrase menjadi unit pencocokan; tiap unit diberi bobot keyphrase TERTINGGI yang memuatnya.

    Ini b_{d,u} pada persamaan (3). Dipakai sama persis untuk dokumen (saat build) dan query (saat search).
    Contoh: [('heart attack', 1.0), ('attack rate', 0.8)] -> t:attack mendapat 1.0, bukan 0.8.
    """
    weights: dict[str, float] = {}
    for phrase, w in keyphrases:
        for u in phrase_units(phrase, language, unit):
            if w > weights.get(u, 0.0):
                weights[u] = float(w)
    return weights
```
<!-- /kode -->

## 4. IDF, bobot, dan cosine

Unit yang muncul di banyak dokumen (misalnya "cell") kurang informatif. Karena itu bobot dikalikan IDF, persamaan (2)
dan (3) di laporan:

$$idf(u) = \ln\frac{N+1}{df(u)+1} + 1, \qquad w_{d,u} = b_{d,u} \cdot idf(u)$$

<!-- kode: src/hybrid_ir/indexes.py#keyphrase_idf -->
*Kode asli: `src/hybrid_ir/indexes.py`, baris 88–90.*

```python
def keyphrase_idf(doc_freq: np.ndarray, n_docs: int) -> np.ndarray:
    """Persamaan (2): idf(u) = ln((N + 1) / (df(u) + 1)) + 1 (versi 'smooth', selalu > 0)."""
    return np.log((n_docs + 1) / (doc_freq + 1)) + 1.0
```
<!-- /kode -->

Contoh nyata dari indeks SciFact (N = 5.183):

| Unit | df | idf |
|---|---|---|
| frasa "kidney paired donation" | 3 | 8.1670 |
| kata "kidney" | 39 | 5.8645 |
| kata "cell" | 961 | 2.6843 |

Frasa langka bernilai paling tinggi. Setiap baris vektor dokumen lalu dinormalkan L2, dan skor kueri–dokumen adalah
cosine, persamaan (4):

$$s_{kp}(q,d) = \frac{w_q \cdot w_d}{\lVert w_q \rVert \, \lVert w_d \rVert}$$

<!-- kode: src/hybrid_ir/indexes.py#KeyphraseIndex._cosine_row -->
*Kode asli: `src/hybrid_ir/indexes.py`, baris 163–172.*

```python
    def _cosine_row(self, known: list[tuple[int, float]]) -> np.ndarray | None:
        """Persamaan (4) untuk satu query: w_q = b_{q,u}·idf(u), dinormalkan L2, lalu M @ w_q."""
        cols = np.array([c for c, _ in known], dtype=np.int64)
        vals = np.array([w * float(self.idf[c]) for c, w in known], dtype=np.float32)
        norm = float(np.linalg.norm(vals))
        if norm == 0.0:
            return None
        qvec = sp.csr_matrix((vals / norm, (np.zeros_like(cols), cols)),
                             shape=(1, self.matrix.shape[1]), dtype=np.float32)
        return (self.matrix @ qvec.T).toarray().ravel()
```
<!-- /kode -->

Unit kueri yang tidak ada di kosakata indeks diabaikan. Kalau tidak ada satu pun unit kueri yang dikenal, skor
keyphrase kueri itu nol untuk semua dokumen.

## 5. Statistik keyphrase proyek (jujur apa adanya)

Dari `analisis/tabel_keyphrase.md`:

| Statistik | SciFact | NFCorpus |
|---|---|---|
| Rata-rata keyphrase per dokumen (batas 10) | 4.44 | 4.33 |
| Rata-rata keyphrase per kueri test | 2.92 | 1.89 |
| Keyphrase dokumen *absent* | 5.4% | 4.5% |
| Keyphrase kueri *absent* | 14.7% | 28.5% |
| Pasangan kueri–dokumen relevan dengan skor keyphrase > 0 | 80.5% | 14.0% |
| Kueri dengan ≥ 1 dokumen relevan berskor keyphrase > 0 | 81.7% | 71.2% |
| Kueri dengan nDCG@10 keyphrase saja = 0 | 44.0% | 41.2% |

Cara membacanya:

- **BART-KPG hampir selalu menyalin** frasa dari teks: hanya sekitar 5% keyphrase dokumen yang *absent*. Jadi jalur
  keyphrase di proyek ini terutama berperan sebagai **penanda istilah inti berbobot**, bukan pencipta sinonim baru.
- Keyphrase kueri lebih sering *absent* (14.7% dan 28.5%), tetapi **belum diukur** apakah itu membantu mengatasi
  *vocabulary gap*. Jangan mengklaimnya saat presentasi.
- Angka 14.0% per pasangan di NFCorpus terlihat buruk, tetapi itu karena qrels NFCorpus padat (rata-rata 38.2 dokumen
  relevan per kueri). Kalau dihitung per kueri, 71.2% kueri NFCorpus tetap punya minimal satu dokumen relevan yang
  "tersentuh" sinyal keyphrase.
- Kualitas keyphrase **tidak diukur langsung**, karena tidak ada keyphrase acuan. Dampak perpindahan domain (ilmu
  komputer ke biomedis) hanya terlihat secara tidak langsung, misalnya dari jumlah keyphrase yang sedikit dan kasus
  *kohlrabi* di Bab 09.

Contoh keyphrase nyata: makalah SciFact tentang STAR (perangkat lunak RNA-seq) mendapat 'rna-seq', 'alignment',
'splicing', 'canonical junctions', 'intergenic splices'.

## Ringkasan

- Keyphrase dibangkitkan BART-KPG di Kaggle, disimpan dengan kunci SHA-1 teks, dan dibaca lokal.
- Bobot urutan $1/(1+0.25i)$; keyphrase dipecah menjadi unit frasa + kata; bobot unit × IDF; skor = cosine.
- Keyphrase dokumen sekitar 95% *present*, jadi perannya sebagai penguat istilah inti.
- Keyphrase saja lemah (nDCG@10 0.4078 dan 0.2321) karena sinyalnya jarang, tetapi berguna sebagai pelengkap
  embedding (Bab 07).

## Soal latihan

**Soal 1.** Apa beda keyphrase *present* dan *absent*? Berapa proporsi *absent* keyphrase dokumen SciFact?

**Soal 2.** Hitung bobot urutan keyphrase ketiga (urutan ke-2, dihitung dari nol).

**Soal 3.** Tuliskan unit pencocokan untuk keyphrase "Heart Attacks" dengan `unit: both`.

**Soal 4.** Kenapa unit "kidney paired donation" punya idf lebih tinggi daripada "cell"?

**Soal 5.** Seorang teman bilang, "keyphrase absent di kueri terbukti mengatasi vocabulary gap." Apakah klaim itu
didukung data proyek?

## Kunci jawaban

**Jawaban 1.** *Present* muncul utuh di teks (setelah normalisasi dan stemming yang sama), *absent* tidak. Di SciFact,
5.4% keyphrase dokumen bersifat *absent*.

**Jawaban 2.** $1/(1+0.25 \cdot 2) = 1/1.5 = 0.667$.

**Jawaban 3.** Frasa `heart attack` (setelah huruf kecil dan stemming "attacks" menjadi "attack"), ditambah kata
`heart` dan `attack`. Di kode: `{'p:heart attack', 't:heart', 't:attack'}`.

**Jawaban 4.** IDF besar untuk unit yang muncul di sedikit dokumen. "kidney paired donation" hanya ada di 3 dari 5.183
dokumen (idf 8.1670), sedangkan "cell" ada di 961 dokumen (idf 2.6843). Unit langka lebih membedakan dokumen.

**Jawaban 5.** Tidak. Proyek hanya mengukur **proporsinya** (14.7% dan 28.5%), bukan dampaknya terhadap kualitas
pencarian. Klaim itu belum didukung data, dan laporan menyatakannya secara eksplisit.
