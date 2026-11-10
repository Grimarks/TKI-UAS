# Bab 10 — Tur kode berkas per berkas

Bab ini menemani kamu membaca `src/hybrid_ir/` dari yang paling sederhana sampai yang paling kompleks. Tabel lengkap
"kotak Gambar 1 → berkas → fungsi → baris" ada di `docs/PETA_KODE.md`. Tabel itu dibangkitkan otomatis dari kode,
jadi nomor barisnya selalu benar, dan bisa dipakai saat presentasi untuk menunjuk kode setiap kotak.

## Tujuan belajar

1. Menyebutkan peran setiap berkas di `src/hybrid_ir/` dan kotak Gambar 1 yang dijalankannya.
2. Menelusuri urutan pemanggilan fungsi saat `hybrid-ir search` dijalankan.
3. Menunjukkan kode untuk setiap persamaan (1)–(9) di laporan.
4. Menjelaskan beberapa keputusan desain kode (validasi ketat, tanpa pickle, deterministik).

## 1. Peta berkas

| Urutan baca | Berkas | Peran (kotak Gambar 1) |
|---|---|---|
| 1 | `text.py` | Pra-pemrosesan; unit pencocokan keyphrase; token BM25 |
| 2 | `config.py` | membaca dan memvalidasi YAML (α, C, model, berkas keyphrase) |
| 3 | `data.py` | Korpus dokumen; Data uji BEIR |
| 4 | `encoders.py` | A. Encoder SBERT / A. Encode query |
| 5 | `keyphrases.py` | B. Keyphrase + bobot (BART-KPG) / B. Keyphrase query |
| 6 | `indexes.py` | Vector index, Keyphrase index; Skor semantik, Skor keyphrase |
| 7 | `bm25.py` | Pembanding: BM25 index, Skor BM25 |
| 8 | `fusion.py` | Normalisasi + Fusi; Ranking top-k |
| 9 | `retriever.py` | perakit Fase 1 (`build`) dan Fase 2 (`search`) |
| 10 | `evaluation.py` | Fase 3: Skenario, Tuning α, Metrik, Analisis |
| 11 | `cli.py`, `ui_app.py` | perintah `hybrid-ir` dan antarmuka Streamlit |
| — | `download.py` | mengunduh dataset BEIR dengan aman (persiapan) |

Setiap berkas dibuka dengan docstring berbahasa Indonesia yang menyebut kotaknya, masukan/keluarannya, dan rumusnya.
Bacalah docstring itu dulu sebelum membaca kodenya.

## 2. Urutan pemanggilan saat `hybrid-ir search` (mode hybrid)

```text
cli._cmd_search
├─ config.load_config                    baca YAML (alpha = 0.9, C = 100, top_k = 10)
├─ HybridRetriever.load                  muat indeks; cek encoder & pengaturan keyphrase cocok
└─ HybridRetriever.search → search_many
   ├─ _check_search_args                 top_k, mode, alpha dalam rentang
   ├─ _scores_for_mode → score_matrices
   │  ├─ validate_query                  normalize_text, tolak kueri kosong/terlalu panjang
   │  ├─ encoder.encode_queries → DenseIndex.scores          Jalur A, persamaan (1)
   │  └─ query_extractor.extract → KeyphraseIndex.scores     Jalur B, unit_weights → (4)
   └─ _hits_for_query
      ├─ fusion.rank_documents → fuse_weighted
      │  ├─ candidate_set                R = top-C(sem) ∪ top-C(kp > 0)
      │  └─ minmax (6), lalu α·sem + (1−α)·kp (7)
      └─ _make_hit                       judul, cuplikan, keyphrase yang cocok
```

Titik masuk CLI:

<!-- kode: src/hybrid_ir/cli.py#_cmd_search -->
*Kode asli: `src/hybrid_ir/cli.py`, baris 101–111.*

```python
def _cmd_search(args: argparse.Namespace) -> int:
    """Fase 2: muat indeks lalu cari satu query (opsi --mode/--alpha/--top-k menimpa konfigurasi)."""
    cfg = load_config(args.config)
    retriever = HybridRetriever.load(cfg)
    res = retriever.search(args.query, mode=args.mode, alpha=args.alpha, top_k=args.top_k)
    if args.json:
        print(json.dumps({"query": res.query, "mode": res.mode, "query_keyphrases": res.query_keyphrases,
                          "hits": [h.__dict__ for h in res.hits]}, ensure_ascii=False, indent=2))
    else:
        _print_result(res)
    return 0
```
<!-- /kode -->

Jalur yang dihitung bergantung pada mode. BM25 hanya dihitung di mode `bm25`:

<!-- kode: src/hybrid_ir/retriever.py#HybridRetriever._scores_for_mode -->
*Kode asli: `src/hybrid_ir/retriever.py`, baris 282–288.*

```python
    def _scores_for_mode(self, queries: Sequence[str], mode: str
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[Keyphrase]]]:
        """Hitung hanya jalur yang dibutuhkan mode ini; jalur lain diisi nol."""
        sem, kp, qkps = self.score_matrices(queries, need_semantic=mode not in ("keyphrase", "bm25"),
                                            need_keyphrase=mode not in ("semantic", "bm25"))
        bm25 = self.bm25_scores(queries) if mode == "bm25" else np.zeros_like(sem)
        return sem, kp, bm25, qkps
```
<!-- /kode -->

Skor jalur A dan B untuk banyak kueri sekaligus. Fungsi ini juga dipakai evaluasi supaya setiap kueri cukup dienkode
sekali:

<!-- kode: src/hybrid_ir/retriever.py#HybridRetriever.score_matrices -->
*Kode asli: `src/hybrid_ir/retriever.py`, baris 230–251.*

```python
    def score_matrices(self, queries: Sequence[str], need_semantic: bool = True,
                       need_keyphrase: bool = True) -> tuple[np.ndarray, np.ndarray, list[list[Keyphrase]]]:
        """Skor mentah semua dokumen untuk tiap query: (semantik Q × N, keyphrase Q × N, keyphrase query).

        Jalur A: A. Encode query → Skor semantik (persamaan (1)).
        Jalur B: B. Keyphrase query → Skor keyphrase (persamaan (4)).
        Dipakai langsung oleh evaluasi supaya query cukup di-encode/di-ekstrak SEKALI untuk semua
        skenario dan nilai alpha. Jalur yang tidak diminta diisi nol.
        """
        qs = [self.validate_query(q) for q in queries]
        n = len(self.doc_ids)
        if need_semantic:
            sem = self.dense.scores(self.encoder.encode_queries(qs))
        else:
            sem = np.zeros((len(qs), n), dtype=np.float32)
        if need_keyphrase:
            qkps = self.query_extractor.extract([TextItem(q) for q in qs], self.cfg.keyphrase.query_top_n)
            kp = self.kp_index.scores(qkps)
        else:
            qkps = [[] for _ in qs]
            kp = np.zeros((len(qs), n), dtype=np.float32)
        return sem, kp, qkps
```
<!-- /kode -->

Satu titik masuk untuk keempat mode:

<!-- kode: src/hybrid_ir/fusion.py#rank_documents -->
*Kode asli: `src/hybrid_ir/fusion.py`, baris 77–98.*

```python
def rank_documents(sem: np.ndarray, kp: np.ndarray, mode: str, alpha: float, candidate_k: int, top_k: int,
                   bm25: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Titik masuk tunggal: kembalikan (indeks_dokumen, skor) teratas untuk mode yang diminta.

    semantic  : urut menurut s_sem
    keyphrase : urut menurut s_kp, hanya dokumen dengan s_kp > 0
    bm25      : urut menurut s_bm25, hanya dokumen dengan s_bm25 > 0 (memerlukan argumen `bm25`)
    hybrid    : fusi linear (7) pada kandidat R
    """
    if mode == "bm25" and bm25 is None:
        raise ValueError("mode bm25 memerlukan skor BM25")
    if mode == "semantic":
        idx, vals = rank_single(sem, top_k)
    elif mode == "keyphrase":
        idx, vals = rank_single(kp, top_k, positive_only=True)
    elif mode == "bm25":
        idx, vals = rank_single(bm25, top_k, positive_only=True)
    elif mode == "hybrid":
        idx, vals = fuse_weighted(sem, kp, alpha, max(candidate_k, top_k))
    else:
        raise ValueError(f"mode tidak dikenal: {mode}")
    return idx[:top_k], vals[:top_k]
```
<!-- /kode -->

Penyusunan hasil, termasuk keyphrase dokumen yang "cocok" dengan kueri (yang tampil sebagai `cocok:` di CLI):

<!-- kode: src/hybrid_ir/retriever.py#HybridRetriever._make_hit -->
*Kode asli: `src/hybrid_ir/retriever.py`, baris 300–308.*

```python
    def _make_hit(self, rank: int, di: int, score: float, sem: float, kp: float, q_units: set[str],
                  bm25: float = 0.0) -> Hit:
        """Satu baris hasil. `matched_keyphrases` = keyphrase dokumen yang berbagi unit dengan query."""
        doc = self.documents[di]
        kcfg = self.cfg.keyphrase
        doc_kps = self.doc_keyphrases[di]
        matched = [p for p, _ in doc_kps if q_units & phrase_units(p, kcfg.language, kcfg.unit)]
        snippet = doc.text[:SNIPPET_CHARS] + ("..." if len(doc.text) > SNIPPET_CHARS else "")
        return Hit(rank, doc.id, doc.title, snippet, score, sem, kp, bm25, [p for p, _ in doc_kps[:10]], matched[:8])
```
<!-- /kode -->

## 3. Persamaan → kode

| Persamaan | Isi | Fungsi |
|---|---|---|
| (1) | cosine semantik | `indexes.DenseIndex.scores` |
| (2) | idf unit keyphrase | `indexes.keyphrase_idf` |
| (3) | bobot unit $b \cdot idf$ | `indexes.unit_weights`, `KeyphraseIndex.build` |
| (4) | cosine keyphrase | `indexes.KeyphraseIndex._cosine_row` |
| (5) | BM25 | `bm25.bm25_idf`, `BM25.fit`, `BM25.scores` |
| (6) | min-max | `fusion.minmax` |
| (7) | fusi linear | `fusion.fuse_weighted` |
| (8) | DCG | `evaluation._dcg` |
| (9) | nDCG | `evaluation.ndcg_at_k` |

Nomor baris setiap fungsi ada di `docs/PETA_KODE.md`.

## 4. Keputusan desain yang layak diceritakan

**Validasi konfigurasi ketat.** Kunci YAML yang tidak dikenal ditolak. Salah ketik seperti `alfa:` tidak akan diam-diam
membuat sistem memakai nilai bawaan. Konfigurasi dari desain lama (YAKE, RRF, `hybrid_bm25`) ditolak dengan pesan
yang menjelaskan sebabnya.

<!-- kode: src/hybrid_ir/config.py#_build_section -->
*Kode asli: `src/hybrid_ir/config.py`, baris 190–199.*

```python
def _build_section(cls: type, name: str, raw: Any):
    if raw is None:
        raw = {}
    _check(isinstance(raw, dict), f"bagian '{name}' harus berupa pemetaan (key: value)")
    for key, msg in LEGACY_KEYS.get(name, {}).items():
        _check(key not in raw, msg)
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    _check(not unknown, f"kunci tidak dikenal di '{name}': {unknown}. Kunci yang valid: {sorted(allowed)}")
    return cls(**raw)
```
<!-- /kode -->

**Indeks harus cocok dengan konfigurasi.** `HybridRetriever.load` membandingkan sidik jari encoder dan pengaturan
keyphrase di `meta.json`. Vektor dari encoder yang berbeda tidak sebanding, sehingga lebih baik berhenti dengan pesan
jelas daripada menghasilkan peringkat yang tidak bermakna.

**Tanpa pickle.** Indeks disimpan sebagai `.npy`, `.npz`, dan `.json`, lalu dimuat dengan `allow_pickle=False`, karena
memuat pickle dari berkas yang tidak tepercaya bisa menjalankan kode sembarang.

**Deterministik.** Seri skor diputus menurut indeks dokumen (`topk_indices`), dan bootstrap memakai seed 13. Satu
pengecualian kecil dicatat di README: skor keyphrase bisa berbeda di digit float32 terakhir antar-proses, karena urutan
iterasi `set` di `phrase_units`. Metrik akhir tetap identik; pakai `PYTHONHASHSEED=0` bila perlu run yang benar-benar
deterministik.

**Tes.** `tests/` berisi 157 tes. Tes memakai encoder pengganti (`HashingEncoder`) dan data mini `data/demo`, sehingga
bisa jalan tanpa unduhan model.

## Ringkasan

- `retriever.py` adalah perakit; logika setiap kotak ada di berkasnya masing-masing.
- `search`: validasi → skor per jalur → fusi → hasil. `evaluate`: split → skor sekali → sapuan α → skenario →
  bootstrap.
- Tabel peta kode dibangkitkan dari kode (`scripts/peta_kode.py`), jadi aman dipakai di presentasi.

## Soal latihan

**Soal 1.** Fungsi apa yang menghitung persamaan (7), dan di berkas mana?

**Soal 2.** Saat mode `semantic`, apakah keyphrase kueri dibangkitkan? Tunjukkan baris kode yang menentukannya.

**Soal 3.** Kenapa evaluasi memanggil `score_matrices` langsung, bukan `search` berkali-kali?

**Soal 4.** Apa yang terjadi kalau di YAML tertulis `alfa: 0.9` (salah ketik)?

## Kunci jawaban

**Jawaban 1.** `fuse_weighted` di `src/hybrid_ir/fusion.py`.

**Jawaban 2.** Tidak. Di `_scores_for_mode`, `need_keyphrase=mode not in ("semantic", "bm25")` bernilai False, sehingga
`score_matrices` mengisi skor keyphrase dengan nol dan tidak memanggil extractor.

**Jawaban 3.** Supaya setiap kueri cukup dienkode dan diekstrak keyphrase-nya **sekali**, lalu skor yang sama dipakai
ulang untuk keempat skenario dan 11 nilai α. Memanggil `search` berkali-kali akan mengulang pekerjaan mahal itu.

**Jawaban 4.** `_build_section` menolak kunci yang tidak dikenal dan melempar `ConfigError` berisi daftar kunci yang
valid. CLI menampilkannya sebagai pesan "Galat: …" dengan kode keluar 2, tanpa *traceback*.
