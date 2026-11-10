"""Utilitas teks: normalisasi, tokenisasi, stemming, unit pencocokan keyphrase, dan sidik jari teks.

Peran di pipeline (Gambar 1)
    Fase 1 dan Fase 2 — kotak "Pra-pemrosesan" (sama untuk dokumen dan query).
    Fase 1 — Jalur B: bagian "Keyphrase index" (`phrase_units` memecah keyphrase menjadi unit).
    Pembanding BM25: kotak "Token + stemming" / "Token query" (`content_tokens`).

Masukan  : teks mentah (judul, abstrak, query, atau satu keyphrase).
Keluaran : teks ternormalisasi, daftar token, himpunan unit pencocokan, atau SHA-1 teks.

Modul ini sengaja kecil dan tanpa efek samping supaya mudah dipahami dan diuji. Semua bagian lain
(indeks keyphrase, BM25, cache keyphrase dari Kaggle) memakai fungsi di sini, sehingga
pemrosesan dokumen dan query dijamin sama.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache

_WS = re.compile(r"\s+")
# Karakter kontrol (selain tab/newline yang akan diratakan oleh _WS) dibuang.
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
# Token = deretan huruf/angka unicode. Tanda hubung memisahkan token ("graph-based" -> graph, based).
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

_EN_STOP = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves also may might shall via using use used based within without among across
    however thus therefore whether""".split()
)
_ID_STOP = frozenset(
    """yang dan di ke dari pada untuk dengan atau ini itu adalah ialah dalam oleh sebagai juga akan telah
    sudah belum tidak bukan saya kami kita mereka dia ia anda para sebuah suatu satu dua tiga karena
    sehingga agar supaya jika bila apabila maka namun tetapi serta bahwa hanya dapat bisa lebih
    sangat sering antara terhadap secara pun lah kah nya""".split()
)


# ------------------------------------------------------------------ Pra-pemrosesan (Fase 1 & 2)
def normalize_text(text: str | None) -> str:
    """Normalisasi dasar: Unicode NFKC, buang karakter kontrol, ratakan spasi."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def make_full_text(title: str | None, body: str | None) -> str:
    """Gabungkan judul dan isi menjadi satu teks dokumen: 'judul. isi'.

    Aturan ini juga dipakai skrip Kaggle (kaggle/generate_keyphrases.py). Jangan diubah
    sepihak, karena kunci cache keyphrase dihitung dari teks hasil fungsi ini.
    """
    title = normalize_text(title)
    body = normalize_text(body)
    if title and body:
        sep = "" if title.endswith((".", "?", "!")) else "."
        return f"{title}{sep} {body}"
    return title or body


def text_fingerprint(text: str) -> str:
    """SHA-1 dari teks yang sudah dinormalisasi. Dipakai sebagai kunci cache (bukan untuk keamanan)."""
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ token, stopword, stemming
def get_stopwords(lang: str) -> frozenset[str]:
    """Daftar stopword untuk bahasa `lang` ('id' = Indonesia; selain itu Inggris)."""
    return _ID_STOP if lang == "id" else _EN_STOP


@lru_cache(maxsize=8)
def _stemmer_for(lang: str):
    if lang in ("none", ""):
        return None
    import snowballstemmer

    name = {"en": "english", "id": "indonesian"}.get(lang)
    return snowballstemmer.stemmer(name) if name else None


@lru_cache(maxsize=400_000)
def _stem_word(lang: str, word: str) -> str:
    stemmer = _stemmer_for(lang)
    return stemmer.stemWord(word) if stemmer else word


def stem(word: str, lang: str) -> str:
    """Stem Snowball satu kata ('networks' -> 'network'); `lang='none'` mengembalikan kata apa adanya."""
    return _stem_word(lang, word)


def tokenize(text: str) -> list[str]:
    """Pisah teks menjadi token huruf kecil."""
    return _TOKEN.findall(normalize_text(text).lower())


def content_tokens(text: str, lang: str = "en", stemmed: bool = True) -> list[str]:
    """Token bermakna: bukan stopword, mengandung huruf, panjang >= 2, (opsional) di-stem.

    Dipakai BM25 (pembanding) dan unit token keyphrase, sehingga keduanya memakai aturan yang sama.
    """
    stop = get_stopwords(lang)
    out: list[str] = []
    for tok in tokenize(text):
        if len(tok) < 2 or tok in stop or not any(c.isalpha() for c in tok):
            continue
        out.append(stem(tok, lang) if stemmed else tok)
    return out


# ------------------------------------------------------------------ unit pencocokan keyphrase (Jalur B)
def normalize_phrase(phrase: str, lang: str = "en") -> str:
    """Bentuk kanonik sebuah frasa: huruf kecil, stopword di tepi dibuang, tiap kata di-stem.

    'The Neural Networks' -> 'neural network'. Frasa yang semuanya stopword menjadi ''.
    """
    stop = get_stopwords(lang)
    toks = tokenize(phrase)
    while toks and toks[0] in stop:
        toks.pop(0)
    while toks and toks[-1] in stop:
        toks.pop()
    return " ".join(stem(t, lang) for t in toks)


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
