"""Antarmuka web lokal (Streamlit) untuk mencoba pencarian hibrida.

Peran di pipeline (Gambar 1)
    Fase 2 — kotak "Ranking top-k" (hasil di UI Streamlit). Tab "Pencarian" menjalankan satu mode;
    tab "Bandingkan mode" menampilkan BM25 | Semantic | Keyphrase | Hybrid berdampingan (empat skenario).

Masukan  : query dari pengguna, mode, α (slider; bawaan = retrieval.alpha di konfigurasi), jumlah hasil.
Keluaran : daftar dokumen dengan skor tiap jalur, keyphrase dokumen, dan keyphrase yang cocok dengan query.

Dijalankan lewat `hybrid-ir ui --config <yaml>` (yang meneruskan lokasi konfigurasi lewat variabel
lingkungan HYBRID_IR_CONFIG). Pertimbangan keamanan:
  * Server hanya mendengarkan 127.0.0.1 (lihat cli.py), tidak terbuka ke jaringan.
  * Tidak ada unggah berkas dan tidak ada input path dari pengguna; lokasi indeks hanya dari konfigurasi.
  * Tidak memakai unsafe_allow_html. Semua teks dari dokumen/query di-escape agar Markdown/HTML di
    dalam data tidak ikut dirender (mis. gambar eksternal atau tautan tersembunyi).
"""
from __future__ import annotations

import os
import re
import threading

import streamlit as st

from hybrid_ir.config import ConfigError, load_config
from hybrid_ir.keyphrases import MissingPrecomputed
from hybrid_ir.retriever import Hit, HybridRetriever, IndexMismatchError, QueryError, SearchResult

MODE_TITLES = {"hybrid": "Hybrid", "semantic": "Semantic saja", "keyphrase": "Keyphrase saja", "bm25": "BM25 saja"}
COMPARE_MODES = ("bm25", "semantic", "keyphrase", "hybrid")
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$&])")


def esc(text: str) -> str:
    """Escape karakter khusus Markdown agar teks data tampil apa adanya."""
    return _MD_SPECIAL.sub(r"\\\1", text or "")


def chips(items: list[str]) -> str:
    """Tampilkan daftar keyphrase sebagai `kode` dipisah koma."""
    return ", ".join(f"`{i.replace('`', '')}`" for i in items) if items else "-"


@st.cache_resource(show_spinner="Memuat indeks dan model ...")
def load_retriever(config_path: str) -> HybridRetriever:
    """Muat indeks dan model SEKALI untuk semua sesi (di-cache oleh Streamlit)."""
    cfg = load_config(config_path)
    return HybridRetriever.load(cfg)


@st.cache_resource
def model_lock() -> threading.Lock:
    """Satu lock untuk semua sesi: Streamlit menjalankan tiap sesi/rerun di thread sendiri, dan pemakaian
    model yang sama secara bersamaan di GPU Apple (MPS) bisa membuat proses crash (assertion Metal)."""
    return threading.Lock()


def search(retriever: HybridRetriever, query: str, **kwargs) -> SearchResult:
    """Pencarian yang dijaga lock (lihat `model_lock`)."""
    with model_lock():
        return retriever.search(query, **kwargs)


def render_hit(hit: Hit, show_bm25: bool) -> None:
    """Satu kartu hasil: judul, skor tiap jalur, cuplikan, keyphrase dokumen, dan keyphrase yang cocok."""
    with st.container(border=True):
        st.markdown(f"**{hit.rank}. {esc(hit.title or hit.doc_id)}**  \nID: `{hit.doc_id.replace('`', '')}`")
        bm = f" | bm25 {hit.bm25_score:.4f}" if show_bm25 else ""   # BM25 hanya dihitung di mode bm25
        st.caption(f"skor akhir {hit.score:.4f} | semantik {hit.semantic_score:.4f} | keyphrase {hit.keyphrase_score:.4f}{bm}")
        st.text(hit.snippet)
        st.markdown("Keyphrase dokumen: " + chips(hit.doc_keyphrases))
        if hit.matched_keyphrases:
            st.markdown("Cocok dengan query: " + chips(hit.matched_keyphrases))


def main() -> None:
    """Susun halaman: sidebar pengaturan, tab Pencarian, dan tab Bandingkan mode."""
    st.set_page_config(page_title="Hybrid Semantic-Keyphrase Retrieval", layout="wide")
    st.title("Hybrid Semantic-Keyphrase Retrieval")
    st.caption("Pencarian dokumen akademik: embedding (semantik) + keyphrase dengan bobot alpha yang bisa diatur. "
               "BM25 tersedia sebagai pembanding.")

    config_path = os.environ.get("HYBRID_IR_CONFIG")
    if not config_path:
        st.error("Variabel HYBRID_IR_CONFIG belum diatur. Jalankan lewat: hybrid-ir ui --config <berkas.yaml>")
        return
    try:
        retriever = load_retriever(config_path)
    except (ConfigError, FileNotFoundError, IndexMismatchError) as exc:
        st.error(f"Gagal memuat indeks: {exc}")
        return

    cfg = retriever.cfg
    with st.sidebar:
        st.header("Pengaturan")
        modes = ["hybrid", "semantic", "keyphrase", "bm25"]
        mode = st.radio("Mode", modes, index=modes.index(cfg.retrieval.mode), format_func=MODE_TITLES.get)
        alpha = st.slider("alpha (bobot semantik; Hybrid)", 0.0, 1.0, float(cfg.retrieval.alpha), 0.05,
                          disabled=mode != "hybrid")
        top_k = st.slider("Jumlah hasil", 1, 30, min(cfg.retrieval.top_k, 30))
        st.divider()
        st.caption(f"Dokumen terindeks: {retriever.meta.get('n_docs')}")
        enc = retriever.meta.get("encoder", {})
        st.caption(f"Encoder: {enc.get('model_name') or enc.get('backend')}")
        st.caption(f"Keyphrase doc: {retriever.meta.get('doc_extractor', {}).get('method')}")

    tab_search, tab_compare = st.tabs(["Pencarian", "Bandingkan mode"])
    max_chars = cfg.retrieval.max_query_chars

    with tab_search:
        with st.form("search_form"):
            query = st.text_input("Query", max_chars=max_chars, placeholder="mis. keyphrases to improve search indexing")
            submitted = st.form_submit_button("Cari")
        if submitted:
            try:
                res = search(retriever, query, mode=mode, alpha=alpha, top_k=top_k)
            except (QueryError, MissingPrecomputed) as exc:
                st.error(str(exc))
            else:
                st.markdown("**Keyphrase query:** " + chips([k for k, _ in res.query_keyphrases]))
                if not res.hits:
                    st.info("Tidak ada hasil untuk mode ini.")
                for hit in res.hits:
                    render_hit(hit, mode == "bm25")

    with tab_compare:
        with st.form("compare_form"):
            cquery = st.text_input("Query untuk dibandingkan", max_chars=max_chars, key="cmp_query")
            csub = st.form_submit_button("Bandingkan")
        if csub:
            try:
                results = {m: search(retriever, cquery, mode=m, alpha=alpha, top_k=5)
                           for m in COMPARE_MODES}
            except (QueryError, MissingPrecomputed) as exc:
                st.error(str(exc))
            else:
                cols = st.columns(len(COMPARE_MODES))
                for col, m in zip(cols, COMPARE_MODES):
                    with col:
                        st.subheader(MODE_TITLES[m])
                        if not results[m].hits:
                            st.caption("tidak ada hasil")
                        for h in results[m].hits:
                            st.markdown(f"{h.rank}. {esc(h.title or h.doc_id)}  \n`{h.score:.3f}`")


main()
