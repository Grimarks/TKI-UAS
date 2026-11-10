#!/usr/bin/env python3
"""Alat bantu modul pembelajaran (folder modul/).

    ../.venv/bin/python scripts/modul.py --contoh     # hitung ulang angka contoh -> modul/data/contoh_angka.json
    ../.venv/bin/python scripts/modul.py --perbarui   # isi ulang blok kode asli di modul/*.md dari src/
    ../.venv/bin/python scripts/modul.py --cek        # periksa: blok kode cocok, angka bersumber, gambar ada
    ../.venv/bin/python scripts/modul.py --docx       # gabungkan modul menjadi Laporan/Modul_Pembelajaran_Kelompok1.docx

Blok kode di Markdown ditulis sebagai penanda, lalu diisi otomatis dengan kode ASLI beserta nomor barisnya:

    <!-- kode: src/hybrid_ir/fusion.py#minmax -->          (fungsi, kelas, atau Kelas.metode)
    <!-- kode: src/hybrid_ir/bm25.py#L44-66 -->            (rentang baris)
    ...isi dibangkitkan...
    <!-- /kode -->

Angka di modul (4 desimal atau persen) harus ada di analisis/ringkasan.json, results/, konfigurasi, atau
modul/data/contoh_angka.json (angka contoh yang dihitung dengan kode proyek oleh --contoh).
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import ast
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODUL = ROOT / "modul"
CONTOH = MODUL / "data" / "contoh_angka.json"
OUT_DOCX = ROOT.parent / "Laporan" / "Modul_Pembelajaran_Kelompok1.docx"
BLOCK = re.compile(r"<!-- kode: (?P<spec>[^ ]+) -->\n.*?<!-- /kode -->", re.S)
URUTAN = ["00_cara_memakai_modul.md", "01_masalah_dan_konsep_ir.md", "02_arsitektur_pipeline.md", "03_data_beir.md",
          "04_jalur_semantik.md", "05_jalur_keyphrase.md", "06_bm25_pembanding.md", "07_fusi_hybrid.md",
          "08_evaluasi_dan_statistik.md", "09_hasil_dan_pembahasan.md", "10_tur_kode.md",
          "11_menjalankan_dan_demo.md", "12_persiapan_tanya_jawab.md", "glosarium.md"]


# ====================================================================== blok kode
def _span(path: Path, name: str) -> tuple[int, int]:
    if name.startswith("L"):
        a, b = name[1:].split("-")
        return int(a), int(b)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parts = name.split(".")
    nodes = tree.body
    found = None
    for i, part in enumerate(parts):
        found = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part), None)
        if found is None:
            raise SystemExit(f"GALAT: {name} tidak ditemukan di {path}")
        nodes = getattr(found, "body", [])
    start = min([found.lineno] + [d.lineno for d in found.decorator_list])
    return start, found.end_lineno


def render_block(spec: str) -> str:
    rel, name = spec.split("#")
    path = ROOT / rel
    a, b = _span(path, name)
    lines = path.read_text(encoding="utf-8").splitlines()[a - 1:b]
    lang = "python" if rel.endswith(".py") else ("yaml" if rel.endswith((".yaml", ".yml")) else "text")
    return (f"<!-- kode: {spec} -->\n*Kode asli: `{rel}`, baris {a}–{b}.*\n\n```{lang}\n" + "\n".join(lines)
            + "\n```\n<!-- /kode -->")


def refresh(text: str) -> str:
    return BLOCK.sub(lambda m: render_block(m.group("spec")), text)


# ====================================================================== angka contoh
def hitung_contoh() -> dict:
    """Angka contoh di modul, dihitung dengan fungsi proyek (bukan diketik)."""
    import numpy as np

    from hybrid_ir.bm25 import BM25, bm25_idf
    from hybrid_ir.config import load_config
    from hybrid_ir.data import load_beir_dir
    from hybrid_ir.evaluation import average_precision, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
    from hybrid_ir.fusion import candidate_set, fuse_weighted
    from hybrid_ir.keyphrases import rank_weights
    from hybrid_ir.retriever import HybridRetriever
    from hybrid_ir.text import content_tokens
    logging.disable(logging.WARNING)
    os.chdir(ROOT)
    out: dict = {}

    # --- semantik: heart attack vs myocardial infarction (bab 01, 04)
    cfg = load_config("configs/scifact_bart.yaml")
    r = HybridRetriever.load(cfg)
    words = ["heart attack", "myocardial infarction", "kidney transplant"]
    v = r.encoder.encode_queries(words)
    out["cos_heart_attack__myocardial_infarction"] = float(v[0] @ v[1])
    out["cos_heart_attack__kidney_transplant"] = float(v[0] @ v[2])
    out["dim_embedding"] = int(v.shape[1])
    out["norma_vektor"] = [float(x) for x in np.linalg.norm(v, axis=1)]

    # --- fusi: kueri SciFact 1185 (bab 07)
    qtext = load_beir_dir(cfg.data.path).queries["1185"]
    sem, kp, qk = r.score_matrices([qtext])
    sem, kp = sem[0], kp[0]
    cand = candidate_set(sem, kp, cfg.retrieval.candidate_k)
    lo_s, hi_s, lo_k, hi_k = (float(sem[cand].min()), float(sem[cand].max()),
                              float(kp[cand].min()), float(kp[cand].max()))
    fus = {"n_kandidat": len(cand), "sem_min": lo_s, "sem_max": hi_s, "kp_min": lo_k, "kp_max": hi_k, "dokumen": {}}
    for did in ("16737210", "21203899"):
        i = r.doc_ids.index(did)
        ns, nk = (sem[i] - lo_s) / (hi_s - lo_s), (kp[i] - lo_k) / (hi_k - lo_k)
        fus["dokumen"][did] = {"sem": float(sem[i]), "kp": float(kp[i]), "sem_norm": float(ns), "kp_norm": float(nk),
                               "fusi_0.9": float(0.9 * ns + 0.1 * nk), "fusi_0.5": float(0.5 * ns + 0.5 * nk)}
    idx, vals = fuse_weighted(sem, kp, 0.9, cfg.retrieval.candidate_k)
    fus["hybrid_top1"] = [r.doc_ids[int(idx[0])], float(vals[0])]
    fus["keyphrase_query"] = [[p, w] for p, w in qk[0]]
    out["fusi_1185"] = fus

    # --- keyphrase: bobot urutan dan idf (bab 05)
    out["rank_weights"] = rank_weights(6)
    ki = r.kp_index
    idf = {}
    for u in ("p:kidney pair donat", "t:kidney", "t:cell"):
        c = ki.term_index[u]
        df = int(ki._binary[:, c].sum())
        idf[u] = {"df": df, "idf": float(ki.idf[c]), "idf_rumus": math.log((len(r.doc_ids) + 1) / (df + 1)) + 1}
    out["idf_scifact"] = idf
    out["n_dok_scifact"] = len(r.doc_ids)

    # --- BM25 kecil (bab 06)
    docs = ["heart attack risk in young adults", "myocardial infarction treatment after heart attack",
            "kidney transplant outcomes"]
    b = BM25("en").fit(docs)
    lens = [len(content_tokens(d)) for d in docs]
    avgdl = sum(lens) / len(lens)
    df_heart = sum("heart" in content_tokens(d) for d in docs)
    idf_heart = float(bm25_idf(np.array([df_heart]), len(docs))[0])
    norm = 1.2 * (1 - 0.75 + 0.75 * lens[0] / avgdl)
    w1 = idf_heart * 1 * 2.2 / (1 + norm)
    out["bm25_kecil"] = {"panjang": lens, "avgdl": avgdl, "df_heart": df_heart, "idf_heart": idf_heart,
                         "b_rasio_panjang": 0.75 * lens[0] / avgdl,
                         "penyebut_norm": norm, "bobot_per_term": w1, "skor_manual": 2 * w1,
                         "skor_kode": [float(x) for x in b.scores(["heart attack"])[0]],
                         "idf_kidney": float(bm25_idf(np.array([1]), 3)[0])}

    # --- nDCG bertingkat: NFCorpus PLAIN-2600 (bab 08)
    cfg_n = load_config("configs/nfcorpus_bart.yaml")
    rn = HybridRetriever.load(cfg_n)
    dn = load_beir_dir(cfg_n.data.path)
    qrel = dn.qrels_for("test")["PLAIN-2600"]
    ranked = [h.doc_id for h in rn.search(dn.queries["PLAIN-2600"], mode="hybrid", top_k=100).hits]
    gains = [qrel.get(d, 0) for d in ranked[:10]]
    ideal = sorted(qrel.values(), reverse=True)[:10]
    terms = [g / math.log2(i + 2) for i, g in enumerate(gains)]
    iterms = [g / math.log2(i + 2) for i, g in enumerate(ideal)]
    out["ndcg_plain_2600"] = {"query": dn.queries["PLAIN-2600"], "gain_top10": gains, "ideal_top10": ideal,
                              "suku_dcg": terms, "dcg": sum(terms), "suku_idcg": iterms, "idcg": sum(iterms),
                              "ndcg": ndcg_at_k(ranked, qrel, 10), "ap100": average_precision(ranked, qrel, 100),
                              "recall10": recall_at_k(ranked, qrel, 10), "precision10": precision_at_k(ranked, qrel, 10),
                              "mrr10": mrr_at_k(ranked, qrel, 10), "n_relevan": sum(1 for x in qrel.values() if x > 0)}
    return out


# ====================================================================== pemeriksaan
def source_numbers() -> list[float]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import cek_laporan
    nums = cek_laporan.source_numbers()
    if CONTOH.is_file():
        cek_laporan.walk_numbers(json.loads(CONTOH.read_text()), nums)
    return nums


def cek() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import cek_laporan
    fails = []
    nums = source_numbers()
    for name in URUTAN:
        path = MODUL / name
        if not path.is_file():
            fails.append(f"{name}: berkas tidak ada")
            continue
        text = path.read_text(encoding="utf-8")
        if refresh(text) != text:
            fails.append(f"{name}: blok kode tidak cocok dengan kode sekarang (jalankan --perbarui)")
        for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            if not (path.parent / img).is_file():
                fails.append(f"{name}: gambar tidak ada: {img}")
        prose = re.sub(r"```.*?```", " ", text, flags=re.S)                  # kode tidak diperiksa
        prose = re.sub(r"\b\d+\.\d+\.\d+\b", " ", prose)                   # versi
        for tok in sorted(set(re.findall(r"[+−-]?\d+\.\d{4}\b|[+−-]?\d+(?:\.\d+)?%", prose))):
            if tok.lstrip("+−-") in cek_laporan.CONSTANTS or tok in ("100%", "50%", "80%"):
                continue
            if not cek_laporan.matches(tok, nums):
                fails.append(f"{name}: angka tanpa sumber: {tok}")
        n_soal = len(re.findall(r"^\*\*Soal \d+", text, flags=re.M))
        n_kunci = len(re.findall(r"^\*\*Jawaban \d+", text, flags=re.M))
        if name.startswith(tuple(f"{i:02d}" for i in range(1, 12))) and not (3 <= n_soal <= 5 and n_soal == n_kunci):
            fails.append(f"{name}: soal latihan {n_soal}, kunci {n_kunci} (harus 3–5 dan sama)")
    print(f"Diperiksa {len(URUTAN)} berkas modul.")
    if fails:
        print("GAGAL:")
        for f in fails:
            print("  -", f)
        return 1
    print("SEMUA PEMERIKSAAN MODUL LULUS")
    return 0


def docx() -> int:
    if not shutil.which("pandoc"):
        raise SystemExit("pandoc belum terpasang")
    # satu berkas gabungan dengan pemisah halaman antarbab (blok OpenXML mentah; komentar HTML dibuang pandoc)
    page_break = '\n\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n'
    gabungan = MODUL / ".gabungan_modul.md"
    teks = page_break.join((MODUL / n).read_text(encoding="utf-8") for n in URUTAN)
    # ukuran gambar khusus docx (atribut {width=..} tidak dipakai di Markdown sumber agar tampilan GitHub tetap bersih)
    teks = re.sub(r"(!\[[^\]]*\]\(gambar/gambar1_pipeline\.png\))", r"\1{width=70%}", teks)
    teks = re.sub(r"(!\[[^\]]*\]\(\.\./analisis/gambar/[^)]+\))", r"\1{width=95%}", teks)
    gabungan.write_text(teks, encoding="utf-8")
    cmd = ["pandoc", str(gabungan), "-o", str(OUT_DOCX), "--toc", "--toc-depth=1", "--resource-path", str(MODUL),
           "--metadata", "title=Modul Pembelajaran: Hybrid Semantic-Keyphrase Retrieval",
           "--metadata", "subtitle=Kelompok 1 — Temu Kembali Informasi", "--metadata", "lang=id",
           "--metadata", "toc-title=Daftar Isi",
           "-f", "markdown+tex_math_dollars"]
    try:
        subprocess.run(cmd, check=True)
    finally:
        gabungan.unlink(missing_ok=True)
    print(f"Ditulis: {OUT_DOCX}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contoh", action="store_true")
    ap.add_argument("--perbarui", action="store_true")
    ap.add_argument("--cek", action="store_true")
    ap.add_argument("--docx", action="store_true")
    a = ap.parse_args(argv)
    if a.contoh:
        CONTOH.parent.mkdir(parents=True, exist_ok=True)
        CONTOH.write_text(json.dumps(hitung_contoh(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Ditulis: {CONTOH.relative_to(ROOT)}")
    if a.perbarui:
        for name in URUTAN:
            p = MODUL / name
            if p.is_file():
                p.write_text(refresh(p.read_text(encoding="utf-8")), encoding="utf-8")
        print("Blok kode diperbarui.")
    rc = cek() if a.cek else 0
    if a.docx and rc == 0:
        rc = docx()
    return rc


if __name__ == "__main__":
    sys.exit(main())
