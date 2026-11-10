#!/usr/bin/env python3
"""Bangun Laporan/Laporan_UAS_Kelompok1.docx dari laporan dasar (2) + hasil di results/ dan analisis/.

Selalu mulai dari salinan BERSIH "Hybrid Semantic-Keyphrase Retrieval (2).docx" (berkas itu tidak diubah), lalu:
  A. menyelaraskan teks dengan desain final (BM25 hanya pembanding; persamaan (8) lama dihapus);
  B. mengisi Bab 4 (tabel, gambar, narasi) dan Bab 5;
  C. mengisi abstrak, lingkungan implementasi, kontribusi tim, singkatan; memperbaiki salah ketik.

SEMUA angka diambil dari analisis/ringkasan.json, results/*/metrics.json, dan configs/*.yaml (bukan diketik).
Fakta studi kasus dijaga dengan assert: bila pemilihan kasus berubah, skrip berhenti agar narasi ditinjau ulang.

Format: paragraf/tabel/gambar baru dibuat dengan MENIRU prototipe di template (pPr dan rPr disalin), sehingga
font, ukuran, spasi, gaya tabel, dan keterangan tetap sama dengan template.

    ../.venv/bin/python scripts/bangun_laporan_uas.py
"""
from __future__ import annotations

import copy
import json
import re
import struct
import sys
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]                      # hybrid-ir/
LAPORAN = ROOT.parent / "Laporan"
BASE = LAPORAN / "Hybrid Semantic-Keyphrase Retrieval (2).docx"
OUT = LAPORAN / "Laporan_UAS_Kelompok1.docx"
GAMBAR1 = ROOT.parents[1] / "Sumber" / "Referensi_UTS" / "Pipeline_V2_laporan.png"
FIG = ROOT / "analisis" / "gambar"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
IMG_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


def w(tag: str) -> str:
    return f"{{{NS['w']}}}{tag}"


# Urutan anak <w:rPr> menurut skema OOXML (agar elemen baru disisipkan di posisi yang sah).
RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
             "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing", "w",
             "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign",
             "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"]


# ====================================================================== data
def load_data() -> dict:
    import yaml
    r = json.loads((ROOT / "analisis" / "ringkasan.json").read_text(encoding="utf-8"))
    m = {ds: json.loads((ROOT / "results" / f"{ds}_bart" / "metrics.json").read_text(encoding="utf-8"))
         for ds in ("scifact", "nfcorpus")}
    cases = json.loads((ROOT / "analisis" / "studi_kasus.json").read_text(encoding="utf-8"))
    cfg = {ds: yaml.safe_load((ROOT / "configs" / f"{ds}_bart.yaml").read_text(encoding="utf-8"))
           for ds in ("scifact", "nfcorpus")}
    return {"r": r, "m": m, "cases": cases, "cfg": cfg}


def f4(x: float) -> str:
    return f"{x:.4f}"


def sd(x: float) -> str:           # selisih bertanda
    return f"{x:+.4f}".replace("-", "−")


def pv(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ====================================================================== dokumen
class Docx:
    def __init__(self, path: Path) -> None:
        self.zin = zipfile.ZipFile(path)
        self.doc = etree.fromstring(self.zin.read("word/document.xml"))
        self.rels = etree.fromstring(self.zin.read("word/_rels/document.xml.rels"))
        self.body = self.doc.find(w("body"))
        self.new_files: dict[str, bytes] = {}
        self._docpr = 100

    # ------------------------------------------------------------------ pencarian
    def paragraphs(self):
        return list(self.body.iter(w("p")))

    def find_p(self, snippet: str):
        hits = [p for p in self.paragraphs() if snippet in ptext(p)]
        if len(hits) != 1:
            raise SystemExit(f"GALAT: '{snippet[:60]}' ditemukan {len(hits)}×, harus tepat 1×")
        return hits[0]

    def top(self, el):
        """Elemen anak langsung <w:body> yang memuat `el` (paragraf atau tabel)."""
        while el.getparent() is not self.body:
            el = el.getparent()
        return el

    # ------------------------------------------------------------------ gambar
    def add_image(self, png: bytes, name: str) -> str:
        rid = f"rIdUAS{len(self.new_files) + 1}"
        self.new_files[f"word/media/{name}"] = png
        rel = etree.SubElement(self.rels, f"{{{REL_NS}}}Relationship")
        rel.set("Id", rid)
        rel.set("Type", IMG_REL)
        rel.set("Target", f"media/{name}")
        return rid

    def next_docpr(self) -> int:
        self._docpr += 1
        return self._docpr

    def save(self, out: Path) -> None:
        tmp = out.with_suffix(".tmp.docx")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in self.zin.infolist():
                if item.filename == "word/document.xml":
                    data = etree.tostring(self.doc, xml_declaration=True, encoding="UTF-8", standalone=True)
                elif item.filename == "word/_rels/document.xml.rels":
                    data = etree.tostring(self.rels, xml_declaration=True, encoding="UTF-8", standalone=True)
                elif item.filename in self.new_files:
                    data = self.new_files.pop(item.filename)
                else:
                    data = self.zin.read(item.filename)
                zout.writestr(item, data)
            for name, data in self.new_files.items():
                zout.writestr(name, data)
        tmp.replace(out)


# ====================================================================== teks & run
def ptext(p) -> str:
    return "".join(t.text or "" for r in p.findall(w("r")) for t in r.findall(w("t")))


def run_text(r) -> str:
    return "".join(t.text or "" for t in r.findall(w("t")))


def set_run_text(r, text: str) -> None:
    ts = r.findall(w("t"))
    for t in ts[1:]:
        r.remove(t)
    if not ts:
        t = etree.SubElement(r, w("t"))
    else:
        t = ts[0]
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def rpr_set(rpr, tag: str, val: str | None) -> None:
    """Set/hapus <w:tag w:val=...> di rPr dengan urutan skema yang benar."""
    el = rpr.find(w(tag))
    if val is None:
        if el is not None:
            rpr.remove(el)
        return
    if el is None:
        el = etree.Element(w(tag))
        pos = RPR_ORDER.index(tag)
        idx = len(rpr)
        for i, child in enumerate(rpr):
            name = etree.QName(child).localname
            if name in RPR_ORDER and RPR_ORDER.index(name) > pos:
                idx = i
                break
        rpr.insert(idx, el)
    el.set(w("val"), val)


def clean_rpr(rpr):
    rpr = copy.deepcopy(rpr) if rpr is not None else etree.Element(w("rPr"))
    rpr_set(rpr, "highlight", None)
    return rpr


TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")


def make_runs(markup: str, base_rpr) -> list:
    """Markup kecil: **tebal**, *miring*, _{subskrip}, ^{superskrip}."""
    runs = []
    for part in TOKEN.split(markup):
        if not part:
            continue
        bold = italic = False
        vert = None
        if part.startswith("**") and part.endswith("**"):
            part, bold = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 1:
            part, italic = part[1:-1], True
        elif part.startswith("_{"):
            part, vert = part[2:-1], "subscript"
        elif part.startswith("^{"):
            part, vert = part[2:-1], "superscript"
        r = etree.Element(w("r"))
        rpr = clean_rpr(base_rpr)
        if bold:
            rpr_set(rpr, "b", "1"); rpr_set(rpr, "bCs", "1")
        if italic:
            rpr_set(rpr, "i", "1"); rpr_set(rpr, "iCs", "1")
        if vert:
            rpr_set(rpr, "vertAlign", vert)
        r.append(rpr)
        t = etree.SubElement(r, w("t"))
        t.text = part
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        runs.append(r)
    return runs


def _split_run(r, offset: int):
    """Belah run di posisi `offset` teksnya; kembalikan run kanan (run kiri = r)."""
    text = run_text(r)
    right = copy.deepcopy(r)
    set_run_text(r, text[:offset])
    set_run_text(right, text[offset:])
    r.addnext(right)
    return right


def replace(p, old: str, markup: str) -> None:
    """Ganti `old` (boleh melintasi beberapa run) dengan `markup`; format dasar diambil dari run pertama."""
    full = ptext(p)
    if full.count(old) != 1:
        raise SystemExit(f"GALAT: '{old[:60]}' muncul {full.count(old)}× di paragraf '{full[:60]}'")
    start, end = full.index(old), full.index(old) + len(old)
    pos = 0
    for r in list(p.findall(w("r"))):                  # belah run di batas awal dan akhir
        t = run_text(r)
        a, b = pos, pos + len(t)
        if a < start < b:
            r = _split_run(r, start - a)
            a = start
            t = run_text(r)
            b = a + len(t)
        if a < end < b:
            _split_run(r, end - a)
        pos = b if not (a < end < b) else end
        if pos >= end and a >= end:
            break
    pos, span = 0, []
    for r in p.findall(w("r")):
        t = run_text(r)
        if start <= pos and pos + len(t) <= end and t:
            span.append(r)
        pos += len(t)
    base = span[0].find(w("rPr"))
    for nr in make_runs(markup, base):
        span[0].addprevious(nr)
    for r in span:
        p.remove(r)


def set_paragraph(p, markup: str, base_rpr=None) -> None:
    runs = p.findall(w("r"))
    if base_rpr is None:
        texted = [r for r in runs if run_text(r)]
        base_rpr = texted[0].find(w("rPr")) if texted else None
    for r in runs:
        p.remove(r)
    for nr in make_runs(markup, base_rpr):
        p.append(nr)


def unhighlight(el) -> None:
    for h in el.iter(w("highlight")):
        h.getparent().remove(h)


def drop_ids(el):
    for e in el.iter():
        for att in list(e.attrib):
            if att.startswith(f"{{{NS['w14']}}}"):
                del e.attrib[att]
    return el


def clone_para(proto, markup: str, base_rpr):
    p = drop_ids(copy.deepcopy(proto))
    for r in p.findall(w("r")):
        p.remove(r)
    for nr in make_runs(markup, base_rpr):
        p.append(nr)
    return p


# ====================================================================== pembangun
class Builder:
    def __init__(self, doc: Docx) -> None:
        self.d = doc
        P = doc.find_p
        # --- prototipe dari template
        self.body_p = P("Pendekatan IR klasik mencocokkan kata")
        self.body_rpr = [r for r in self.body_p.findall(w("r")) if run_text(r)][0].find(w("rPr"))
        self.cap_tab = P("Tabel 1. Perbandingan dua penelitian")
        self.cap_fig = P("Gambar 1. Arsitektur sistem")
        # run kedua keterangan ("Perbandingan dua ...") = format teks keterangan biasa (run pertama tebal)
        self.cap_rpr = [r for r in self.cap_tab.findall(w("r")) if run_text(r)][1].find(w("rPr"))
        self.fig_p = doc.top(next(doc.body.iter(w("drawing"))))
        tab2_cap = doc.top(P("Tabel 2. Statistik koleksi uji"))
        self.tab_proto = tab2_cap.getnext()
        self.spacer = self.tab_proto.getnext()
        assert etree.QName(self.tab_proto).localname == "tbl" and not ptext(self.spacer)

    # ------------------------------------------------------------------ elemen baru
    def para(self, markup: str):
        return clone_para(self.body_p, markup, self.body_rpr)

    def caption(self, kind: str, num: int, text: str):
        proto = self.cap_tab if kind == "Tabel" else self.cap_fig
        p = clone_para(proto, "", None)
        p.extend(make_runs(f"**{kind} {num}.**", self.cap_rpr))
        p.extend(make_runs(" " + text, self.cap_rpr))
        return p

    def figure(self, png_path: Path, name: str, width_emu: int = 5257800):
        data = png_path.read_bytes()
        wpx, hpx = struct.unpack(">II", data[16:24])
        cx, cy = width_emu, int(width_emu * hpx / wpx)
        rid = self.d.add_image(data, name)
        p = drop_ids(copy.deepcopy(self.fig_p))
        for ext in p.iter(f"{{{NS['wp']}}}extent", f"{{{NS['a']}}}ext"):
            ext.set("cx", str(cx)); ext.set("cy", str(cy))
        dp = p.find(f".//{{{NS['wp']}}}docPr")
        dp.set("id", str(self.d.next_docpr())); dp.set("name", name)
        p.find(f".//{{{NS['pic']}}}cNvPr").set("name", name)
        p.find(f".//{{{NS['a']}}}blip").set(f"{{{NS['r']}}}embed", rid)
        src = p.find(f".//{{{NS['a']}}}srcRect")
        if src is not None:
            src.getparent().remove(src)
        return p

    def table(self, header: list[str], rows: list[list[str]], widths: list[int], align: list[str]):
        """Tabel bergaya sama dengan Tabel 2 (garis atas/bawah header, garis bawah akhir, latar f2f2f2)."""
        t = drop_ids(copy.deepcopy(self.tab_proto))
        trs = t.findall(w("tr"))
        head_proto, body_proto, last_proto = trs[0], trs[1], trs[-1]
        for tr in trs:
            t.remove(tr)
        scale = 8788 / sum(widths)
        cols = [int(round(x * scale)) for x in widths]
        cols[-1] += 8788 - sum(cols)
        for grid in list(t.iter(w("tblGrid"))):
            for gc in grid.findall(w("gridCol")):
                grid.remove(gc)
            for c in cols:
                g = etree.Element(w("gridCol")); g.set(w("w"), str(c))
                change = grid.find(w("tblGridChange"))
                if change is not None:
                    change.addprevious(g)
                else:
                    grid.append(g)
        all_rows = [(header, head_proto, True)] + [
            (r, last_proto if i == len(rows) - 1 else body_proto, False) for i, r in enumerate(rows)]
        for cells, proto, is_head in all_rows:
            tr = copy.deepcopy(proto)
            tcs = tr.findall(w("tc"))
            tc_proto = tcs[0]
            for tc in tcs:
                tr.remove(tc)
            for j, text in enumerate(cells):
                tc = copy.deepcopy(tc_proto)
                p = tc.find(w("p"))
                for extra in tc.findall(w("p"))[1:]:
                    tc.remove(extra)
                runs = p.findall(w("r"))
                base = next((r.find(w("rPr")) for r in runs if run_text(r)), None)
                if base is None:
                    base = etree.Element(w("rPr"))
                base = copy.deepcopy(base)
                rpr_set(base, "sz", "16"); rpr_set(base, "szCs", "16")
                for r in runs:
                    p.remove(r)
                ppr = p.find(w("pPr"))
                ppr.find(w("jc")).set(w("val"), "center" if is_head else align[j])
                mark = ppr.find(w("rPr"))                   # ukuran paragraph mark = 8 pt agar sel kosong tidak lebih tinggi
                if mark is None:
                    mark = etree.SubElement(ppr, w("rPr"))
                rpr_set(mark, "sz", "16"); rpr_set(mark, "szCs", "16")
                p.extend(make_runs(text if not is_head else f"**{text}**", base))
                tr.append(tc)
            t.append(tr)
        return t

    def spacer_p(self):
        return drop_ids(copy.deepcopy(self.spacer))

    def insert_after(self, anchor, elements: list) -> None:
        for el in elements:
            anchor.addnext(el)
            anchor = el

    def replace_placeholder(self, snippet: str, elements: list) -> None:
        ph = self.d.top(self.d.find_p(snippet))
        self.insert_after(ph, elements)
        self.d.body.remove(ph)


# ====================================================================== A. selaraskan desain final
def bagian_a(b: Builder, D: dict) -> None:
    d = b.d
    P = d.find_p
    # salah ketik e-mail Leonardo
    replace(P("@student.unsri.ac.idl"), "@student.unsri.ac.idl", "@student.unsri.ac.id")
    # abstrak: BM25 hanya pembanding
    p = P("Kontribusi penelitian ini adalah rancangan pipeline")
    replace(p, "ditambah jalur BM25 sebagai jalur opsional dan pembanding", "dengan BM25 sebagai pembanding")
    replace(p, "bobotnya ditala pada data latih", "bobotnya ditala pada data penalaan")
    # Pendahuluan
    replace(P("BM25 disertakan sebagai jalur ketiga"), "BM25 disertakan sebagai jalur ketiga yang opsional sekaligus pembanding.",
            "BM25 disertakan hanya sebagai pembanding leksikal dan tidak ikut digabungkan.")
    p = P("Pertanyaan penelitian yang dijawab")
    replace(p, "dibandingkan setiap jalur secara terpisah dan dibandingkan BM25.", "dibandingkan setiap jalur secara terpisah.")
    replace(p, "Ketiga, apakah penambahan BM25 sebagai sinyal ketiga memberi peningkatan lebih lanjut.",
            "Ketiga, seberapa dekat kualitas peringkat sistem hibrida dengan BM25 sebagai pembanding leksikal yang kuat.")
    replace(P("Kontribusi penelitian ini adalah sebagai berikut"),
            "Ketiga, evaluasi terhadap BM25, pencarian semantik saja, dan pencarian keyphrase saja pada dua koleksi uji berformat BEIR dengan uji signifikansi.",
            "Ketiga, evaluasi empat skenario, yaitu BM25 sebagai pembanding, pencarian semantik saja, pencarian keyphrase saja, "
            "dan hibrida, pada dua koleksi uji berformat BEIR dengan uji signifikansi dan analisis per kueri.")
    replace(P("Makalah ini disusun sebagai berikut"), " [Bagian 4 dan 5 diisi pada laporan UAS.]", "")
    # Posisi penelitian + Tabel 1
    replace(P("sinyal pelengkap utama pada penelitian ini adalah keyphrase"), "sedangkan BM25 menjadi pembanding dan jalur opsional.",
            "sedangkan BM25 hanya menjadi pembanding.")
    replace(P("dinormalisasi lalu digabung dengan kombinasi linear; BM25 opsional"), "BM25 opsional", "BM25 sebagai pembanding")
    # 3.1 Arsitektur
    p = P("Sistem memiliki tiga jalur skor")
    replace(p, "Sistem memiliki tiga jalur skor, yaitu jalur A (semantik), jalur B (keyphrase), dan jalur C (BM25). "
               "Jalur A dan B adalah inti usulan, sedangkan jalur C bersifat opsional untuk uji coba dan juga dipakai sebagai pembanding.",
            "Sistem memiliki dua jalur skor utama, yaitu jalur A (semantik) dan jalur B (keyphrase), yang digabung menjadi skor "
            "hibrida. BM25 dihitung terpisah sebagai pembanding: skornya langsung dinilai pada fase evaluasi dan tidak ikut digabung.")
    replace(p, "Indeks BM25 dibangun saat dibutuhkan.", "Indeks BM25 dibangun di memori saat pertama kali dibutuhkan.")
    replace(p, "Setiap jalur menghasilkan skor untuk dokumen, skor dinormalisasi lalu digabung,",
            "Jalur A dan B masing-masing menghasilkan skor untuk dokumen, kedua skor dinormalisasi lalu digabung,")
    replace(p, "daftar dokumen dibandingkan dengan penilaian relevansi",
            "daftar dokumen dari keempat skenario dibandingkan dengan penilaian relevansi")
    # 3.2 prapemrosesan: huruf kecil hanya untuk pencocokan keyphrase dan BM25 (sesuai kode)
    replace(P("Prapemrosesan dasar meliputi normalisasi Unicode"),
            "perataan spasi, dan pengubahan ke huruf kecil.",
            "dan perataan spasi. Untuk pencocokan keyphrase dan BM25, teks juga diubah ke huruf kecil, ditokenisasi, dan "
            "di-*stem*, sedangkan teks untuk encoder hanya dinormalisasi karena tokenizer model menangani sendiri huruf besar dan kecil.")
    # Gambar 1: keterangan
    set_paragraph(b.cap_fig, "", None)
    b.cap_fig.extend(make_runs("**Gambar 1.**", b.cap_rpr))
    b.cap_fig.extend(make_runs(
        " Arsitektur sistem Hybrid Semantic-Keyphrase Retrieval. Garis dan kotak penuh adalah alur utama (jalur A dan B yang "
        "digabung); garis dan kotak putus-putus adalah BM25 sebagai pembanding, yang skornya langsung dinilai pada fase "
        "evaluasi dan tidak ikut difusikan.", b.cap_rpr))
    # 3.3 SBERT disebut di teks (dipakai di Gambar 1 dan daftar singkatan)
    replace(P("Setiap teks dokumen dienkode oleh model"), "model sentence embedding all-MiniLM-L6-v2",
            "model *sentence embedding* keluarga Sentence-BERT (SBERT), yaitu all-MiniLM-L6-v2,")
    # 3.4 konfigurasi
    replace(P("Pada konfigurasi awal dipakai paling banyak 10 frasa"), "Pada konfigurasi awal dipakai paling banyak",
            "Sistem memakai paling banyak")
    # 3.5 BM25 -> pembanding
    h = P("Jalur C (Opsional): BM25")
    replace(h, "Jalur C (Opsional): BM25", "Pembanding: BM25")
    p = P("Jalur C memakai tokenisasi")
    replace(p, "Jalur C memakai tokenisasi, pembuangan stopword, dan stemming yang sama dengan jalur B.",
            "BM25 memakai tokenisasi, pembuangan *stopword*, dan *stemming* yang sama dengan jalur B.")
    replace(p, "dihitung sekali saat indeks dibangun dan disimpan sebagai matriks jarang",
            "dihitung sekali saat BM25 pertama kali dibutuhkan dan disimpan di memori sebagai matriks jarang")
    p.extend(make_runs(" Skor BM25 tidak digabung dengan skor jalur A dan B; dokumen hanya diperingkat menurut skor BM25 "
                       "sebagai pembanding leksikal.", b.body_rpr))
    # 3.6 Pemrosesan kueri
    replace(P("Cabang BM25 memecah kueri menjadi term."), "Cabang BM25 memecah kueri menjadi term.",
            "Untuk pembanding BM25, kueri dipecah menjadi term dengan aturan yang sama dengan dokumen.")
    # 3.7 Normalisasi & fusi
    p = P("Untuk satu kueri, sistem menghitung skor setiap jalur")
    replace(p, "sistem menghitung skor setiap jalur terhadap seluruh dokumen", "sistem menghitung skor jalur A dan B terhadap seluruh dokumen")
    replace(p, "gabungan C dokumen teratas dari setiap jalur, dengan C = 100 pada konfigurasi awal",
            "gabungan C dokumen teratas dari kedua jalur, dengan C = 100")
    replace(P("Nilai awal α adalah 0.6"),
            "Nilai awal α adalah 0.6, dan nilai akhirnya dipilih dari 0, 0.1, ..., 1 berdasarkan nDCG@10 pada kueri latih.",
            "Nilai α ditala pada fase evaluasi: setiap nilai 0, 0.1, ..., 1 dicoba pada kueri penalaan (split train untuk SciFact "
            "dan split dev untuk NFCorpus), lalu dipilih nilai dengan nDCG@10 tertinggi; bila ada nilai yang sama, dipilih yang "
            "paling dekat ke 0.5. Perintah pencarian dan antarmuka memakai α hasil penalaan tersebut, yang dicatat pada "
            "berkas konfigurasi.")
    # hapus paragraf mode tiga jalur + persamaan (8)
    p8 = d.top(P("Jika jalur C diaktifkan"))
    t8 = p8.getnext()
    assert "shyb3" in "".join(x.text or "" for x in t8.iter()) or "w3" in "".join(x.text or "" for x in t8.iter())
    d.body.remove(p8)
    d.body.remove(t8)
    # nomor persamaan (9),(10) -> (8),(9)
    for old, new in (("(9)", "(8)"), ("(10)", "(9)")):
        cell = [x for x in d.paragraphs() if ptext(x).strip() == old]
        assert len(cell) == 1, old
        replace(cell[0], old, new)
    replace(P("Nilai nDCG@k dihitung dengan (9) dan (10)"), "dengan (9) dan (10)", "dengan (8) dan (9)")
    replace(P("Empat skenario dibandingkan, yaitu BM25,"), "yaitu BM25,", "yaitu BM25 sebagai pembanding,")
    # Algoritma 1
    p = P("Keluaran: matriks semantik V, matriks keyphrase M, dan matriks BM25 B (opsional)")
    replace(p, ", dan matriks BM25 B (opsional)", "")
    replace(p, ", matriks keyphrase", " dan matriks keyphrase")
    l10 = P("10: jika jalur C aktif maka")
    l10.getparent().remove(l10)
    replace(P("11: simpan V dan M"), "11:", "10:")
    # Algoritma 2
    p = P("Masukan: kueri q, matriks V dan M (dan B)")
    replace(p, " (dan B)", "")
    replace(p, " (atau w)", "")
    l7 = P("7:  (opsional) tambahkan skor BM25")
    l7.getparent().remove(l7)
    replace(P("8:  kembalikan T dokumen"), "8:", "7:")
    # Gambar 1 baru (aspek sama -> ukuran tampilan sama)
    if GAMBAR1.is_file():
        d.new_files["word/media/image1.png"] = GAMBAR1.read_bytes()
    else:
        print(f"PERINGATAN: {GAMBAR1} tidak ada; Gambar 1 lama dipertahankan. Ekspor dulu dari draw.io.")


# ====================================================================== B. Bab 4 + C. bagian lain
def bagian_b(b: Builder, D: dict) -> None:
    r, M, cfg = D["r"], D["m"], D["cfg"]
    tu = r["tabel_utama"]
    sig = {(s["dataset"], s["b"]): s for s in r["signifikansi"]}
    msk = r["menang_seri_kalah"]
    kp = r["keyphrase"]
    al = r["alpha"]
    sweep = {ds: {x["alpha"]: x for x in M[ds]["alpha_sweep_tune"]} for ds in M}
    waktu = {(x["dataset"], x["langkah"].split(" (")[0]): x for x in r["waktu"]}
    H = lambda ds, s, met="ndcg@10": tu[ds][s][met]

    # ---------- klaim yang dicek sebelum ditulis
    for ds in ("scifact", "nfcorpus"):
        assert al[ds]["alpha_selected"] == 0.9
        assert sig[(ds, "semantic")]["p_value"] < 0.05 and sig[(ds, "keyphrase")]["p_value"] < 0.05
        assert sig[(ds, "bm25")]["p_value"] >= 0.05 and sig[(ds, "bm25")]["mean_diff"] < 0
        assert H(ds, "hybrid_weighted", "recall@100") == max(H(ds, s, "recall@100") for s in tu[ds])
        assert al[ds]["alpha_terbaik_posthoc_test"] == 0.9
        best = max(sweep[ds].values(), key=lambda x: x["ndcg@10"])
        assert best["alpha"] == 0.9 and sweep[ds][1.0]["ndcg@10"] < best["ndcg@10"]

    sf, nf = "scifact", "nfcorpus"
    kc_sf, kc_nf = cfg[sf]["keyphrase"], cfg[sf]["retrieval"]
    assert cfg[sf]["keyphrase"] == {**cfg[nf]["keyphrase"], "precomputed_files": cfg[sf]["keyphrase"]["precomputed_files"]}

    # ---------- pengantar Bab 4
    intro = [b.para(
        "Bagian ini menyajikan hasil keempat skenario pada split test kedua koleksi, pengaruh bobot fusi, analisis "
        "keyphrase, dan studi kasus. Seluruh angka diambil dari keluaran perintah hybrid-ir evaluate dan skrip analisis "
        "pada repositori kode, tanpa pelatihan ulang model.")]
    b.replace_placeholder("[Bagian ini diisi pada laporan UAS.", intro)

    # ---------- 4.1 Pengaturan Eksperimen
    w_ = lambda ds, k: waktu[(ds, k)]["rata_ms"]
    s41 = [
        b.para(f"Bobot fusi α dipilih dari 0, 0.1, ..., 1 pada kueri penalaan, yaitu {al[sf]['n_tune']} kueri train SciFact "
               f"dan {al[nf]['n_tune']} kueri dev NFCorpus. Pada kedua koleksi nilai terpilih adalah α = "
               f"{al[sf]['alpha_selected']}, dengan nDCG@10 kueri penalaan {f4(al[sf]['ndcg_tune_at_selected'])} untuk "
               f"SciFact dan {f4(al[nf]['ndcg_tune_at_selected'])} untuk NFCorpus. Nilai ini dipakai tanpa perubahan pada "
               f"{al[sf]['n_eval']} kueri test SciFact dan {al[nf]['n_eval']} kueri test NFCorpus. Konfigurasi lain sama untuk "
               f"kedua koleksi: encoder all-MiniLM-L6-v2 dengan panjang masukan maksimum {cfg[sf]['encoder']['max_seq_length']} "
               f"token; BART-KPG dengan *beam search* berukuran {kc_sf['num_beams']}, paling banyak {kc_sf['top_n']} keyphrase "
               f"per dokumen dan {kc_sf['query_top_n']} per kueri; unit pencocokan berupa frasa utuh dan kata; kemiripan "
               f"kosinus; ukuran kandidat C = {kc_nf['candidate_k']}; serta BM25 dengan *k*_{{1}} = 1.2 dan *b* = 0.75."),
        b.para(f"Perangkat keras dan pustaka yang dipakai dijelaskan pada Subbab Lingkungan Implementasi. Rata-rata waktu "
               f"pencarian per kueri pada komputer lokal adalah {w_(sf, 'Hybrid'):.1f} ms (SciFact) dan "
               f"{w_(nf, 'Hybrid'):.1f} ms (NFCorpus) untuk hibrida, {w_(sf, 'Semantic'):.1f} ms dan "
               f"{w_(nf, 'Semantic'):.1f} ms untuk semantik saja, serta kurang dari 1 ms untuk BM25 maupun keyphrase saja, "
               f"karena keyphrase kueri uji sudah dibangkitkan sebelumnya. Untuk kueri bebas di antarmuka, pembangkitan "
               f"keyphrase kueri dengan BART-KPG menambah rata-rata {w_(nf, 'BART-KPG query bebas'):.1f} ms hingga "
               f"{w_(sf, 'BART-KPG query bebas'):.1f} ms per kueri. Indeks BM25 dibangun di memori satu kali dalam "
               f"{w_(nf, 'Bangun indeks BM25 di memori'):.0f} ms hingga {w_(sf, 'Bangun indeks BM25 di memori'):.0f} ms."),
    ]
    b.replace_placeholder("[Isi: nilai α terpilih, konfigurasi model, dan lingkungan.]", s41)

    # ---------- 4.2 Perbandingan Metode
    order = ["bm25", "semantic", "keyphrase", "hybrid_weighted"]
    nama = {"bm25": "BM25 (pembanding)", "semantic": "Semantik saja", "keyphrase": "Keyphrase saja",
            "hybrid_weighted": "Hibrida (α = 0.9)"}
    mets = ["ndcg@10", "map@100", "recall@10", "recall@100", "precision@10", "mrr@10"]
    rows = []
    for ds, dn in ((sf, "SciFact"), (nf, "NFCorpus")):
        for i, s in enumerate(order):
            cells = [dn if i == 0 else "", nama[s]]
            for met in mets:
                v = H(ds, s, met)
                best = max(H(ds, x, met) for x in order)
                cells.append(f"**{f4(v)}**" if v == best else f4(v))
            rows.append(cells)
    tab3 = b.table(["Koleksi", "Skenario", "nDCG@10", "MAP@100", "R@10", "R@100", "P@10", "MRR@10"], rows,
                   [12, 18, 10, 10, 10, 10, 10, 10], ["left", "left"] + ["center"] * 6)
    sig_rows = []
    for ds, dn in ((sf, "SciFact"), (nf, "NFCorpus")):
        for i, other in enumerate(("semantic", "keyphrase", "bm25")):
            s = sig[(ds, other)]
            m_ = msk[ds][other]
            lawan = {"semantic": "semantik saja", "keyphrase": "keyphrase saja", "bm25": "BM25"}[other]
            sig_rows.append([dn if i == 0 else "", f"Hibrida − {lawan}", sd(s["mean_diff"]),
                             f"[{sd(s['ci95'][0])}, {sd(s['ci95'][1])}]", pv(s["p_value"]).replace("= ", ""),
                             f"{m_['menang']} / {m_['seri']} / {m_['kalah']}"])
    tab4 = b.table(["Koleksi", "Perbandingan", "Selisih rata-rata", "CI 95%", "p", "Menang / seri / kalah"], sig_rows,
                   [11, 21, 15, 22, 9, 18], ["left", "left", "center", "center", "center", "center"])
    d_sem_bm = {ds: H(ds, "semantic") - H(ds, "bm25") for ds in (sf, nf)}
    s42 = [
        b.para(f"Tabel 3 memuat keenam metrik untuk keempat skenario, dan Gambar 2 meringkas nDCG@10. Di antara skenario "
               f"yang memakai embedding atau keyphrase, hibrida memperoleh nDCG@10 tertinggi pada kedua koleksi, yaitu "
               f"{f4(H(sf, 'hybrid_weighted'))} pada SciFact dan {f4(H(nf, 'hybrid_weighted'))} pada NFCorpus, dibanding "
               f"{f4(H(sf, 'semantic'))} dan {f4(H(nf, 'semantic'))} untuk semantik saja serta {f4(H(sf, 'keyphrase'))} dan "
               f"{f4(H(nf, 'keyphrase'))} untuk keyphrase saja. Hibrida juga memperoleh Recall@100 tertinggi di antara keempat "
               f"skenario ({f4(H(sf, 'hybrid_weighted', 'recall@100'))} dan {f4(H(nf, 'hybrid_weighted', 'recall@100'))}), "
               f"meskipun selisihnya dengan semantik saja kecil dan tidak diuji signifikansinya; kenaikan ini kemungkinan "
               f"karena kandidat dari jalur keyphrase ikut membawa sebagian dokumen relevan ke 100 teratas."),
        b.caption("Tabel", 3, "Hasil keempat skenario pada split test (nilai terbaik per koleksi dicetak tebal)"),
        tab3, b.spacer_p(),
        b.figure(FIG / "g1_ndcg10_skenario.png", "image_uas_g1.png"),
        b.caption("Gambar", 2, "nDCG@10 keempat skenario pada split test SciFact dan NFCorpus."),
        b.para(f"Tabel 4 memuat hasil uji *paired bootstrap*. Keunggulan hibrida atas semantik saja signifikan pada kedua "
               f"koleksi, yaitu {sd(sig[(sf, 'semantic')]['mean_diff'])} pada SciFact (p {pv(sig[(sf, 'semantic')]['p_value'])}) "
               f"dan {sd(sig[(nf, 'semantic')]['mean_diff'])} pada NFCorpus (p {pv(sig[(nf, 'semantic')]['p_value'])}), dan "
               f"keunggulannya atas keyphrase saja juga signifikan (p {pv(sig[(sf, 'keyphrase')]['p_value'])}). Selisih "
               f"terhadap semantik saja kecil karena pada sebagian besar kueri nDCG@10 tidak berubah: "
               f"hibrida seri dengan semantik saja pada {msk[sf]['semantic']['seri']} dari {msk[sf]['semantic']['n']} kueri "
               f"SciFact dan {msk[nf]['semantic']['seri']} dari {msk[nf]['semantic']['n']} kueri NFCorpus, menang pada "
               f"{msk[sf]['semantic']['menang']} dan {msk[nf]['semantic']['menang']} kueri, serta kalah pada "
               f"{msk[sf]['semantic']['kalah']} dan {msk[nf]['semantic']['kalah']} kueri (Gambar 3)."),
        b.figure(FIG / "g3_selisih_per_query.png", "image_uas_g3.png", width_emu=4754880),   # 5.2 inci
        b.caption("Gambar", 3, "Selisih nDCG@10 per kueri test antara hibrida dan pembanding, diurutkan dari selisih "
                               "terbesar. Biru: hibrida lebih baik; merah: hibrida lebih buruk; latar abu-abu: seri."),
        b.caption("Tabel", 4, "Uji signifikansi *paired bootstrap* atas nDCG@10 (hibrida dikurangi pembanding) dan "
                              "jumlah kueri menang, seri, dan kalah"),
        tab4, b.spacer_p(),
        b.para(f"BM25 memperoleh nDCG@10 sedikit lebih tinggi daripada hibrida, yaitu {f4(H(sf, 'bm25'))} pada SciFact dan "
               f"{f4(H(nf, 'bm25'))} pada NFCorpus. Selisihnya ({sd(sig[(sf, 'bm25')]['mean_diff'])} dan "
               f"{sd(sig[(nf, 'bm25')]['mean_diff'])}) tidak signifikan karena selang kepercayaan 95% mencakup nol "
               f"(p {pv(sig[(sf, 'bm25')]['p_value'])} dan p {pv(sig[(nf, 'bm25')]['p_value'])}). Hasil per kueri juga "
               f"seimbang: hibrida menang pada {msk[sf]['bm25']['menang']} kueri dan kalah pada {msk[sf]['bm25']['kalah']} "
               f"kueri SciFact, serta menang pada {msk[nf]['bm25']['menang']} dan kalah pada {msk[nf]['bm25']['kalah']} "
               f"kueri NFCorpus. Temuan ini sejalan dengan Ren dkk. [5] bahwa *dense retriever* yang dipakai secara "
               f"*zero-shot* sering belum mampu mengungguli BM25, dan dengan Chen dkk. [6] yang mencatat kelemahan "
               f"*dense retriever* pada frasa penting dan entitas langka. Kueri SciFact berupa klaim ilmiah [21] yang dapat "
               f"memuat istilah teknis spesifik, seperti nama protein pada studi kasus di Subbab 4.5, yang dicocokkan BM25 "
               f"secara persis. Penambahan keyphrase menutup sebagian kesenjangan tersebut: selisih nDCG@10 semantik saja "
               f"terhadap BM25 adalah {sd(d_sem_bm[sf])} pada SciFact dan {sd(d_sem_bm[nf])} pada NFCorpus, sedangkan "
               f"selisih hibrida terhadap BM25 menjadi {sd(sig[(sf, 'bm25')]['mean_diff'])} dan "
               f"{sd(sig[(nf, 'bm25')]['mean_diff'])}."),
    ]
    b.replace_placeholder("[Isi: tabel nDCG@10, MAP@100, Recall, Precision, dan MRR", s42)

    # ---------- 4.3 Pengaruh Bobot Fusi
    flat = max(sweep[nf][a]["ndcg@10"] for a in (0.7, 0.8, 0.9)) - min(sweep[nf][a]["ndcg@10"] for a in (0.7, 0.8, 0.9))
    s43 = [
        b.figure(FIG / "g2_pengaruh_alpha.png", "image_uas_g2.png"),
        b.caption("Gambar", 4, "Pengaruh α terhadap nDCG@10. Garis biru: kueri penalaan (SciFact: train, NFCorpus: dev) "
                               "yang dipakai memilih α; garis oranye putus-putus: kueri test, hanya untuk analisis; garis "
                               "titik-titik: BM25 pada kueri test."),
        b.para(f"Gambar 4 menunjukkan nDCG@10 terhadap α. Pada kueri penalaan kedua koleksi, nDCG@10 naik ketika α bertambah "
               f"dari 0, mencapai puncak pada α = 0.9 ({f4(sweep[sf][0.9]['ndcg@10'])} untuk SciFact dan "
               f"{f4(sweep[nf][0.9]['ndcg@10'])} untuk NFCorpus), lalu turun pada α = 1, yaitu ketika hanya skor semantik "
               f"yang dipakai ({f4(sweep[sf][1.0]['ndcg@10'])} dan {f4(sweep[nf][1.0]['ndcg@10'])}). Nilai α = 0.9 berarti "
               f"sinyal semantik tetap dominan, sedangkan sinyal keyphrase berperan sebagai koreksi yang kecil tetapi "
               f"berguna: bobot 0.1 cukup untuk menaikkan dokumen yang berbagi istilah inti dengan kueri tanpa merusak urutan "
               f"dari embedding. Sebaliknya, bobot keyphrase yang besar menurunkan kualitas peringkat; pada α = 0.5 nDCG@10 "
               f"kueri penalaan hanya {f4(sweep[sf][0.5]['ndcg@10'])} dan {f4(sweep[nf][0.5]['ndcg@10'])}. Penyebabnya "
               f"kemungkinan adalah sinyal keyphrase yang jarang dan kasar: keyphrase saja memperoleh nDCG@10 nol pada "
               f"{pct(kp[sf]['query_proporsi_ndcg_keyphrase_nol'])} kueri test SciFact dan "
               f"{pct(kp[nf]['query_proporsi_ndcg_keyphrase_nol'])} kueri test NFCorpus (Subbab 4.4). Pada NFCorpus kurva penalaan cukup datar "
               f"antara α = 0.7 dan 0.9 (rentang {f4(flat)}), sehingga pilihan α tidak sensitif di sekitar nilai tersebut."),
        b.para("Kurva pada kueri test hanya dihitung untuk analisis setelah α dipilih dan tidak dipakai untuk memilih α. "
               "Bentuknya mengikuti kurva penalaan, dan nilai α terbaik pada kueri test juga 0.9 untuk kedua koleksi. Dengan "
               "demikian, bobot yang ditala pada data penalaan berpindah dengan baik ke data uji, sejalan dengan [18] [19] "
               "bahwa bobot kombinasi linear dapat ditala dengan sedikit data penilaian relevansi."),
    ]
    b.replace_placeholder("[Isi: grafik nDCG@10 terhadap nilai α", s43)

    # ---------- 4.4 Analisis Keyphrase
    import csv as _csv
    with (ROOT / "analisis" / "contoh_keyphrase.csv").open(encoding="utf-8") as fh:
        rows_ex = list(_csv.DictReader(fh))
    def kps(doc_id):
        got = [x["keyphrase"] for x in rows_ex if x["doc_id"] == doc_id]
        assert got, doc_id
        return got
    star, milk = kps("11244195"), kps("MED-2056")
    k_sf, k_nf = kp[sf], kp[nf]
    rows5 = [
        ["Rata-rata keyphrase per dokumen", f"{k_sf['dok_rata_keyphrase']:.2f}", f"{k_nf['dok_rata_keyphrase']:.2f}"],
        ["Rata-rata keyphrase per kueri test", f"{k_sf['query_rata_keyphrase']:.2f}", f"{k_nf['query_rata_keyphrase']:.2f}"],
        ["Keyphrase dokumen yang *absent*", pct(k_sf["dok_proporsi_absent"]), pct(k_nf["dok_proporsi_absent"])],
        ["Keyphrase kueri yang *absent*", pct(k_sf["query_proporsi_absent"]), pct(k_nf["query_proporsi_absent"])],
        ["Rata-rata dokumen relevan per kueri test", f"{k_sf['rata_dok_relevan_per_query_test']:.1f}",
         f"{k_nf['rata_dok_relevan_per_query_test']:.1f}"],
        ["Pasangan kueri–dokumen relevan dengan skor keyphrase > 0", pct(k_sf["proporsi_pasangan_relevan_skor_kp_positif"]),
         pct(k_nf["proporsi_pasangan_relevan_skor_kp_positif"])],
        ["Kueri dengan ≥ 1 dokumen relevan berskor keyphrase > 0", pct(k_sf["proporsi_query_min1_relevan_skor_kp_positif"]),
         pct(k_nf["proporsi_query_min1_relevan_skor_kp_positif"])],
        ["Kueri tanpa skor keyphrase pada semua dokumen", pct(k_sf["query_proporsi_skor_kp_nol"]),
         pct(k_nf["query_proporsi_skor_kp_nol"])],
        ["Kueri dengan nDCG@10 keyphrase saja = 0", pct(k_sf["query_proporsi_ndcg_keyphrase_nol"]),
         pct(k_nf["query_proporsi_ndcg_keyphrase_nol"])],
    ]
    tab5 = b.table(["Statistik", "SciFact", "NFCorpus"], rows5, [60, 20, 20], ["left", "center", "center"])
    q = lambda xs: ", ".join(f"'{x}'" for x in xs)
    s44 = [
        b.para(f"Tabel 5 merangkum keyphrase yang dibangkitkan BART-KPG. Setiap dokumen memperoleh rata-rata "
               f"{k_sf['dok_rata_keyphrase']:.2f} keyphrase pada SciFact dan {k_nf['dok_rata_keyphrase']:.2f} pada NFCorpus, "
               f"jauh di bawah batas {kc_sf['top_n']}, dan kueri memperoleh lebih sedikit, yaitu "
               f"{k_sf['query_rata_keyphrase']:.2f} dan {k_nf['query_rata_keyphrase']:.2f}, karena kueri jauh lebih pendek "
               f"daripada abstrak. Sebagai contoh, makalah SciFact tentang perangkat lunak STAR memperoleh keyphrase "
               f"{q(star)}, sedangkan artikel NFCorpus tentang alergi susu sapi memperoleh {q(milk)}."),
        b.caption("Tabel", 5, "Statistik keyphrase BART-KPG dan cakupan sinyal keyphrase pada kueri test"),
        tab5, b.spacer_p(),
        b.para(f"Sebuah keyphrase disebut *absent* jika, setelah dinormalisasi dengan aturan yang sama dengan sistem (huruf "
               f"kecil, *stopword* di tepi dibuang, dan *stemming*), frasa tersebut tidak muncul berurutan di judul dan "
               f"abstrak. Hanya {pct(k_sf['dok_proporsi_absent'])} keyphrase dokumen SciFact dan "
               f"{pct(k_nf['dok_proporsi_absent'])} keyphrase dokumen NFCorpus yang *absent*. Artinya, pada kedua koleksi "
               f"ini BART-KPG hampir selalu menyalin frasa yang sudah tertulis di dokumen, sehingga jalur keyphrase terutama "
               f"berfungsi sebagai penanda istilah inti yang diberi bobot, bukan sebagai sumber istilah baru di luar teks. "
               f"Proporsi *absent* pada keyphrase kueri lebih tinggi, yaitu {pct(k_sf['query_proporsi_absent'])} dan "
               f"{pct(k_nf['query_proporsi_absent'])}. Penelitian ini tidak mengukur apakah keyphrase kueri yang *absent* "
               f"tersebut membantu menjembatani *vocabulary gap*, sehingga hal itu belum dapat disimpulkan."),
        b.para(f"Model BART-KPG dilatih pada KP20k yang berdomain ilmu komputer, sedangkan kedua koleksi uji berdomain "
               f"biomedis, dan Boudin dkk. [11] mencatat perpindahan domain sebagai salah satu kesulitan keyphrase hasil "
               f"pembangkitan. Penelitian ini tidak memiliki keyphrase acuan untuk mengukur kualitas keyphrase secara "
               f"langsung, sehingga dampak perpindahan domain hanya dapat dilihat secara tidak langsung dari cakupan sinyal "
               f"keyphrase. Pada SciFact, {pct(k_sf['proporsi_pasangan_relevan_skor_kp_positif'])} pasangan kueri dan "
               f"dokumen relevan memiliki skor keyphrase lebih dari nol, sedangkan pada NFCorpus hanya "
               f"{pct(k_nf['proporsi_pasangan_relevan_skor_kp_positif'])}. Angka per pasangan NFCorpus tersebut dipengaruhi "
               f"oleh qrels yang padat, yaitu rata-rata {k_nf['rata_dok_relevan_per_query_test']:.1f} dokumen relevan per "
               f"kueri dibandingkan {k_sf['rata_dok_relevan_per_query_test']:.1f} pada SciFact, sehingga tidak dapat "
               f"disimpulkan sebagai kegagalan keyphrase semata. Jika dihitung per kueri, "
               f"{pct(k_sf['proporsi_query_min1_relevan_skor_kp_positif'])} kueri SciFact dan "
               f"{pct(k_nf['proporsi_query_min1_relevan_skor_kp_positif'])} kueri NFCorpus memiliki paling tidak satu "
               f"dokumen relevan dengan skor keyphrase lebih dari nol. Di sisi lain, "
               f"{pct(k_nf['query_proporsi_skor_kp_nol'])} kueri NFCorpus tidak memperoleh skor keyphrase pada dokumen mana "
               f"pun, dan nDCG@10 keyphrase saja bernilai nol pada {pct(k_sf['query_proporsi_ndcg_keyphrase_nol'])} kueri "
               f"SciFact dan {pct(k_nf['query_proporsi_ndcg_keyphrase_nol'])} kueri NFCorpus. Sinyal yang jarang ini "
               f"menjelaskan lemahnya keyphrase saja dan sejalan dengan bobot keyphrase yang kecil pada α terpilih."),
    ]
    b.replace_placeholder("[Isi: contoh keyphrase yang dibangkitkan", s44)

    # ---------- 4.5 Studi Kasus
    cases = {(c["dataset"], c["jenis"]): c for c in D["cases"] if c["utama"]}
    c1, c2 = cases[(sf, "hybrid_menolong")], cases[(nf, "hybrid_menolong")]
    c3, c4 = cases[(sf, "hybrid_kalah_bm25")], cases[(nf, "hybrid_kalah_bm25")]
    assert (c1["qid"], c2["qid"], c3["qid"], c4["qid"]) == ("1185", "PLAIN-681", "1024", "PLAIN-1473"), \
        "pemilihan studi kasus berubah: tinjau ulang narasi 4.5"
    rk = lambda c, s: c["dokumen_relevan_utama"]["peringkat"][s]
    fr = lambda c, s: str(rk(c, s)) if rk(c, s) else "–"
    assert c1["dokumen_relevan_utama"]["frasa_keyphrase_sama"] == ["kidney paired donation"]
    assert not c2["dokumen_relevan_utama"]["frasa_keyphrase_sama"] and \
        c2["dokumen_relevan_utama"]["kata_keyphrase_sama"] == ["betel", "nuts"]
    assert not c3["dokumen_relevan_utama"]["kata_keyphrase_sama"] and not c4["dokumen_relevan_utama"]["kata_keyphrase_sama"]
    assert "kohlrabi" in c4["dokumen_relevan_utama"]["judul"].lower() and rk(c4, "semantic") is None
    match = {
        c1["qid"]: "frasa sama: kidney paired donation",
        c2["qid"]: "kata sama: betel, nuts",
        c3["qid"]: "tidak ada frasa atau kata yang sama",
        c4["qid"]: "tidak ada; dokumen tidak memiliki keyphrase kohlrabi",
    }
    rows6 = []
    for c, jenis in ((c1, "Hibrida menolong"), (c2, "Hibrida menolong"), (c3, "Hibrida kalah dari BM25"),
                     (c4, "Hibrida kalah dari BM25")):
        qtxt = c["query"] if len(c["query"]) <= 70 else c["query"][:67].rstrip() + "..."
        rows6.append([jenis, f"{'SciFact' if c['dataset'] == sf else 'NFCorpus'} {c['qid']}: {qtxt}",
                      fr(c, "bm25"), fr(c, "semantic"), fr(c, "keyphrase"), fr(c, "hybrid_weighted"), match[c["qid"]]])
    tab6 = b.table(["Jenis", "Kueri", "BM25", "Semantik", "Keyphrase", "Hibrida", "Kecocokan keyphrase"], rows6,
                   [10, 31, 8, 10, 11, 9, 21], ["left", "left", "center", "center", "center", "center", "left"])
    d3 = c3["dokumen_relevan_utama"]
    d4 = c4["dokumen_relevan_utama"]
    s45 = [
        b.para("Tabel 6 merangkum empat kueri yang dipilih otomatis dari hasil per kueri: dua kueri dengan kenaikan "
               "nDCG@10 terbesar dari semantik saja ke hibrida, dan dua kueri dengan penurunan terbesar dari BM25 ke "
               "hibrida, masing-masing satu dari tiap koleksi. Kolom peringkat menunjukkan posisi dokumen relevan utama "
               "pada setiap skenario."),
        b.caption("Tabel", 6, "Studi kasus: peringkat dokumen relevan utama pada setiap skenario (– berarti di luar 100 "
                              "teratas)"),
        tab6, b.spacer_p(),
        b.para(f"Pada kueri SciFact {c1['qid']} tentang program *kidney paired donation*, keyphrase kueri dan keyphrase "
               f"dokumen relevan sama-sama memuat frasa 'kidney paired donation'. Dokumen relevan berada di peringkat "
               f"{fr(c1, 'semantic')} pada semantik saja dan naik ke peringkat {fr(c1, 'hybrid_weighted')} pada hibrida. "
               f"Pada kueri NFCorpus '{c2['query']}', tidak ada frasa yang sama utuh, tetapi kata 'betel' dan 'nuts' muncul "
               f"pada keyphrase dokumen relevan '{c2['dokumen_relevan_utama']['keyphrase_dokumen'][0]}', sehingga dokumen "
               f"tersebut naik dari peringkat {fr(c2, 'semantic')} ke peringkat {fr(c2, 'hybrid_weighted')}. Dokumen yang "
               f"ditempatkan semantik saja di peringkat 1 membahas kacang secara umum, yang dekat secara makna tetapi tidak "
               f"relevan. Kedua kasus menunjukkan peran keyphrase sebagai penguat istilah inti."),
        b.para(f"Dua kasus lainnya menunjukkan keterbatasan sistem. Kueri SciFact {c3['qid']} memuat istilah langka "
               f"'CTCF' dan 'anchor sites'. BM25 menempatkan dokumen relevan di peringkat {fr(c3, 'bm25')} karena kata-kata "
               f"tersebut muncul persis di dokumen, sedangkan semantik saja menempatkannya di peringkat {fr(c3, 'semantic')} "
               f"dan hibrida di peringkat {fr(c3, 'hybrid_weighted')}. Keyphrase dokumen relevan, seperti "
               f"'{d3['keyphrase_dokumen'][4]}' dan '{d3['keyphrase_dokumen'][5]}', tidak memuat frasa atau kata dari "
               f"keyphrase kueri, sehingga sinyal keyphrase tidak membantu. Kueri NFCorpus '{c4['query']}' adalah kasus yang "
               f"lebih ekstrem: kata tersebut tertulis pada judul dokumen relevan, tetapi BART-KPG hanya membangkitkan "
               f"{len(d4['keyphrase_dokumen'])} keyphrase ({q(d4['keyphrase_dokumen'])}) yang tidak memuatnya, dan "
               f"embedding tidak menempatkan dokumen tersebut di 100 teratas. Kedua kasus sesuai dengan catatan Chen dkk. "
               f"[6] tentang kelemahan *dense retriever* pada istilah langka, dan menunjukkan bahwa keyphrase yang "
               f"dibangkitkan tidak selalu mencakup semua istilah penting dokumen."),
    ]
    assert d3["keyphrase_dokumen"][4:6] == ["chromosome structure", "gene regulation"]
    b.replace_placeholder("[Isi: contoh kueri yang berhasil dan gagal", s45)

    # ---------- Bab 5 Kesimpulan
    ds_ = (sf, nf)
    s5 = [
        b.para(f"Penelitian ini membangun sistem temu kembali hibrida yang menggabungkan embedding all-MiniLM-L6-v2 dan "
               f"keyphrase hasil pembangkitan BART-KPG, lalu mengujinya pada SciFact dan NFCorpus dengan BM25 sebagai "
               f"pembanding. Jawaban atas pertanyaan penelitian pertama: penambahan jalur keyphrase meningkatkan kualitas "
               f"peringkat dibandingkan setiap jalur secara terpisah. nDCG@10 hibrida ({f4(H(sf, 'hybrid_weighted'))} dan "
               f"{f4(H(nf, 'hybrid_weighted'))}) lebih tinggi secara signifikan daripada semantik saja "
               f"({f4(H(sf, 'semantic'))} dan {f4(H(nf, 'semantic'))}) dan keyphrase saja ({f4(H(sf, 'keyphrase'))} dan "
               f"{f4(H(nf, 'keyphrase'))}), walaupun peningkatan atas semantik saja kecil "
               f"({sd(sig[(sf, 'semantic')]['mean_diff'])} dan {sd(sig[(nf, 'semantic')]['mean_diff'])}). Jawaban atas "
               f"pertanyaan kedua: bobot fusi terbaik adalah α = 0.9 pada kedua koleksi, dan kualitas peringkat turun bila "
               f"bobot keyphrase diperbesar, sehingga keyphrase paling berguna sebagai sinyal pelengkap yang kecil. Jawaban "
               f"atas pertanyaan ketiga: BM25 masih sedikit lebih tinggi daripada hibrida, tetapi selisihnya tidak "
               f"signifikan (p {pv(sig[(sf, 'bm25')]['p_value'])} dan p {pv(sig[(nf, 'bm25')]['p_value'])}), dan "
               f"penambahan keyphrase memperkecil selisih semantik saja terhadap BM25."),
        b.para(f"Ketiga kontribusi yang dijanjikan terpenuhi: rancangan pipeline dengan dua jalur utama dan BM25 sebagai "
               f"pembanding; implementasi lengkap dari pengindeksan sampai antarmuka pencarian dengan waktu pencarian hibrida "
               f"sekitar {w_(sf, 'Hybrid'):.1f} ms per kueri; serta evaluasi empat skenario pada dua koleksi BEIR dengan uji "
               f"signifikansi, analisis per kueri, analisis keyphrase, dan studi kasus."),
        b.para("Penelitian ini memiliki beberapa keterbatasan. Pertama, seluruh pengujian bersifat *zero-shot* karena encoder "
               "dan generator keyphrase tidak dilatih ulang pada koleksi uji. Kedua, hanya satu encoder berukuran kecil yang "
               "diuji. Ketiga, kedua koleksi berbahasa Inggris dan berdomain biomedis, sehingga hasilnya belum tentu berlaku "
               "untuk domain atau bahasa lain, termasuk dokumen berbahasa Indonesia. Keempat, keyphrase dibangkitkan oleh "
               "model yang dilatih pada domain ilmu komputer, jumlahnya rata-rata hanya sekitar empat per dokumen, dan "
               "kualitasnya tidak diukur terhadap keyphrase acuan. Rencana pengembangan meliputi pengujian encoder yang lebih "
               "besar atau khusus domain biomedis, generator keyphrase yang disesuaikan dengan domain dokumen, jumlah "
               "keyphrase per dokumen yang lebih banyak, pengukuran langsung manfaat keyphrase *absent* terhadap *vocabulary "
               "gap*, serta pengujian pada koleksi lain, termasuk koleksi berbahasa Indonesia."),
    ]
    b.replace_placeholder("[Diisi pada laporan UAS: jawaban atas tiga pertanyaan penelitian", s5)

    # ---------- Abstrak: hasil + kesimpulan
    p = b.d.find_p("[Hasil: isi setelah eksperimen UAS selesai")
    replace(p, "[Hasil: isi setelah eksperimen UAS selesai, 4 sampai 5 kalimat.]",
            f"Bobot fusi terpilih adalah α = 0.9 pada kedua koleksi. Pada split test, hibrida memperoleh nDCG@10 "
            f"{f4(H(sf, 'hybrid_weighted'))} pada SciFact dan {f4(H(nf, 'hybrid_weighted'))} pada NFCorpus, lebih tinggi "
            f"secara signifikan daripada pencarian semantik saja ({f4(H(sf, 'semantic'))}, p "
            f"{pv(sig[(sf, 'semantic')]['p_value'])}; {f4(H(nf, 'semantic'))}, p {pv(sig[(nf, 'semantic')]['p_value'])}) dan "
            f"pencarian keyphrase saja ({f4(H(sf, 'keyphrase'))} dan {f4(H(nf, 'keyphrase'))}). BM25 memperoleh nDCG@10 "
            f"sedikit lebih tinggi ({f4(H(sf, 'bm25'))} dan {f4(H(nf, 'bm25'))}), tetapi selisihnya dengan hibrida tidak "
            f"signifikan (p {pv(sig[(sf, 'bm25')]['p_value'])} dan p {pv(sig[(nf, 'bm25')]['p_value'])}). Hanya "
            f"{pct(kp[sf]['dok_proporsi_absent'])} dan {pct(kp[nf]['dok_proporsi_absent'])} keyphrase dokumen yang tidak "
            f"muncul di teks, sehingga sinyal keyphrase terutama berperan sebagai penguat istilah inti.")
    replace(p, " [Kesimpulan: isi setelah eksperimen UAS selesai, 1 sampai 2 kalimat.]",
            " Penambahan keyphrase hasil pembangkitan memberi peningkatan kecil tetapi signifikan pada pencarian semantik "
            "tanpa pelatihan ulang, dan selisih hibrida dengan BM25 tidak signifikan pada dua koleksi biomedis.")
    unhighlight(p)

    # ---------- Lingkungan implementasi
    env = r["lingkungan"]
    pk, hw = env["pustaka"], env["perangkat"]
    kag = env["kaggle"]
    secs = [wk["seconds"] for k in kag.values() for wk in k["workers"]]
    wk0 = kag[sf]["workers"][0]
    cores = hw["Total Number of Cores"].split(" ")[0]
    set_paragraph(b.d.find_p("Sistem diimplementasikan dengan Python."),
                  f"Sistem diimplementasikan dengan Python {pk['python']} memakai NumPy {pk['numpy']}, SciPy {pk['scipy']}, "
                  f"PyTorch {pk['torch']}, Transformers {pk['transformers']}, dan Sentence-Transformers "
                  f"{pk['sentence-transformers']}. Pembangkitan keyphrase untuk seluruh dokumen dan kueri dijalankan di Kaggle "
                  f"dengan dua GPU NVIDIA T4 (PyTorch {wk0['torch'].split('+')[0]}, Transformers {wk0['transformers']}), "
                  f"satu proses untuk tiap GPU, karena komputasinya besar; setiap proses selesai dalam {min(secs):.0f} sampai "
                  f"{max(secs):.0f} detik. Pengindeksan, pencarian, dan evaluasi berjalan pada laptop {hw['Model Name']} "
                  f"dengan cip {hw['Chip']} ({cores} inti CPU) dan memori {hw['Memory']}, memakai akselerasi GPU Apple "
                  f"(MPS) untuk encoder. Antarmuka pencarian dibuat dengan Streamlit {pk['streamlit']}.",
                  b.body_rpr)


def bagian_c(b: Builder) -> None:
    d = b.d
    # Kontribusi tiap tim: satu sumber kebenaran di PEMBAGIAN_TUGAS.md (antara penanda kontribusi:mulai/selesai)
    sumber = (ROOT / "PEMBAGIAN_TUGAS.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- kontribusi:mulai -->\n(.+?)\n<!-- kontribusi:selesai -->", sumber, re.S)
    if not m:
        raise SystemExit("GALAT: penanda kontribusi tidak ditemukan di PEMBAGIAN_TUGAS.md")
    p = d.find_p("Semua penulis berkontribusi setara pada penulisan makalah ini.")
    set_paragraph(p, " ".join(m.group(1).split()))
    # Singkatan: tambah yang kini muncul, urut alfabet
    anchor = d.find_p("BART: Bidirectional and Auto-Regressive Transformers")
    proto = anchor
    existing = []
    el = anchor
    while el is not None and re.match(r"^[A-Za-z0-9]+: ", ptext(el) or ""):
        existing.append(el)
        el = el.getnext()
    new = {"CI": "Confidence Interval", "GPU": "Graphics Processing Unit", "MPS": "Metal Performance Shaders"}
    entries = {ptext(e).split(":")[0]: ptext(e) for e in existing}
    for k, v in new.items():
        entries[k] = f"{k}: {v}"
    base_rpr = [r for r in proto.findall(w("r")) if run_text(r)][0].find(w("rPr"))
    first = existing[0]
    for e in existing[1:]:
        e.getparent().remove(e)
    keys = sorted(entries, key=lambda s: s.lower())
    set_paragraph(first, entries[keys[0]], base_rpr)
    prev = first
    for k in keys[1:]:
        np_ = clone_para(proto, entries[k], base_rpr)
        prev.addnext(np_)
        prev = np_


def main() -> int:
    D = load_data()
    doc = Docx(BASE)
    b = Builder(doc)
    bagian_a(b, D)
    bagian_b(b, D)
    bagian_c(b)
    doc.save(OUT)
    print(f"Laporan ditulis: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
