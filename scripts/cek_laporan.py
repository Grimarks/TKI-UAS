#!/usr/bin/env python3
"""Pemeriksaan akhir Laporan_UAS_Kelompok1.docx.

  1. Sitasi: setiap [n] di teks ada di daftar pustaka dan setiap pustaka dikutip.
  2. Aturan pustaka: >= 15 artikel, >= 80% terbitan 2022 ke atas, tanpa arXiv.
  3. Tidak ada highlight kuning dan placeholder [..] selain yang diizinkan.
  4. Nomor Gambar/Tabel/persamaan berurutan dan setiap gambar/tabel dirujuk di teks.
  5. Setiap angka hasil di laporan cocok dengan analisis/ringkasan.json, results/*/metrics.json, atau konfigurasi.
  6. (opsional, --pdf) jumlah halaman lewat LibreOffice.

    ../.venv/bin/python scripts/cek_laporan.py [--pdf]
Keluar dengan kode 1 bila ada yang gagal.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT.parent / "Laporan" / "Laporan_UAS_Kelompok1.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ALLOWED_PLACEHOLDERS = {"[tautan repositori]"}
# angka desimal yang memang konstanta metode/konfigurasi, bukan hasil eksperimen
CONSTANTS = {"0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "0.25", "1.2", "0.75", "0.05",
             "0.001", "0.0", "95%"}          # 95% = taraf selang kepercayaan


def w(t: str) -> str:
    return f"{{{W}}}{t}"


def paragraphs(doc) -> list[tuple[str, object]]:
    out = []
    for p in doc.iter(w("p")):
        out.append(("".join(t.text or "" for t in p.iter(w("t"))), p))
    return out


def walk_numbers(obj, acc: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            walk_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            walk_numbers(v, acc)
    elif isinstance(obj, str):
        for m in re.findall(r"\d+(?:\.\d+)?", obj):
            acc.append(float(m))


def source_numbers() -> list[float]:
    acc: list[float] = []
    walk_numbers(json.loads((ROOT / "analisis" / "ringkasan.json").read_text()), acc)
    walk_numbers(json.loads((ROOT / "analisis" / "studi_kasus.json").read_text()), acc)   # peringkat & teks kueri
    for ds in ("scifact", "nfcorpus"):
        m = json.loads((ROOT / "results" / f"{ds}_bart" / "metrics.json").read_text())
        walk_numbers(m, acc)
        walk_numbers(yaml.safe_load((ROOT / "configs" / f"{ds}_bart.yaml").read_text()), acc)
        for s in m["significance"]:                       # selisih turunan yang dilaporkan
            acc.append(s["mean_diff"])
        sc = m["scenarios"]
        acc.append(sc["semantic"]["metrics"]["ndcg@10"] - sc["bm25"]["metrics"]["ndcg@10"])
        tune = {r["alpha"]: r["ndcg@10"] for r in m["alpha_sweep_tune"]}
        acc.append(max(tune[a] for a in (0.7, 0.8, 0.9)) - min(tune[a] for a in (0.7, 0.8, 0.9)))
    return acc


def matches(token: str, nums: list[float]) -> bool:
    neg = token.startswith(("−", "-"))
    signed = token.startswith(("+", "−", "-"))
    t = token.lstrip("+−-")
    is_pct = t.endswith("%")
    t = t.rstrip("%")
    dec = len(t.split(".")[1]) if "." in t else 0
    val = float(t) * (-1 if neg else 1)
    for v in nums:
        x = v * 100 if is_pct else v
        if round(x, dec) == round(val, dec):
            return True
        # angka tanpa tanda (mis. "selisihnya 0.0163") boleh cocok dengan nilai mutlak; angka bertanda harus sama tandanya
        if not signed and round(abs(x), dec) == round(val, dec):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="render ke PDF dengan LibreOffice untuk menghitung halaman")
    args = ap.parse_args(argv)
    z = zipfile.ZipFile(DOCX)
    doc = etree.fromstring(z.read("word/document.xml"))
    paras = paragraphs(doc)
    texts = [t for t, _ in paras]
    fails: list[str] = []

    i_ref = next(i for i, t in enumerate(texts) if t.strip() == "DAFTAR PUSTAKA")
    i_hasil = next(i for i, t in enumerate(texts) if t.strip() == "HASIL DAN DISKUSI")
    i_dekl = next(i for i, t in enumerate(texts) if t.strip() == "DEKLARASI")
    body = "\n".join(texts[:i_ref])
    refs = [t for t in texts[i_ref + 1:] if t.strip()]

    # 1. sitasi
    cited = {int(n) for grp in re.findall(r"\[(\d+(?:\]\s*\[\d+)*)\]", body) for n in re.findall(r"\d+", grp)}
    n_ref = len(refs)
    missing = sorted(c for c in cited if c > n_ref or c < 1)
    uncited = sorted(set(range(1, n_ref + 1)) - cited)
    print(f"[1] sitasi: {len(cited)} nomor dikutip, {n_ref} pustaka")
    if missing:
        fails.append(f"sitasi tanpa pustaka: {missing}")
    if uncited:
        fails.append(f"pustaka tidak dikutip: {uncited}")

    # 2. aturan pustaka
    years = []
    for r in refs:
        ys = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", r.split("https://")[0])]
        years.append(max(ys) if ys else 0)
    recent = sum(1 for y in years if y >= 2022)
    arxiv = [r[:50] for r in refs if "arxiv" in r.lower()]
    print(f"[2] pustaka: {n_ref} total, {recent} terbitan >= 2022 ({100 * recent / n_ref:.0f}%), "
          f"{n_ref - recent} sebelum 2022, arXiv: {len(arxiv)}")
    if n_ref < 15 or recent / n_ref < 0.8 or arxiv:
        fails.append("aturan pustaka dosen tidak terpenuhi")

    # 3. highlight & placeholder
    hl = [("".join(t.text or "" for t in h.getparent().getparent().iter(w("t"))))
          for h in doc.iter(w("highlight"))]
    ph = [m for m in re.findall(r"\[[^\]]+\]", "\n".join(texts))
          if not re.fullmatch(r"\[\d+\]", m) and not re.fullmatch(r"\[[+−-]\d.*\]", m)]
    bad_ph = sorted(set(ph) - ALLOWED_PLACEHOLDERS)
    print(f"[3] highlight: {len(hl)} run {sorted(set(hl))}; placeholder: {sorted(set(ph))}")
    if any(h not in ALLOWED_PLACEHOLDERS for h in hl):
        fails.append(f"highlight tersisa di luar yang diizinkan: {hl}")
    if bad_ph:
        fails.append(f"placeholder tersisa: {bad_ph}")

    # 4. penomoran
    for kind in ("Gambar", "Tabel"):
        caps = [int(m) for t in texts for m in re.findall(rf"^{kind} (\d+)\.", t)]
        refs_in_text = {int(m) for t in texts if not re.match(rf"^{kind} \d+\.", t)
                        for m in re.findall(rf"{kind} (\d+)", t)}
        print(f"[4] {kind}: keterangan {caps}, dirujuk {sorted(refs_in_text)}")
        if caps != list(range(1, len(caps) + 1)):
            fails.append(f"nomor {kind} tidak berurutan: {caps}")
        if set(caps) - refs_in_text:
            fails.append(f"{kind} tidak dirujuk di teks: {sorted(set(caps) - refs_in_text)}")
    eq = [int(t.strip()[1:-1]) for t in texts if re.fullmatch(r"\(\d+\)", t.strip())]
    eq_refs = {int(m) for t in texts if not re.fullmatch(r"\(\d+\)", t.strip()) for m in re.findall(r"\((\d)\)", t)}
    print(f"[4] persamaan: {eq}, dirujuk {sorted(eq_refs)}")
    if eq != list(range(1, len(eq) + 1)) or eq_refs - set(eq):
        fails.append("nomor persamaan tidak berurutan atau ada rujukan ke persamaan yang tidak ada")

    # 5. angka
    nums = source_numbers()
    abstrak = next(t for t in texts if t.startswith("Pencarian dokumen akademik yang hanya"))
    region = "\n".join([abstrak] + texts[i_hasil:i_dekl])
    region = re.sub(r"\b\d+\.\d+\.\d+\b", " ", region)                # versi pustaka diperiksa terpisah
    region = re.sub(r"\[\d+\]", " ", region)                           # nomor sitasi
    region = re.sub(r"(Subbab|subbab)\s+\d+\.\d+", " ", region)
    region = re.sub(r"\b(SciFact|NFCorpus) (\d+|PLAIN-\d+)\b", " ", region)   # id kueri
    region = re.sub(r"[A-Za-z]+@\d+", " ", region)                       # nama metrik (nDCG@10)
    region = re.sub(r"\b\d+[-–]\w+", " ", region)                         # mis. 4-hydroxyglucobrassicin
    tokens = re.findall(r"[+−-]?\d+\.\d+%?|\b\d+(?:\.\d+)?%", region)
    bad = sorted({t for t in tokens if t.lstrip("+−-") not in CONSTANTS and not matches(t, nums)})
    ints = [t for t in re.findall(r"(?<![\d.])\d{2,}(?![\d.%])", region)]
    bad_int = sorted({t for t in ints if not matches(t, nums)}, key=int)
    print(f"[5] angka: {len(tokens)} desimal/persen dan {len(ints)} bilangan bulat diperiksa di Abstrak + Bab 4 + Bab 5")
    if bad:
        fails.append(f"angka desimal tanpa sumber: {bad}")
    if bad_int:
        fails.append(f"bilangan bulat tanpa sumber: {bad_int}")
    env = json.loads((ROOT / "analisis" / "lingkungan.json").read_text())
    known_versions = set(env["pustaka"].values()) | {w_[k].split("+")[0] for v in env["kaggle"].values()
                                                     for w_ in v["workers"] for k in ("torch", "transformers")}
    versions = set(re.findall(r"\b\d+\.\d+\.\d+\b", "\n".join(texts[:i_ref])))
    unknown_v = sorted(versions - known_versions)
    print(f"[5] versi pustaka di teks: {sorted(versions)}")
    if unknown_v:
        fails.append(f"versi tanpa sumber: {unknown_v}")

    # 6. halaman
    if args.pdf and shutil.which("soffice"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, str(DOCX)],
                           capture_output=True, timeout=300)
            pdf = Path(tmp) / (DOCX.stem + ".pdf")
            info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
            pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        print(f"[6] halaman (LibreOffice): {pages}")
        if not 10 <= pages <= 20:
            fails.append(f"jumlah halaman {pages} di luar 10–20")

    print()
    if fails:
        print("GAGAL:")
        for f in fails:
            print("  -", f)
        return 1
    print("SEMUA PEMERIKSAAN LULUS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
