"""Pengunduh dataset BEIR yang aman (perintah `hybrid-ir download`).

Peran di pipeline (Gambar 1)
    Persiapan sebelum Fase 1: mengisi `data/<nama>/` yang menjadi kotak "Korpus dokumen" dan "Data uji BEIR".
    Tidak ikut dijalankan saat build/search/evaluate.

Masukan  : nama dataset pada daftar putih (scifact, nfcorpus, scidocs).
Keluaran : folder `data/<nama>/` berisi corpus.jsonl, queries.jsonl, dan qrels/.

Pengamanan yang diterapkan:
  * Hanya dataset pada daftar putih (allowlist) dan hanya satu host resmi (HTTPS).
  * Ukuran unduhan dibatasi, ada timeout, dan checksum MD5 dicocokkan dengan nilai resmi BEIR
    sebelum arsip dibuka.
  * Ekstraksi ZIP menolak path absolut / '..' (zip-slip), symlink, dan arsip yang membengkak
    (batas jumlah berkas dan total ukuran terdekompresi).
MD5 dipakai di sini hanya untuk memeriksa keutuhan unduhan terhadap nilai yang dipublikasikan BEIR,
bukan sebagai jaminan keamanan kriptografis.
"""
from __future__ import annotations

import hashlib
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

import requests

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
ALLOWED_HOST = "public.ukp.informatik.tu-darmstadt.de"
# MD5 resmi dari README BEIR
BEIR_MD5 = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "scidocs": "38121350fc3a4d2f48850f6aff52e4a9",
}
MAX_DOWNLOAD_BYTES = 400 * 1024 * 1024
MAX_UNZIPPED_BYTES = 2 * 1024 * 1024 * 1024
MAX_ZIP_ENTRIES = 200


class DownloadError(RuntimeError):
    """Unduhan atau ekstraksi gagal, atau melanggar batas keamanan."""


def verify_md5(path: Path, expected: str) -> None:
    """Hentikan proses bila MD5 berkas berbeda dari nilai resmi BEIR."""
    h = hashlib.md5()  # noqa: S324 - pemeriksaan keutuhan terhadap nilai terbitan BEIR
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise DownloadError(f"checksum MD5 tidak cocok untuk {path.name}: {h.hexdigest()} != {expected}")


def safe_extract_zip(zip_path: Path, dest: Path) -> list[Path]:
    """Ekstrak ZIP ke `dest` dengan menolak entri berbahaya. Mengembalikan daftar berkas hasil ekstraksi."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise DownloadError(f"arsip memuat terlalu banyak entri ({len(infos)} > {MAX_ZIP_ENTRIES})")
        for info in infos:
            name = info.filename
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or name.startswith(("/", "\\")) or "\\" in name:
                raise DownloadError(f"entri arsip berbahaya ditolak: {name!r}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DownloadError(f"symlink dalam arsip ditolak: {name!r}")
            target = (dest / name).resolve()
            if dest != target and dest not in target.parents:
                raise DownloadError(f"entri keluar dari folder tujuan: {name!r}")
            total += info.file_size
            if total > MAX_UNZIPPED_BYTES:
                raise DownloadError("ukuran terdekompresi melebihi batas aman")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(target)
    return extracted


def _open_stream(url: str, timeout: int, max_redirects: int = 3):
    """GET dengan redirect manual yang hanya boleh menuju HTTPS pada host resmi yang sama."""
    for _ in range(max_redirects + 1):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            raise DownloadError(f"URL di luar host resmi ditolak: {url}")
        resp = requests.get(url, stream=True, timeout=timeout, allow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            url = urljoin(url, resp.headers["Location"])
            resp.close()
            continue
        return resp
    raise DownloadError("terlalu banyak pengalihan (redirect)")


def download_beir(name: str, dest_dir: str | Path = "data", timeout: int = 60) -> Path:
    """Unduh dataset BEIR `name` ke `<dest_dir>/<name>/` (berisi corpus.jsonl, queries.jsonl, qrels/)."""
    if name not in BEIR_MD5:
        raise DownloadError(f"dataset '{name}' tidak ada di daftar putih: {sorted(BEIR_MD5)}")
    final = Path(dest_dir) / name
    if (final / "corpus.jsonl").is_file():
        return final
    url = BEIR_URL.format(name=name)
    assert url.startswith(f"https://{ALLOWED_HOST}/")
    final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=final.parent) as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / f"{name}.zip"
        with _open_stream(url, timeout) as resp:
            if resp.status_code != 200:
                raise DownloadError(f"unduhan gagal: HTTP {resp.status_code} dari {ALLOWED_HOST}")
            size = 0
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise DownloadError("ukuran unduhan melebihi batas aman")
                    fh.write(chunk)
        verify_md5(zip_path, BEIR_MD5[name])
        safe_extract_zip(zip_path, tmp_dir / "out")
        inner = tmp_dir / "out" / name
        if not (inner / "corpus.jsonl").is_file():
            raise DownloadError("struktur arsip tidak sesuai harapan (corpus.jsonl tidak ditemukan)")
        shutil.move(str(inner), str(final))
    return final
