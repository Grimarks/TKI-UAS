import io
import zipfile

import pytest

from hybrid_ir.download import DownloadError, download_beir, safe_extract_zip, verify_md5


def _zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def test_extract_ok(tmp_path):
    z = tmp_path / "a.zip"
    _zip(z, {"scifact/corpus.jsonl": "{}", "scifact/qrels/test.tsv": "x"})
    files = safe_extract_zip(z, tmp_path / "out")
    assert (tmp_path / "out" / "scifact" / "corpus.jsonl").is_file() and len(files) == 2


@pytest.mark.parametrize("name", ["../evil.txt", "/abs.txt", "a/../../evil.txt", "a\\..\\evil.txt"])
def test_zip_slip_rejected(tmp_path, name):
    z = tmp_path / "bad.zip"
    _zip(z, {name: "boom"})
    with pytest.raises(DownloadError):
        safe_extract_zip(z, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_symlink_rejected(tmp_path):
    z = tmp_path / "link.zip"
    with zipfile.ZipFile(z, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(DownloadError, match="symlink"):
        safe_extract_zip(z, tmp_path / "out")


def test_too_many_entries(tmp_path):
    z = tmp_path / "many.zip"
    _zip(z, {f"f{i}.txt": "x" for i in range(300)})
    with pytest.raises(DownloadError, match="terlalu banyak"):
        safe_extract_zip(z, tmp_path / "out")


def test_md5_mismatch(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    with pytest.raises(DownloadError, match="MD5"):
        verify_md5(f, "0" * 32)
    verify_md5(f, "5d41402abc4b2a76b9719d911017c592")


def test_unknown_dataset_rejected(tmp_path):
    with pytest.raises(DownloadError, match="daftar putih"):
        download_beir("../../etc", tmp_path)
    with pytest.raises(DownloadError):
        download_beir("msmarco", tmp_path)


class _FakeResp:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code, self._content, self.headers = status, content, headers or {}

    def iter_content(self, chunk_size=1 << 20):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]

    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _zip_bytes(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, d in entries.items():
            zf.writestr(n, d)
    return buf.getvalue()


def test_full_download_flow_with_mock(tmp_path, monkeypatch):
    import hashlib
    from hybrid_ir import download as dl
    data = _zip_bytes({"scifact/corpus.jsonl": '{"_id":"1","text":"x"}\n', "scifact/queries.jsonl": "", "scifact/qrels/test.tsv": ""})
    monkeypatch.setitem(dl.BEIR_MD5, "scifact", hashlib.md5(data).hexdigest())
    seen = []
    def fake_get(url, **kw):
        seen.append((url, kw.get("allow_redirects")))
        return _FakeResp(200, data)
    monkeypatch.setattr(dl.requests, "get", fake_get)
    out = download_beir("scifact", tmp_path)
    assert (out / "corpus.jsonl").is_file() and seen[0][1] is False
    assert download_beir("scifact", tmp_path) == out and len(seen) == 1     # sudah ada: tidak mengunduh ulang


def test_checksum_mismatch_blocks_extraction(tmp_path, monkeypatch):
    from hybrid_ir import download as dl
    monkeypatch.setattr(dl.requests, "get", lambda url, **kw: _FakeResp(200, _zip_bytes({"scifact/corpus.jsonl": "x"})))
    with pytest.raises(DownloadError, match="MD5"):
        download_beir("scifact", tmp_path)
    assert not (tmp_path / "scifact").exists()


def test_redirect_to_other_host_rejected(tmp_path, monkeypatch):
    from hybrid_ir import download as dl
    monkeypatch.setattr(dl.requests, "get", lambda url, **kw: _FakeResp(302, headers={"Location": "https://evil.example/x.zip"}))
    with pytest.raises(DownloadError, match="luar host resmi"):
        download_beir("scifact", tmp_path)


def test_oversize_download_rejected(tmp_path, monkeypatch):
    from hybrid_ir import download as dl
    monkeypatch.setattr(dl, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(dl.requests, "get", lambda url, **kw: _FakeResp(200, b"x" * 100))
    with pytest.raises(DownloadError, match="melebihi batas"):
        download_beir("scifact", tmp_path)


def test_http_error_status(tmp_path, monkeypatch):
    from hybrid_ir import download as dl
    monkeypatch.setattr(dl.requests, "get", lambda url, **kw: _FakeResp(404))
    with pytest.raises(DownloadError, match="HTTP 404"):
        download_beir("scifact", tmp_path)
