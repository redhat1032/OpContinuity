"""Offline tests for scripts/fetch_docs.py (no network access)."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import fetch_docs


def test_real_manifest_is_valid_and_excludes_tm_31_210():
    docs = fetch_docs.load_manifest()
    names = {d["filename"] for d in docs}
    assert len(docs) == 5
    assert not any("31-210" in n or "munitions" in n.lower() for n in names)
    for d in docs:
        assert d["url"] is None or d["url"].startswith("https://")
    raw = json.loads(fetch_docs.MANIFEST.read_text())
    assert any("31-210" in e["filename"] for e in raw["excluded"])


def test_sources_md_lists_every_manifest_hash():
    text = (Path(__file__).parents[1] / "SOURCES.md").read_text()
    for d in fetch_docs.load_manifest():
        assert d["sha256"] in text


def _write_manifest(tmp_path, docs, excluded=()):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"documents": docs, "excluded": list(excluded)}))
    return path


def _entry(name="a.pdf", content=b"%PDF-1.4 test", url="https://example.org/a.pdf"):
    return {
        "filename": name,
        "title": "T",
        "sha256": hashlib.sha256(content).hexdigest(),
        "license": "PD",
        "purpose": "test",
        "url": url,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update(filename="../evil.pdf"),
        lambda e: e.update(filename="a.exe"),
        lambda e: e.update(sha256="xyz"),
        lambda e: e.update(url="http://insecure.example/a.pdf"),
        lambda e: e.pop("license"),
    ],
)
def test_manifest_rejects_bad_entries(tmp_path, mutate):
    e = _entry()
    mutate(e)
    with pytest.raises(fetch_docs.ManifestError):
        fetch_docs.load_manifest(_write_manifest(tmp_path, [e]))


def test_manifest_rejects_excluded_file(tmp_path):
    path = _write_manifest(tmp_path, [_entry()], excluded=[{"filename": "a.pdf"}])
    with pytest.raises(fetch_docs.ManifestError):
        fetch_docs.load_manifest(path)


def _fake_downloader(content):
    def dl(_url, dest):
        dest.write_bytes(content)

    return dl


def test_download_verified(tmp_path):
    content = b"%PDF-1.4 good"
    assert (
        fetch_docs.process(
            _entry(content=content), tmp_path, False, False, _fake_downloader(content)
        )
        == "downloaded"
    )
    assert (tmp_path / "a.pdf").read_bytes() == content


def test_download_mismatch_is_quarantined(tmp_path):
    entry = _entry(content=b"expected")
    result = fetch_docs.process(entry, tmp_path, False, False, _fake_downloader(b"tampered"))
    assert result == "mismatch"
    assert not (tmp_path / "a.pdf").exists()
    assert (tmp_path / "a.pdf.mismatch").exists()


def test_existing_file_checked(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"other")
    assert fetch_docs.process(_entry(), tmp_path, True, False) == "mismatch"


def test_manual_entry_missing(tmp_path):
    assert fetch_docs.process(_entry(url=None), tmp_path, False, False) == "manual-missing"


def test_main_exit_codes(tmp_path):
    content = b"%PDF-1.4 x"
    docs = tmp_path / "docs"
    docs.mkdir()
    manifest = _write_manifest(tmp_path, [_entry(content=content, url=None)])
    argv = ["--manifest", str(manifest), "--docs", str(docs)]
    assert fetch_docs.main(argv) == 0
    assert fetch_docs.main([*argv, "--strict"]) == 3
    (docs / "a.pdf").write_bytes(b"wrong")
    assert fetch_docs.main(argv) == 1
    (docs / "a.pdf").write_bytes(content)
    assert fetch_docs.main([*argv, "--strict", "--verify-only"]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert fetch_docs.main(["--manifest", str(bad), "--docs", str(docs)]) == 2
