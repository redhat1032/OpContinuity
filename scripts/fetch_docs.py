#!/usr/bin/env python3
"""Download the source PDFs listed in scripts/sources.json and verify SHA-256.

Standard library only. Exit codes:
  0  every file is present (or was downloaded) and matches its hash
  1  a hash mismatch or a download error occurred
  2  the manifest is invalid
  3  --strict was given and a manual-only file (no URL) is missing

Usage:
  python scripts/fetch_docs.py                 # download missing files into docs/
  python scripts/fetch_docs.py --verify-only   # only check files already present
  python scripts/fetch_docs.py --force         # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).with_name("sources.json")
ALLOWED_SCHEMES = {"https"}
USER_AGENT = "OpContinuity-fetch/1.0 (+https://github.com/redhat1032/OpContinuity)"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = ("filename", "title", "sha256", "license", "purpose")


class ManifestError(ValueError):
    """The manifest is malformed."""


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    """Load and validate the manifest, returning the list of documents."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read manifest {path}: {exc}") from exc
    docs = data.get("documents")
    if not isinstance(docs, list) or not docs:
        raise ManifestError("Manifest has no 'documents' list")
    excluded = {e.get("filename") for e in data.get("excluded", [])}
    seen: set[str] = set()
    for doc in docs:
        for key in _REQUIRED:
            if not doc.get(key):
                raise ManifestError(f"Entry {doc.get('filename')!r} is missing '{key}'")
        name = doc["filename"]
        if name != os.path.basename(name) or not name.lower().endswith(".pdf"):
            raise ManifestError(f"Unsafe or non-PDF filename: {name!r}")
        if name in seen:
            raise ManifestError(f"Duplicate filename: {name!r}")
        if name in excluded:
            raise ManifestError(f"{name!r} is listed as excluded")
        seen.add(name)
        if not _SHA256_RE.match(doc["sha256"]):
            raise ManifestError(f"Bad sha256 for {name!r}")
        url = doc.get("url")
        if url is not None:
            scheme = urllib.parse.urlparse(url).scheme
            if scheme not in ALLOWED_SCHEMES:
                raise ManifestError(f"URL for {name!r} must use https, got {url!r}")
    return docs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Download ``url`` to ``dest`` atomically (temp file, then rename)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".part-", suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def process(doc: dict, docs_dir: Path, verify_only: bool, force: bool, downloader=download) -> str:
    """Handle one manifest entry. Returns one of: ok, downloaded, manual-missing,
    mismatch, error."""
    dest = docs_dir / doc["filename"]
    expected = doc["sha256"]
    url = doc.get("url")
    if dest.exists() and not force:
        actual = sha256_file(dest)
        if actual == expected:
            print(f"OK        {dest.name}")
            return "ok"
        print(f"MISMATCH  {dest.name}: expected {expected}, got {actual}", file=sys.stderr)
        return "mismatch"
    if url is None:
        print(f"MANUAL    {dest.name}: no verified source URL; add it to {docs_dir}/ by hand")
        return "manual-missing"
    if verify_only:
        print(f"MISSING   {dest.name} (run without --verify-only to download)", file=sys.stderr)
        return "error"
    print(f"FETCH     {dest.name} <- {url}")
    try:
        downloader(url, dest)
    except Exception as exc:  # network errors, HTTP errors, disk errors
        print(f"ERROR     {dest.name}: {exc}", file=sys.stderr)
        return "error"
    actual = sha256_file(dest)
    if actual != expected:
        bad = dest.with_suffix(dest.suffix + ".mismatch")
        os.replace(dest, bad)
        print(
            f"MISMATCH  {dest.name}: expected {expected}, got {actual}; kept as {bad.name}",
            file=sys.stderr,
        )
        return "mismatch"
    print(f"VERIFIED  {dest.name}")
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--docs", default="docs", help="destination folder (default: docs)")
    p.add_argument("--manifest", default=str(MANIFEST), help="path to sources.json")
    p.add_argument("--verify-only", action="store_true", help="do not download, only verify")
    p.add_argument("--force", action="store_true", help="re-download files that already exist")
    p.add_argument("--strict", action="store_true", help="fail if manual-only files are missing")
    args = p.parse_args(argv)

    try:
        docs = load_manifest(Path(args.manifest))
    except ManifestError as exc:
        print(f"Invalid manifest: {exc}", file=sys.stderr)
        return 2

    docs_dir = Path(args.docs)
    docs_dir.mkdir(parents=True, exist_ok=True)
    results = [process(d, docs_dir, args.verify_only, args.force) for d in docs]

    if "mismatch" in results or "error" in results:
        return 1
    if args.strict and "manual-missing" in results:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
