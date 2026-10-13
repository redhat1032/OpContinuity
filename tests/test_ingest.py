from pathlib import Path

import pytest
from langchain_core.documents import Document

from opcontinuity.ingest import find_pdfs, load_documents, load_pdf, split_documents


def test_find_pdfs_ignores_non_pdfs(docs_dir: Path):
    assert [p.name for p in find_pdfs(docs_dir)] == ["fire.pdf", "shelter.pdf"]


def test_find_pdfs_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_pdfs(tmp_path / "nope")


def test_load_pdf_one_document_per_page(docs_dir: Path):
    docs = load_pdf(docs_dir / "shelter.pdf")
    assert len(docs) == 2
    assert "lean-to" in docs[0].page_content
    assert "debris hut" in docs[1].page_content
    assert docs[1].metadata == {"source": str(docs_dir / "shelter.pdf"), "page": 1}


def test_load_documents_reads_all_and_skips_corrupt(docs_dir: Path):
    (docs_dir / "broken.pdf").write_bytes(b"this is not really a pdf")
    docs = load_documents(docs_dir)
    sources = {Path(d.metadata["source"]).name for d in docs}
    assert sources == {"fire.pdf", "shelter.pdf"}
    assert len(docs) == 3


def test_load_documents_empty_dir(tmp_path: Path):
    assert load_documents(tmp_path) == []


def test_split_documents_respects_size_and_keeps_metadata():
    text = " ".join(f"word{i}" for i in range(400))
    docs = [Document(page_content=text, metadata={"source": "x.pdf", "page": 7})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata == {"source": "x.pdf", "page": 7} for c in chunks)


def test_split_documents_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_documents([], chunk_size=100, chunk_overlap=100)
