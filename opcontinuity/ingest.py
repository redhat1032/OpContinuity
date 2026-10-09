"""PDF loading and chunking.

These functions are pure (no network, no Ollama) so they can be unit tested.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def find_pdfs(docs_dir: str | Path) -> list[Path]:
    """Return the PDF files directly inside ``docs_dir``, sorted by name."""
    path = Path(docs_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load one PDF as one Document per page that has extractable text.

    Pages with no text (e.g. scanned images without OCR) are skipped.
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(pdf_path), "page": page_number},
            )
        )
    return docs


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Load every PDF in ``docs_dir``. Unreadable PDFs are logged and skipped."""
    docs: list[Document] = []
    for pdf in find_pdfs(docs_dir):
        try:
            docs.extend(load_pdf(pdf))
        except (PdfReadError, OSError, ValueError) as exc:
            logger.warning("Skipping unreadable PDF %s: %s", pdf, exc)
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split page documents into overlapping chunks, keeping metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
