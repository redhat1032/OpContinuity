"""Vector store location and lifecycle.

Each embedding model gets its own Chroma directory, because vectors from
different models have different dimensions and are not comparable.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from opcontinuity.ingest import load_documents, split_documents

logger = logging.getLogger(__name__)

DEFAULT_DB_ROOT = Path("chroma_db")
_CHROMA_MARKER = "chroma.sqlite3"


def sanitize_model_name(model: str) -> str:
    """Turn a model name like ``mxbai-embed-large:latest`` into a safe directory name."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip()).strip("._")
    if not name:
        raise ValueError(f"Invalid embedding model name: {model!r}")
    return name


def db_path_for(embed_model: str, db_root: str | Path = DEFAULT_DB_ROOT) -> Path:
    """Return the Chroma directory for ``embed_model`` under ``db_root``."""
    root = Path(db_root)
    path = root / sanitize_model_name(embed_model)
    # Defence in depth: the sanitized name can never escape db_root.
    if path.resolve().parent != root.resolve():
        raise ValueError(f"Refusing DB path outside {root}: {path}")
    return path


def index_exists(db_path: str | Path) -> bool:
    """True if ``db_path`` holds a persisted Chroma index."""
    return (Path(db_path) / _CHROMA_MARKER).is_file()


def build_vectorstore(
    docs_dir: str | Path,
    embedding: Embeddings,
    db_path: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chroma:
    """(Re)index every PDF in ``docs_dir`` into a fresh store at ``db_path``."""
    db_path = Path(db_path)
    docs = load_documents(docs_dir)
    if not docs:
        raise RuntimeError(
            f"No extractable text found in PDFs under {docs_dir}. "
            "Add text-based (not scanned-only) PDFs."
        )
    kwargs = {}
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        kwargs["chunk_overlap"] = chunk_overlap
    chunks = split_documents(docs, **kwargs)
    if index_exists(db_path):
        # Drop the old collection through Chroma rather than deleting files, so
        # an open client in this process never points at a vanished database.
        Chroma(persist_directory=str(db_path), embedding_function=embedding).delete_collection()
    elif db_path.exists():
        shutil.rmtree(db_path)  # leftover dir without an index
    db_path.mkdir(parents=True, exist_ok=True)
    logger.info("Embedding %d chunks from %d pages into %s", len(chunks), len(docs), db_path)
    return Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=str(db_path)
    )


def get_vectorstore(
    docs_dir: str | Path,
    embedding: Embeddings,
    db_path: str | Path,
    rebuild: bool = False,
) -> Chroma:
    """Load the store at ``db_path``, building it if missing or if ``rebuild``."""
    if rebuild or not index_exists(db_path):
        return build_vectorstore(docs_dir, embedding, db_path)
    logger.info("Loading existing vector store from %s", db_path)
    return Chroma(persist_directory=str(db_path), embedding_function=embedding)
