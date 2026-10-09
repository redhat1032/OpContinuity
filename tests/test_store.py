from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from opcontinuity.store import (
    db_path_for,
    get_vectorstore,
    index_exists,
    sanitize_model_name,
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("nomic-embed-text", "nomic-embed-text"),
        ("mxbai-embed-large:latest", "mxbai-embed-large_latest"),
        ("hf.co/org/model:Q4", "hf.co_org_model_Q4"),
        ("../../etc", "etc"),
    ],
)
def test_sanitize_model_name(model, expected):
    assert sanitize_model_name(model) == expected


@pytest.mark.parametrize("bad", ["", "   ", "..", "///"])
def test_sanitize_rejects_empty(bad):
    with pytest.raises(ValueError):
        sanitize_model_name(bad)


def test_db_path_is_per_model(tmp_path: Path):
    a = db_path_for("nomic-embed-text", tmp_path)
    b = db_path_for("mxbai-embed-large", tmp_path)
    assert a != b
    assert a.parent == tmp_path and b.parent == tmp_path
    assert db_path_for("../escape", tmp_path).parent == tmp_path


def test_build_then_load_then_rebuild(docs_dir: Path, tmp_path: Path):
    emb = DeterministicFakeEmbedding(size=16)
    db = db_path_for("fake-embed", tmp_path / "chroma_db")
    assert not index_exists(db)

    store = get_vectorstore(docs_dir, emb, db)
    assert index_exists(db)
    assert store._collection.count() == 3

    # Adding a PDF without --rebuild keeps the existing index.
    from tests.conftest import make_pdf

    make_pdf(docs_dir / "water.pdf", ["Boil water for one minute to purify it."])
    reloaded = get_vectorstore(docs_dir, emb, db)
    assert reloaded._collection.count() == 3

    rebuilt = get_vectorstore(docs_dir, emb, db, rebuild=True)
    assert rebuilt._collection.count() == 4


def test_separate_models_do_not_collide(docs_dir: Path, tmp_path: Path):
    root = tmp_path / "chroma_db"
    small = get_vectorstore(docs_dir, DeterministicFakeEmbedding(size=8), db_path_for("a", root))
    big = get_vectorstore(docs_dir, DeterministicFakeEmbedding(size=32), db_path_for("b", root))
    assert small.similarity_search("fire", k=1)
    assert big.similarity_search("fire", k=1)


def test_build_with_no_text_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No extractable text"):
        get_vectorstore(empty, DeterministicFakeEmbedding(size=8), tmp_path / "db" / "m")
