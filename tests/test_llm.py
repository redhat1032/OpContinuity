from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from opcontinuity.llm import (
    PROMPT,
    ModelConfigError,
    OllamaUnavailableError,
    build_chain,
    check_chat_capability,
    check_ollama,
    format_docs,
    looks_like_embedding_model,
    validate_chat_model_name,
)
from opcontinuity.store import db_path_for, get_vectorstore


def test_prompt_is_well_formed():
    msgs = PROMPT.format_messages(context="CTX", question="Q?")
    human = msgs[-1].content
    assert "<context>\nCTX\n</context>" in human
    assert human.count("<context>") == human.count("</context>") == 1


@pytest.mark.parametrize("name", ["nomic-embed-text", "mxbai-embed-large", "all-minilm"])
def test_embedding_models_rejected_as_chat(name):
    assert looks_like_embedding_model(name)
    with pytest.raises(ModelConfigError):
        validate_chat_model_name(name, "nomic-embed-text")


@pytest.mark.parametrize("name", ["llama3", "phi3:mini", "mistral", "tinyllama"])
def test_chat_models_accepted(name):
    assert not looks_like_embedding_model(name)
    validate_chat_model_name(name, "nomic-embed-text")


def test_same_model_for_both_rejected():
    with pytest.raises(ModelConfigError):
        validate_chat_model_name("llama3", "llama3")


class _DownClient:
    def list(self):
        raise ConnectionError("connection refused")


class _UpClient:
    def __init__(self, names, caps=None):
        self._names = names
        self._caps = caps or []

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=n) for n in self._names])

    def show(self, _model):
        return SimpleNamespace(capabilities=self._caps)


def test_check_ollama_unreachable_message():
    with pytest.raises(OllamaUnavailableError, match="ollama serve"):
        check_ollama(None, ["llama3"], client=_DownClient())


def test_check_ollama_missing_model_suggests_pull():
    client = _UpClient(["llama3:latest"])
    with pytest.raises(ModelConfigError, match="ollama pull nomic-embed-text"):
        check_ollama(None, ["nomic-embed-text", "llama3"], client=client)


def test_check_ollama_ok():
    check_ollama(
        None,
        ["nomic-embed-text", "llama3"],
        client=_UpClient(["nomic-embed-text:latest", "llama3:latest"]),
    )


def test_check_chat_capability():
    with pytest.raises(ModelConfigError):
        check_chat_capability("x", _UpClient([], caps=["embedding"]))
    check_chat_capability("x", _UpClient([], caps=["completion"]))
    check_chat_capability("x", _UpClient([], caps=[]))  # unknown -> allow


def test_format_docs_cites_source_and_page():
    out = format_docs([Document(page_content="text", metadata={"source": "a.pdf", "page": 0})])
    assert out.startswith("[a.pdf p.1]")


def test_chain_end_to_end_with_fakes(docs_dir, tmp_path):
    store = get_vectorstore(
        docs_dir, DeterministicFakeEmbedding(size=16), db_path_for("fake", tmp_path)
    )
    llm = FakeListChatModel(responses=["Heat, fuel, and oxygen."])
    chain = build_chain(store.as_retriever(search_kwargs={"k": 2}), llm)
    assert chain.invoke("What is the fire triangle?") == "Heat, fuel, and oxygen."
