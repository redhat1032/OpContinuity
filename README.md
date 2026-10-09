# OpContinuity

A local, offline-capable **retrieval-augmented generation (RAG)** advisor for fiction
writers who want military and survival realism. Point it at a folder of PDF field
manuals and ask questions in plain English. It finds the most relevant passages and
has a local LLM (via [Ollama](https://ollama.com)) answer **only from those passages**,
citing the source file and page.

Nothing leaves your machine: documents, embeddings, and the model all run locally.

## Architecture

```
 docs/*.pdf
     │  pypdf: one Document per page (source + page metadata)
     ▼
 ingest.py ── RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
     │
     ▼
 store.py ── OllamaEmbeddings(--embed-model) ──► Chroma at chroma_db/<embed-model>/
     │                                              (built once, reused; --rebuild re-indexes)
     ▼
 llm.py  ── retriever (top --k chunks) ─► prompt (system rules + <context>…</context>)
                                         ─► ChatOllama(--chat-model) ─► answer with citations
     ▲
 cli.py  ── argparse, Ollama health checks, interactive REPL or one-shot --question
```

| Module | Responsibility |
| --- | --- |
| `opcontinuity/ingest.py` | Find PDFs, extract text per page with `pypdf`, split into chunks |
| `opcontinuity/store.py` | Per-embedding-model Chroma path, build / load / rebuild the index |
| `opcontinuity/llm.py` | Prompt, RAG chain (LangChain LCEL), Ollama reachability and model checks |
| `opcontinuity/cli.py` | Command-line interface (`python -m opcontinuity`) |

**Why one database per embedding model?** Vectors from different embedding models have
different dimensions and meanings. Mixing them in one store either crashes or silently
returns irrelevant passages, so each model gets its own `chroma_db/<model>/` folder.

## Setup

Requirements: Python 3.11+ and [Ollama](https://ollama.com/download).

```bash
# 1. Pull the default models (an embedding model and a chat model)
ollama pull nomic-embed-text
ollama pull llama3
#    Optional lighter or alternative models:
#    ollama pull phi3:mini          # smaller chat model for low-RAM machines
#    ollama pull mxbai-embed-large  # alternative embedding model

# 2. Install Python dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Make sure Ollama is running (the desktop app, or:)
ollama serve
```

## Usage

```bash
# Interactive session using docs/ and the default models
python -m opcontinuity

# One question, then exit
python -m opcontinuity -q "What are the key differences between a lean-to and a debris hut?"

# Different models, more context per answer
python -m opcontinuity --embed-model mxbai-embed-large --chat-model phi3:mini --k 6

# Re-index after adding or removing PDFs
python -m opcontinuity --rebuild
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--docs` | `docs/` | Folder of PDFs to index |
| `--embed-model` | `nomic-embed-text` | Ollama embedding model (selects `chroma_db/<model>/`) |
| `--chat-model` | `llama3` | Ollama chat model; embedding models are rejected |
| `--k` | `4` | Chunks retrieved per question |
| `--rebuild` | off | Re-index the docs for this embedding model |
| `--db-root` | `chroma_db` | Where the per-model stores live |
| `--ollama-host` | `OLLAMA_HOST` / localhost | Ollama server URL |
| `-q, --question` | none | Ask a single question non-interactively |
| `-v, --verbose` | off | Verbose logging |

Type `exit`, `quit`, Ctrl-D or Ctrl-C to leave the interactive session.

The first run with a given embedding model indexes every PDF, which can take several
minutes for large manuals. Later runs reuse the index. **New PDFs are not picked up
automatically; run with `--rebuild`.**

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest                     # no Ollama needed: uses a generated PDF and fake embeddings
pip-audit -r requirements.txt
```

CI (`.github/workflows/ci.yml`) runs ruff, pytest, and pip-audit on every push and PR.

## Limitations

- **Text-based PDFs only.** Scanned pages with no text layer are skipped; there's no OCR.
- **Answers are only as good as retrieval.** Small chat models (e.g. `tinyllama`) often
  ignore the context or hallucinate. Always check the cited pages.
- **No incremental indexing.** Adding or removing documents requires `--rebuild`.
- **Not a safety-filtered system.** The model answers from whatever is in `docs/`. There's
  no content moderation layer, so curate the corpus deliberately.
- **Prompt injection.** Retrieved text goes into the prompt. The system prompt tells the
  model to ignore instructions found in documents, but that isn't a guarantee. Only
  index documents you trust.
- **Embedding-model check is heuristic.** Names containing `embed`, `minilm`, `bge`, and
  so on are rejected as chat models. Ollama's reported capabilities are also checked
  when available.
- **Experimental single-user CLI.** It has no web UI, authentication, or multi-user support.

## Security notes

- `chromadb` is pinned to the latest release (1.5.9), but `pip-audit` reports advisories
  with **no fixed version available**: PYSEC-2026-311 (CVE-2026-45829), PYSEC-2026-3813
  (CVE-2026-45830), PYSEC-2026-3814 (CVE-2026-45833) and PYSEC-2026-3815 (CVE-2026-45831).
  All of them affect the **Chroma HTTP server** (its REST API and RBAC authorization).
  OpContinuity uses Chroma only as an **embedded, in-process library**; it never starts
  or exposes a Chroma server. These IDs are explicitly ignored in CI as an accepted
  risk, so any new advisory still fails the build. Revisit when upstream ships a fix.
- No credentials are needed or stored. `.env`, `venv/` and `chroma_db/` are git-ignored.

## Repository layout

```
opcontinuity/      the application package (python -m opcontinuity)
tests/             pytest suite (no network / Ollama required)
experiments/       the original prototype scripts, kept for reference (see below)
docs/              source PDFs to index
```

### `experiments/`

The original prototype scripts (`query_manual*.py`, `query_manualOLD.py`,
`query_manual_text.py`, `debug_loader.py`, `debug_retriever.py`) record how the project
evolved: from one hard-coded PDF to a whole folder, through several model choices.
They are **kept for history only and aren't maintained**. Some have known bugs (for
example `query_manual2.py` uses an embedding model as the chat model, and
`query_manualOLD.py` calls a non-existent `ollama.Llama3()`), and they all share one
`chroma_db/` folder, which mixes embedding dimensions. Use `python -m opcontinuity`
instead.
