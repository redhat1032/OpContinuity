"""Command-line interface: ``python -m opcontinuity``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from opcontinuity import __version__
from opcontinuity.llm import (
    ModelConfigError,
    OllamaUnavailableError,
    build_chain,
    check_chat_capability,
    check_ollama,
    validate_chat_model_name,
)
from opcontinuity.store import DEFAULT_DB_ROOT, db_path_for, get_vectorstore

EXIT_WORDS = {"exit", "quit", ":q"}


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="opcontinuity",
        description="Ask questions about the PDF field manuals in a docs folder, "
        "answered by a local Ollama model (RAG).",
    )
    p.add_argument("--docs", default="docs/", help="folder of PDFs to index (default: docs/)")
    p.add_argument(
        "--embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model (default: nomic-embed-text)",
    )
    p.add_argument("--chat-model", default="llama3", help="Ollama chat model (default: llama3)")
    p.add_argument(
        "--k", type=positive_int, default=4, help="chunks to retrieve per question (default: 4)"
    )
    p.add_argument(
        "--rebuild", action="store_true", help="re-index the docs for this embedding model"
    )
    p.add_argument(
        "--db-root",
        default=str(DEFAULT_DB_ROOT),
        help="root folder for per-model vector stores (default: chroma_db)",
    )
    p.add_argument(
        "--ollama-host", default=None, help="Ollama URL (default: OLLAMA_HOST or localhost)"
    )
    p.add_argument("-q", "--question", help="ask one question and exit (non-interactive)")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def repl(ask: Callable[[str], str], input_fn: Callable[[str], str] = input) -> int:
    """Interactive loop. Exits cleanly on quit/exit, Ctrl-C, or Ctrl-D."""
    print("Ready. Ask a question, or type 'exit' to quit.")
    while True:
        try:
            question = input_fn("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("Goodbye.")
            return 0
        try:
            answer = ask(question)
        except KeyboardInterrupt:
            print("\n(cancelled)")
            continue
        except Exception as exc:  # keep the session alive on transient errors
            print(f"\nError while answering: {exc}", file=sys.stderr)
            continue
        print(f"\n--- Answer ---\n{answer}\n--------------")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        # pypdf logs noisy font/encoding warnings on many military PDFs.
        logging.getLogger("pypdf").setLevel(logging.ERROR)

    try:
        validate_chat_model_name(args.chat_model, args.embed_model)

        import ollama
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        client = ollama.Client(host=args.ollama_host) if args.ollama_host else ollama.Client()
        check_ollama(args.ollama_host, [args.embed_model, args.chat_model], client=client)
        check_chat_capability(args.chat_model, client)

        llm_kwargs = {"base_url": args.ollama_host} if args.ollama_host else {}
        embedding = OllamaEmbeddings(model=args.embed_model, **llm_kwargs)
        db_path = db_path_for(args.embed_model, args.db_root)
        print(f"Using vector store {db_path} (embed model: {args.embed_model})")
        if args.rebuild:
            print("Rebuilding index; this can take a while for large PDFs...")
        vectorstore = get_vectorstore(args.docs, embedding, db_path, rebuild=args.rebuild)
        retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
        chain = build_chain(retriever, ChatOllama(model=args.chat_model, **llm_kwargs))
    except (OllamaUnavailableError, ModelConfigError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if args.question:
        try:
            print(chain.invoke(args.question))
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"Error while answering: {exc}", file=sys.stderr)
            return 1
        return 0
    return repl(chain.invoke)


if __name__ == "__main__":
    sys.exit(main())
