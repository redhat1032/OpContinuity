import pytest

from opcontinuity import cli


def _inputs(*items):
    it = iter(items)

    def fake_input(_prompt):
        value = next(it)
        if isinstance(value, BaseException):
            raise value
        return value

    return fake_input


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.docs == "docs/"
    assert args.embed_model == "nomic-embed-text"
    assert args.chat_model == "llama3"
    assert args.k == 4
    assert args.rebuild is False


def test_parser_rejects_bad_k():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--k", "0"])


@pytest.mark.parametrize("ending", ["quit", "exit", "EXIT", EOFError(), KeyboardInterrupt()])
def test_repl_exits_cleanly(ending, capsys):
    asked = []
    rc = cli.repl(lambda q: asked.append(q) or "ans", _inputs("", "hello", ending))
    assert rc == 0
    assert asked == ["hello"]
    assert "ans" in capsys.readouterr().out


def test_repl_survives_errors(capsys):
    def boom(_q):
        raise RuntimeError("model crashed")

    assert cli.repl(boom, _inputs("q", "quit")) == 0
    assert "model crashed" in capsys.readouterr().err


def test_main_rejects_embedding_chat_model(capsys):
    rc = cli.main(["--chat-model", "mxbai-embed-large"])
    assert rc == 2
    assert "embedding model" in capsys.readouterr().err


def test_main_reports_unreachable_ollama(capsys):
    rc = cli.main(["--ollama-host", "http://127.0.0.1:9"])
    assert rc == 2
    assert "Could not reach the Ollama server" in capsys.readouterr().err
