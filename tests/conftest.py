"""Shared fixtures. Nothing here needs Ollama or network access."""

from __future__ import annotations

from pathlib import Path

import pytest


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal, valid PDF with one line of Helvetica text per page."""
    n = len(pages)
    font_id = 3 + 2 * n
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
            + f"] /Count {n} >>"
        ).encode(),
    ]
    for i, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET".encode("latin-1")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {4 + 2 * i} 0 R >>"
            ).encode()
        )
        objects.append(
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    make_pdf(
        d / "shelter.pdf",
        [
            "A lean-to shelter uses a ridgepole against a tree.",
            "A debris hut is insulated with leaves and forest litter.",
        ],
    )
    make_pdf(d / "fire.pdf", ["The fire triangle is heat, fuel, and oxygen."])
    (d / "notes.txt").write_text("not a pdf")
    return d
