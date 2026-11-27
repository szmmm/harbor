"""Rasterize one page of the PDF received through a browser download."""

import subprocess
from pathlib import Path


def render_pdf_page(path: Path, page: int) -> bytes:
    if type(page) is not int or page < 1:
        raise ValueError("page must be a positive integer")
    with path.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise ValueError("download is not a PDF")
    try:
        result = subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-scale-to",
                "1400",
                "-png",
                str(path.absolute()),
            ],
            capture_output=True,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("could not render that page of the downloaded PDF") from exc
    if not result.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("renderer returned no page image")
    return result.stdout
