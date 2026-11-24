"""Serve the exported DDB build without running workspace scripts."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

PREVIEW_URL_PATH = Path("/tmp/harbor_preview_url")


class LocalDistError(RuntimeError):
    """The downloaded build cannot be served."""


class InvalidDistError(LocalDistError):
    """The export contains no usable dist entry point."""


class EntryAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("src"):
            self.refs.append(attrs["src"])
        if (
            tag == "link"
            and attrs.get("href")
            and {"stylesheet", "modulepreload"}.intersection(
                (attrs.get("rel") or "").lower().split()
            )
        ):
            self.refs.append(attrs["href"])


def public_path(root: Path, path: Path) -> bool:
    return path.is_relative_to(root) and all(
        not part.startswith(".") for part in path.relative_to(root).parts
    )


class DistHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        root = Path(self.directory)
        path = Path(self.translate_path(self.path)).resolve()
        if path.is_dir():
            path = (path / "index.html").resolve()
        if not public_path(root, path) or not path.is_file():
            self.send_error(404)
            return None
        return super().send_head()

    def log_message(self, format, *args):
        # Request paths can contain app data.
        pass


def validate_dist(root: Path) -> None:
    index = root / "index.html"
    if not public_path(root, index.resolve()) or not index.is_file():
        raise InvalidDistError("dist/index.html is missing")
    html = index.read_text(encoding="utf-8")
    if not html.strip():
        raise InvalidDistError("dist/index.html is empty")
    if "__deepdiver_ready" in html or "Finalizing your app" in html:
        raise InvalidDistError("dist/index.html is a publishing placeholder")

    assets = EntryAssets()
    assets.feed(html)
    for ref in assets.refs:
        url = urlsplit(ref)
        if url.scheme or url.netloc:
            continue
        asset = (root / unquote(url.path).lstrip("/")).resolve()
        if (
            not public_path(root, asset)
            or not asset.is_file()
            or not asset.stat().st_size
        ):
            raise InvalidDistError(
                "dist/index.html references a missing or empty local asset"
            )


@contextmanager
def serve_dist(dist: Path):
    try:
        if dist.is_symlink():
            raise InvalidDistError("dist is a symlink")
        root = dist.resolve()
        validate_dist(root)
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(DistHandler, directory=str(root))
        )
    except InvalidDistError:
        raise
    except UnicodeError as exc:
        raise InvalidDistError("dist/index.html is not UTF-8") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise LocalDistError(
            f"Could not read or serve dist ({type(exc).__name__})"
        ) from exc

    with server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            raise LocalDistError("Could not start the static server thread") from exc
        try:
            yield f"http://127.0.0.1:{server.server_port}/"
        finally:
            server.shutdown()
            thread.join()


def remote_preview_url() -> str:
    override = os.environ.get("APP_BASE_URL", "").strip()
    if override:
        return override
    try:
        return PREVIEW_URL_PATH.read_text().strip()
    except (OSError, ValueError):
        return ""
