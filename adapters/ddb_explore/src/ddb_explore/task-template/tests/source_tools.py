"""Read-only access to the app's source files for the judge.

Some rubric leaves are about the code, not the screen: the framework, a
self-contained HTML file, no CDN, data stored on a server. The DDB agent copies
the builder's workspace to /app/workspace (without .git, node_modules and dist),
and these tools let the judge read it. Paths cannot leave the root.
"""

from __future__ import annotations

import re
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", "dist", "build", ".next", "tool_call_logs", "__pycache__"}
MAX_FILES = 300
READ_CHARS = 6000
MAX_MATCHES = 40
MAX_FILE_BYTES = 2_000_000


class SourceTree:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        files = []
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if any(part in SKIP_DIRS for part in relative.parts) or not path.is_file():
                continue
            files.append(path)
        return files

    def _text(self, path: Path) -> str | None:
        """File text, or None for a binary or very large file."""
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
        if b"\0" in data[:4096]:
            return None
        return data.decode("utf-8", errors="replace")

    def list_files(self) -> dict:
        files = self._files()
        if not files:
            return {"success": False, "error": f"no source files under {self.root}"}
        lines = [f"{p.relative_to(self.root)} ({p.stat().st_size} bytes)" for p in files[:MAX_FILES]]
        if len(files) > MAX_FILES:
            lines.append(f"... and {len(files) - MAX_FILES} more files")
        return {"success": True, "text": "\n".join(lines)}

    def read(self, path: str, offset: int = 0) -> dict:
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root) or not target.is_file():
            return {"success": False, "error": f"no file {path!r} in the source tree"}
        text = self._text(target)
        if text is None:
            return {"success": False, "error": f"{path!r} is binary or larger than {MAX_FILE_BYTES} bytes"}
        offset = max(0, int(offset or 0))
        part = text[offset : offset + READ_CHARS]
        end = offset + len(part)
        more = f"\n[characters {offset}-{end} of {len(text)}; read again with offset {end}]" if end < len(text) else ""
        return {"success": True, "text": part + more}

    def search(self, pattern: str) -> dict:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"success": False, "error": f"invalid pattern: {exc}"}
        matches = []
        for path in self._files():
            text = self._text(path)
            if text is None:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{path.relative_to(self.root)}:{number}: {line.strip()[:200]}")
                    if len(matches) >= MAX_MATCHES:
                        return {"success": True, "text": "\n".join(matches) + "\n[more matches not shown]"}
        return {"success": True, "text": "\n".join(matches) or "no matches"}
