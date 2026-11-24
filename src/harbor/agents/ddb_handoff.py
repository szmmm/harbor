"""Completed-task evidence and reliable transfers through existing DDB APIs."""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import tarfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import zstandard

EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    ".next",
    ".cache",
    "__pycache__",
    ".venv",
    "tool_call_logs",
    "venv",
}

EXCLUDED_PATHS = {".deepdiver/skills", ".deepdiver/api_catalog"}


class WorkspaceTransferError(RuntimeError):
    """Infrastructure failed to provide a complete, stable source export."""


def safe_detail(value, secret: str = "") -> str:
    text = str(value or "")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    return re.sub(r"https?://[^\s<>\"']+", "<URL redacted>", text)[:2000]


def make_handoff(
    task: dict[str, Any], transfer: dict[str, Any], *, secret: str = ""
) -> dict[str, Any]:
    result = task.get("result") or {}
    build = next(
        (
            e.get("data") or {}
            for e in reversed(task.get("events") or [])
            if e.get("type") == "build_complete"
        ),
        {},
    )
    return {
        "schema_version": 1,
        "execution": {
            "status": task.get("status"),
            "reported_success": result.get("success"),
            "error": safe_detail(result.get("error") or task.get("error"), secret),
        },
        "actor_claim": {
            "completion_status": result.get("completion_status"),
            "final_answer": safe_detail(result.get("final_answer"), secret),
            "note": "Actor claims are not verification. A null completion_status means the headless API did not expose it.",
        },
        "build": {
            "mode": build.get("mode"),
            "skipped": build.get("skipped"),
            "runtime_state": build.get("runtime_state"),
            "success": build.get("success"),
            "error": safe_detail(build.get("error"), secret),
        },
        "preview_url_provided": bool(result.get("preview_url")),
        "recent_execution_errors": [
            {
                "type": event.get("type"),
                "detail": safe_detail(
                    (event.get("data") or {}).get("error")
                    or (event.get("data") or {}).get("message"),
                    secret,
                ),
            }
            for event in (task.get("events") or [])
            if event.get("type") == "error"
        ][-3:],
        "source_transfer": {
            **{
                key: transfer[key]
                for key in (
                    "complete",
                    "inventory_source",
                    "files_expected",
                    "files_downloaded",
                    "excluded_directories",
                )
            },
            "inventory_complete": transfer.get("inventory_complete", False),
        },
        "source_scope": {
            "inventory_complete": transfer.get("inventory_complete", False),
            "limitations": transfer.get("limitations", []),
            "unavailable_supplements": transfer.get("unavailable_supplements", []),
            "note": "Transfer checks are automatic harness checks, not judge tasks. Matching local hashes prove upload integrity, not that DDB exposed every actor file.",
        },
        "excluded_platform_paths": sorted(EXCLUDED_PATHS),
        "delivery_notes": [
            "Inspect .deepdiver/artifact.json, package.json and actual source to understand the entrypoint; actor claims alone are not proof.",
            "Static sites can declare a named HTML entry. React/Vite can use custom roots; Next.js does not require root index.html.",
            "Installed dependencies and caches are intentionally excluded. Package manifests, lockfiles, vendored dependencies, assets and build outputs are retained when returned by the existing API; consult source_scope for upstream omissions. Platform-provided skill/catalog links are excluded; their absence is not an actor omission.",
        ],
    }


def excluded_path(path: PurePosixPath) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts) or any(
        path.as_posix() == excluded or path.as_posix().startswith(excluded + "/")
        for excluded in EXCLUDED_PATHS
    )


SOURCE_LIMITATIONS = [
    "Existing DDB listings can hide directories or stop at a depth limit. Missing local files are not proof the actor omitted them.",
    "Saved snapshots can omit dist and preview runtime logs and may be from an earlier revision. API source labels do not pin a revision.",
    "Local hashes verify transfer to the judge, not freshness or equality with the remote filesystem.",
]


def workspace_receipt(root: Path, source: str, *, unavailable=None) -> dict[str, Any]:
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    return {
        "schema_version": 1,
        "complete": True,
        "inventory_complete": False,
        "inventory_source": source,
        "files_expected": len(hashes),
        "files_downloaded": len(hashes),
        "excluded_directories": sorted(EXCLUDED_DIRS),
        "excluded_paths": sorted(EXCLUDED_PATHS),
        "sha256": hashes,
        "limitations": SOURCE_LIMITATIONS,
        "unavailable_supplements": unavailable or [],
    }


def extract_workspace_archive(archive: bytes, root: Path) -> dict[str, Any]:
    """Retain original paths and all app files from an existing DDB snapshot."""
    reader = (
        zstandard.ZstdDecompressor().stream_reader(BytesIO(archive))
        if archive.startswith(b"\x28\xb5\x2f\xfd")
        else BytesIO(archive)
    )
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    with reader, tarfile.open(fileobj=reader, mode="r|*") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise WorkspaceTransferError("Unsafe archive member path")
            if excluded_path(path) or member.isdir():
                continue
            if not member.isfile() or not path.parts or path.as_posix() in seen:
                raise WorkspaceTransferError("Unsupported or duplicate archive member")
            seen.add(path.as_posix())
            stream = tar.extractfile(member)
            if stream is None:
                raise WorkspaceTransferError("Unreadable archive member")
            content = stream.read()
            if len(content) != member.size:
                raise WorkspaceTransferError("Truncated archive member")
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    return workspace_receipt(root, "snapshot_archive")


def _inventory(payload: dict[str, Any], max_depth: int) -> tuple[list[str], str]:
    source = payload.get("source")
    if (
        not isinstance(source, str)
        or source not in {"snapshot", "live", "empty", "deferred"}
        or payload.get("degraded")
        or payload.get("truncated")
    ):
        raise WorkspaceTransferError(
            "Source inventory is unavailable, stale or truncated"
        )
    nodes = payload.get("files")
    if not isinstance(nodes, list):
        raise WorkspaceTransferError("Invalid source inventory")
    files = []

    def walk(items):
        for item in items:
            raw = item.get("path", "")
            path = PurePosixPath(raw)
            if not raw or path.is_absolute() or ".." in path.parts or "\\" in raw:
                raise WorkspaceTransferError("Unsafe source inventory path")
            if excluded_path(path):
                continue
            if len(path.parts) >= max_depth or item.get("truncated"):
                raise WorkspaceTransferError("Source listing reached its depth limit")
            if item.get("type") == "file":
                files.append(path.as_posix())
            elif item.get("type") not in {"directory", "dir", "folder"}:
                raise WorkspaceTransferError("Unsupported source inventory node")
            walk(item.get("children") or [])

    walk(nodes)
    if len(set(files)) != len(files):
        raise WorkspaceTransferError("Duplicate source inventory paths")
    return sorted(files), source


async def download_complete_workspace(
    client,
    files_url: str,
    root: Path,
    read_file,
    *,
    max_depth=32,
    concurrency=8,
    attempts=3,
    retry_delay=1.0,
) -> dict[str, Any]:
    """Finish all advertised downloads; explicitly qualify upstream completeness."""

    async def listing(path: str = "") -> tuple[list[str], str]:
        response = await client.get(
            files_url,
            params={"path": path, "max_depth": max_depth, "mode": "edit"},
            timeout=30.0,
        )
        response.raise_for_status()
        return _inventory(response.json(), max_depth)

    last_error = "Source inventory unavailable"
    for attempt in range(attempts):
        try:
            files, source = await listing()
            break
        except (httpx.HTTPError, ValueError, WorkspaceTransferError) as exc:
            last_error = type(exc).__name__
            if attempt + 1 < attempts:
                await asyncio.sleep(retry_delay)
    else:
        raise WorkspaceTransferError(f"Cannot obtain source inventory: {last_error}")

    targets = set(files)
    unavailable = []
    # The existing API accepts subdirectory listings, even for roots hidden
    # in its normal UI tree. Paths in those responses are relative to the root.
    supplements = {}
    if source in {"live", "snapshot"}:

        async def supplement(prefix):
            try:
                extra, extra_source = await listing(prefix)
                if extra_source != source:
                    raise WorkspaceTransferError("Supplement source changed")
                supplements[prefix] = extra
            except (httpx.HTTPError, ValueError, WorkspaceTransferError):
                unavailable.append(prefix + "/")

        async with asyncio.TaskGroup() as group:
            for prefix in (".deepdiver", "dist", "build", "out"):
                group.create_task(supplement(prefix))
        for prefix, extra in supplements.items():
            targets.update(
                (PurePosixPath(prefix) / path).as_posix()
                for path in extra
                if not excluded_path(PurePosixPath(prefix) / path)
            )

    root.mkdir(parents=True, exist_ok=True)
    pending = set(targets)
    semaphore = asyncio.Semaphore(concurrency)

    async def download_file(path: str) -> bool:
        try:
            content = await read_file(client, files_url, path, source=source)
        except (httpx.HTTPError, ValueError, TypeError, OSError):
            return False
        if content is None:
            return False
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return True

    async def fetch(path: str) -> None:
        async with semaphore:
            if await download_file(path):
                pending.remove(path)

    for attempt in range(attempts):
        async with asyncio.TaskGroup() as group:
            for path in sorted(pending):
                group.create_task(fetch(path))
        if not pending:
            break
        if attempt + 1 < attempts:
            await asyncio.sleep(retry_delay * (attempt + 1))
    if pending:
        raise WorkspaceTransferError(
            f"Source download incomplete: {len(pending)}/{len(targets)} advertised files unavailable"
        )
    # A hidden artifact declaration is especially useful for judging delivery.
    # Failure of this unadvertised probe is diagnostic, never proof of absence.
    artifact = ".deepdiver/artifact.json"
    if source in {"live", "snapshot"} and artifact not in targets:
        for attempt in range(attempts):
            if await download_file(artifact):
                break
            if attempt + 1 < attempts:
                await asyncio.sleep(retry_delay)
        else:
            unavailable.append(artifact)

    async def recheck_inventory(path: str = "") -> tuple[list[str], str]:
        for attempt in range(1, attempts + 1):
            try:
                return await listing(path)
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                    if status not in (408, 429) and not 500 <= status < 600:
                        raise
                if attempt == attempts:
                    raise
                # Jitter keeps concurrent trials from retrying in lockstep.
                await asyncio.sleep(retry_delay * (attempt + random.random()))
        raise ValueError("Inventory recheck requires at least one attempt")

    final_files, final_source = await recheck_inventory()
    if final_files != files or final_source != source:
        raise WorkspaceTransferError("Source inventory changed during download")
    for prefix, extra in supplements.items():
        final_extra, final_source = await recheck_inventory(prefix)
        if final_extra != extra or final_source != source:
            raise WorkspaceTransferError("Supplement inventory changed during download")
    return workspace_receipt(root, source, unavailable=sorted(unavailable))
