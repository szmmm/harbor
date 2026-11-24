"""Completion admission and trustworthy local source evidence for DDB judging."""

import asyncio
import base64
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from harbor.agents.ddb_handoff import (
    WorkspaceTransferError,
    download_complete_workspace,
    extract_workspace_archive,
    make_handoff,
)
from harbor.agents.installed.ddb import DDB

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "adapters/ddb_explore/src/ddb_explore/task-template/tests"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TEMPLATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt(files):
    return {
        "schema_version": 1,
        "complete": True,
        "inventory_source": "snapshot",
        "files_expected": len(files),
        "files_downloaded": len(files),
        "excluded_directories": ["node_modules"],
        "sha256": {p: hashlib.sha256(b).hexdigest() for p, b in files.items()},
    }


@pytest.mark.parametrize(
    ("status", "success"), [("completed", False), ("failed", True)]
)
def test_admission_uses_execution_completion(tmp_path, monkeypatch, status, success):
    from harbor.agents.installed import ddb

    agent = object.__new__(DDB)
    agent.logger = logging.getLogger(__name__)
    for key, value in {
        "logs_dir": tmp_path,
        "_callback_bind_host": "localhost",
        "_callback_bind_port": 0,
        "_callback_url": None,
        "_callback_secret": "test",
        "_base_url": "https://ddb.test",
        "_api_key": "test",
        "_log_preview_state": False,
        "training": False,
        "_publish_preview_url": False,
        "_preview_base_url": "https://ddb.test",
        "_workspace_transfer": receipt({}),
    }.items():
        setattr(agent, key, value)
    listener = SimpleNamespace(advertised_url="https://callback.test", secret="test")
    monkeypatch.setattr(ddb, "_acquire_listener", AsyncMock(return_value=listener))
    monkeypatch.setattr(ddb, "_release_listener", AsyncMock())
    agent.submit_task = AsyncMock(return_value={"task_id": "task", "app_id": "app"})
    agent.wait_for_webhook = AsyncMock(
        return_value={
            "status": status,
            "result": {"success": success},
            "events": [
                {
                    "type": "build_complete",
                    "data": {"success": False, "error": "Syntax error"},
                }
            ],
        }
    )
    agent._capture_failure = AsyncMock(return_value={"error": "Task failed"})
    agent.push_data_to_sandbox = AsyncMock()
    agent._abandon = AsyncMock()
    uploaded = []

    async def upload(path, target):
        uploaded.append((target, json.loads(path.read_text())))

    env = SimpleNamespace(upload_file=upload)
    context = SimpleNamespace(metadata={})
    if status == "failed":
        with pytest.raises(RuntimeError, match="Task failed"):
            asyncio.run(agent.run("Build an app", env, context))
        agent.push_data_to_sandbox.assert_not_awaited()
        agent._abandon.assert_awaited_once()
        assert len(uploaded) == 1
        assert uploaded[0][1]["source_transfer"]["complete"] is False
        return

    asyncio.run(agent.run("Build an app", env, context))
    agent.push_data_to_sandbox.assert_awaited_once()
    agent._abandon.assert_not_awaited()
    assert uploaded[0][1]["source_transfer"]["complete"] is False
    assert uploaded[-1][1]["build"]["success"] is False
    assert uploaded[-1][1]["source_transfer"]["complete"] is True


def test_source_retry_keeps_binary_assets_and_skips_nested_dependencies(tmp_path):
    files = {
        "package.json": b"{}",
        "sub/app.tsx": b"app",
        "public/logo.png": b"\x89PNG",
        "vendor/sdk.tgz": b"vendored",
        "dist/index.html": b"built",
    }
    excluded = [
        "node_modules/lib/index.js",
        "sub/node_modules/lib/a.js",
        "sub/.next/cache/x",
    ]
    payload = {
        "source": "snapshot",
        "files": [{"type": "file", "path": p} for p in [*files, *excluded]],
    }
    reads = []

    async def read(client, url, path, *, source=None):
        reads.append(path)
        if path == "sub/app.tsx" and reads.count(path) == 1:
            return None
        return files.get(path)

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    json={"source": payload["source"], "files": []}
                    if r.url.params.get("path")
                    else payload,
                )
            ),
        ) as client:
            return await download_complete_workspace(
                client, "/files", tmp_path, read, retry_delay=0
            )

    result = asyncio.run(run())
    assert result["complete"] and result["files_downloaded"] == len(files)
    assert reads.count("sub/app.tsx") == 2
    assert not set(excluded) & set(reads)
    assert result["inventory_complete"] is False
    for path, body in files.items():
        assert (tmp_path / path).read_bytes() == body


def test_unreliable_inventory_never_counts_as_empty_work(tmp_path):
    source = "live_stale"

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    json={
                        "source": source,
                        "files": [],
                        "complete": True,
                    },
                )
            ),
        ) as client:
            return await download_complete_workspace(
                client, "/files", tmp_path, AsyncMock(), retry_delay=0
            )

    with pytest.raises(WorkspaceTransferError):
        asyncio.run(run())


def test_missing_file_after_retries_is_not_a_successful_partial_transfer(tmp_path):
    def respond(request):
        return httpx.Response(
            200,
            json={
                "source": "snapshot",
                "files": []
                if request.url.params.get("path")
                else [{"type": "file", "path": "index.html"}],
            },
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test", transport=httpx.MockTransport(respond)
        ) as client:
            return await download_complete_workspace(
                client, "/files", tmp_path, AsyncMock(return_value=None), retry_delay=0
            )

    with pytest.raises(WorkspaceTransferError, match="incomplete"):
        asyncio.run(run())


def test_changed_inventory_is_not_complete(tmp_path):
    roots = iter([[], [{"type": "file", "path": "late.html"}]])

    def respond(request):
        files = [] if request.url.params.get("path") else next(roots)
        return httpx.Response(200, json={"source": "snapshot", "files": files})

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test", transport=httpx.MockTransport(respond)
        ) as client:
            return await download_complete_workspace(
                client, "/files", tmp_path, AsyncMock(return_value=None), retry_delay=0
            )

    with pytest.raises(WorkspaceTransferError, match="changed"):
        asyncio.run(run())


def test_handoff_contains_separate_outcomes_without_secrets():
    result = make_handoff(
        {
            "status": "completed",
            "result": {
                "success": False,
                "final_answer": "Claim",
                "error": "Bearer SECRET https://private/?token=x",
            },
            "events": [
                {
                    "type": "build_complete",
                    "data": {"success": False, "error": "Syntax error"},
                }
            ],
        },
        receipt({}),
        secret="SECRET",
    )
    assert result["execution"]["status"] == "completed"
    assert result["build"]["error"] == "Syntax error"
    assert result["actor_claim"]["completion_status"] is None
    assert "SECRET" not in json.dumps(result) and "token=x" not in json.dumps(result)
    deferred = make_handoff(
        {
            "status": "completed",
            "events": [
                {
                    "type": "build_complete",
                    "data": {
                        "mode": "none",
                        "success": True,
                        "skipped": True,
                        "runtime_state": "deferred",
                    },
                }
            ],
        },
        receipt({}),
    )
    assert deferred["build"]["skipped"] is True
    assert deferred["build"]["runtime_state"] == "deferred"


def test_sandbox_hashes_checked_before_judging(tmp_path):
    files = {"index.html": b"hello"}
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "index.html").write_bytes(files["index.html"])
    transfer = receipt(files)
    rp, hp = tmp_path / "receipt.json", tmp_path / "handoff.json"
    rp.write_text(json.dumps(transfer))
    hp.write_text(json.dumps(make_handoff({"status": "completed"}, transfer)))
    loader = load("handoff").load_handoff
    assert loader(root, hp, rp)["source_transfer"]["sandbox_hashes_verified"]
    (root / "index.html").write_text("truncated")
    with pytest.raises(ValueError, match="does not match"):
        loader(root, hp, rp)
    hp = tmp_path / "handoff.json"
    hp.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "execution": {"status": "completed"},
                "source_transfer": {"complete": False},
            }
        )
    )
    with pytest.raises(ValueError, match="has not completed"):
        load("handoff").load_handoff(tmp_path, hp, tmp_path / "missing-receipt")


def test_existing_api_supplements_hidden_product_directories(tmp_path):
    roots = {
        "": [{"type": "file", "path": "src/main.tsx"}],
        ".deepdiver": [{"type": "file", "path": "artifact.json"}],
        "dist": [
            {"type": "file", "path": "index.html"},
            {"type": "file", "path": "assets/main.js"},
        ],
        "build": [],
        "out": [],
    }
    files = {
        p: p.encode()
        for p in [
            "src/main.tsx",
            ".deepdiver/artifact.json",
            "dist/index.html",
            "dist/assets/main.js",
        ]
    }
    requests = []
    reads = []

    def respond(request):
        requests.append(request)
        assert not request.url.path.endswith("/verification-tree")
        prefix = request.url.params["path"]
        if prefix in ("", "dist") and sum(
            r.url.params["path"] == prefix for r in requests
        ) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"source": "live", "files": roots[prefix]})

    async def read(client, url, path, *, source=None):
        reads.append(path)
        assert source == "live"
        if path == "dist/assets/main.js" and reads.count(path) == 1:
            return None
        return files[path]

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test", transport=httpx.MockTransport(respond)
        ) as client:
            return await download_complete_workspace(
                client, "/files", tmp_path, read, retry_delay=0
            )

    result = asyncio.run(run())
    assert result["complete"] is True and result["inventory_complete"] is False
    assert result["files_downloaded"] == 4
    assert reads.count("dist/assets/main.js") == 2
    for prefix in ("", "dist"):
        assert sum(r.url.params["path"] == prefix for r in requests) == 3
    for path, content in files.items():
        assert (tmp_path / path).read_bytes() == content


def test_existing_empty_exports_finish_with_visibility_limitations(tmp_path):
    source = "snapshot"

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test",
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"source": source, "files": []})
            ),
        ) as client:
            return await download_complete_workspace(
                client, "/files", tmp_path, AsyncMock(return_value=None), retry_delay=0
            )

    result = asyncio.run(run())
    assert result["complete"] and result["files_downloaded"] == 0
    assert result["inventory_complete"] is False and result["limitations"]


def test_failed_supplement_is_reported_without_discarding_downloaded_source(tmp_path):
    def respond(request):
        if request.url.params.get("path"):
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"source": "live", "files": [{"type": "file", "path": "index.html"}]},
        )

    async def read(client, url, path, **kwargs):
        return b"source" if path == "index.html" else None

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test", transport=httpx.MockTransport(respond)
        ) as client:
            return await download_complete_workspace(
                client, "/files", tmp_path, read, retry_delay=0
            )

    result = asyncio.run(run())
    assert result["complete"] and result["files_downloaded"] == 1
    assert "dist/" in result["unavailable_supplements"]
    assert ".deepdiver/artifact.json" in result["unavailable_supplements"]


@pytest.mark.parametrize("compressed", [True, False])
def test_archive_keeps_product_paths_and_excludes_dependencies(tmp_path, compressed):
    import io
    import tarfile

    import zstandard

    files = {
        "site/dist/index.html": b"product",
        "site/.deepdiver/artifact.json": b"{}",
        "site/build/app.js": b"script",
        "site/node_modules/react/index.js": b"dependency",
        "site/.next/cache/0.pack": b"intermediate",
    }
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for path, content in files.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    blob = data.getvalue()
    if compressed:
        blob = zstandard.ZstdCompressor().compress(blob)
    result = extract_workspace_archive(blob, tmp_path)
    assert result["files_downloaded"] == 3
    assert result["inventory_complete"] is False
    assert (tmp_path / "site/dist/index.html").read_bytes() == b"product"
    assert not (tmp_path / "dist").exists()  # No speculative wrapper stripping.
    assert not (tmp_path / "site/node_modules").exists()
    assert not (tmp_path / "site/.next").exists()


def test_existing_live_read_does_not_accept_snapshot_substitution(tmp_path):
    agent = object.__new__(DDB)
    agent.logger = logging.getLogger(__name__)

    def respond(request):
        assert request.url.params["mode"] == "edit"
        assert "verification_source" not in request.url.params
        return httpx.Response(200, json={"source": "snapshot", "content": "old"})

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test", transport=httpx.MockTransport(respond)
        ) as client:
            return await agent._read_workspace_file(
                client, "/files", "index.html", source="live"
            )

    assert asyncio.run(run()) is None


def test_truncated_archive_fails_and_removes_partial_extraction(tmp_path):
    import io
    import tarfile

    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for name in ["first.txt", "last.txt"]:
            member = tarfile.TarInfo(name)
            member.size = 4096
            archive.addfile(member, io.BytesIO(b"x" * 4096))
    blob = data.getvalue()[:6000]
    agent = object.__new__(DDB)
    agent.logger = logging.getLogger(__name__)

    async def run():
        async with httpx.AsyncClient(
            base_url="https://test",
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, content=blob)
            ),
        ) as client:
            return await agent._pull_workspace_archive(client, "/files", tmp_path)

    assert asyncio.run(run()) is False
    assert list(tmp_path.iterdir()) == []


def test_failure_extracts_event_error_and_captures_proxy_progress(tmp_path):
    agent = object.__new__(DDB)
    agent.logs_dir = tmp_path
    agent._api_key = "private-key"
    agent.proxy_url = "http://proxy"
    agent._get_session = AsyncMock(
        return_value={
            "turns": [
                {
                    "agent_id": "ddt_agent1",
                    "completion_token_ids": [1, 2],
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    task = {
        "task_id": "task",
        "status": "failed",
        "result": None,
        "events": [
            {
                "type": "error",
                "data": {
                    "iteration": 11,
                    "message": "provider failed",
                    "retryable": True,
                },
            },
            {
                "type": "complete",
                "data": {"error": "Provider failed Bearer private-key"},
            },
        ],
    }
    failure = asyncio.run(agent._capture_failure(task, "session"))
    assert failure["proxy_progress"]["completed_turns"] == 1
    assert failure["proxy_progress"]["completion_tokens"] == 2
    saved = (tmp_path / "failure.json").read_text()
    assert "private-key" not in saved and "Provider failed" in failure["error"]
    assert json.loads(saved)["events"][0]["retryable"] is True


def test_compaction_tokens_are_excluded_from_training():
    agent = object.__new__(DDB)
    # No policy fields are needed: an auxiliary stream must be skipped before
    # token extraction or trajectory scoring, even if it is the only record.
    assert all(
        not value
        for value in agent._parse_record({"turns": [{"agent_id": "compaction123"}]})
    )


def test_file_pull_skips_next_build_files_before_reading_and_keeps_assets(tmp_path):
    files = {
        "package.json": b'{"scripts":{"build":"next build"}}',
        "app/page.tsx": b"export default function Page() { return null; }",
        "public/logo.png": b"\x89PNG\r\n\x1a\n",
        "vendor/sdk.tgz": b"\x1f\x8bvendored dependency",
        "out/index.html": b"<html>built app</html>",
        ".deepdiver/artifact.json": b'{"artifacts":[]}',
    }
    caches = [".next/cache/webpack/0.pack", ".next/server/app/page.js"]
    reads = []

    def respond(request):
        if request.url.path.endswith("/read"):
            path = request.url.params["path"]
            reads.append(path)
            assert path not in caches
            return httpx.Response(
                200,
                json={
                    "content": base64.b64encode(files[path]).decode(),
                    "encoding": "base64",
                    "source": "snapshot",
                },
            )
        if request.url.params.get("path"):
            return httpx.Response(200, json={"source": "snapshot", "files": []})
        return httpx.Response(
            200,
            json={
                "source": "snapshot",
                "files": [{"type": "file", "path": path} for path in [*caches, *files]],
            },
        )

    agent = object.__new__(DDB)
    agent.logger = logging.getLogger(__name__)

    async def pull():
        async with httpx.AsyncClient(
            base_url="https://ddb.test", transport=httpx.MockTransport(respond)
        ) as client:
            return await agent._pull_workspace_files(client, "/api/files/app", tmp_path)

    assert asyncio.run(pull())
    assert set(reads) == set(files)
    for path, content in files.items():
        assert (tmp_path / path).read_bytes() == content
    assert not (tmp_path / ".next").exists()
