import asyncio
import base64
import logging

import httpx

from harbor.agents.installed.ddb import DDB, _prune_workspace


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
                },
            )
        return httpx.Response(
            200,
            json={
                "files": [{"type": "file", "path": path} for path in [*caches, *files]]
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


def test_archive_prune_removes_next_intermediates_but_keeps_built_app(tmp_path):
    for path in [".next/cache/webpack/0.pack", "out/index.html", "public/logo.png"]:
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"content")
    _prune_workspace(tmp_path)
    assert not (tmp_path / ".next").exists()
    assert (tmp_path / "out/index.html").exists()
    assert (tmp_path / "public/logo.png").exists()
