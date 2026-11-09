import asyncio
import hashlib
import json
from unittest.mock import AsyncMock

import pytest

from harbor.environments.docker.image_cache import ImageCache, ImageCacheError


def cache(tmp_path, monkeypatch):
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"archive")
    manifest = {
        "version": 1,
        "images": {
            "hb__test": {
                "image_id": "sha256:test",
                "archive": "image.tar",
                "sha256": hashlib.sha256(b"archive").hexdigest(),
                "size_bytes": 100,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    c = ImageCache(tmp_path, tmp_path, min_free_bytes=100, runtime_reserve_bytes=100)
    state = {"present": False, "loads": 0, "containers": "", "free": 1000}

    async def docker(*args):
        if args[0] == "info":
            return "test-daemon"
        if args[:2] == ("image", "ls"):
            return "sha256:test" if state["present"] else ""
        if args[:2] == ("image", "inspect"):
            return json.dumps(
                [
                    {
                        "Id": "sha256:test",
                        "RepoTags": ["hb__test:latest"],
                        "RepoDigests": [],
                    }
                ]
            )
        if args[:2] == ("image", "load"):
            state["present"] = True
            state["loads"] += 1
            return ""
        if args[:2] == ("image", "rm"):
            state["present"] = False
            return ""
        if args[0] == "ps":
            return state["containers"]
        raise AssertionError(args)

    monkeypatch.setattr(c, "_docker", docker)
    monkeypatch.setattr(c, "_free_bytes", lambda: state["free"])
    return c, state


@pytest.mark.asyncio
async def test_bundle_loads_once_and_keeps_image_until_all_work_finishes(
    tmp_path, monkeypatch
):
    c, state = cache(tmp_path, monkeypatch)
    async with c.session():
        async with c.lease_bundle(["hb__test", "hb__test"]):
            await c.require_image("hb__test")
            await c.require_image("hb__test")
            assert state["present"] and state["loads"] == 1
        assert not state["present"]


@pytest.mark.asyncio
async def test_insufficient_space_does_not_load_or_start_work(tmp_path, monkeypatch):
    c, state = cache(tmp_path, monkeypatch)
    state["free"] = 299
    async with c.session():
        with pytest.raises(ImageCacheError, match="space"):
            async with c.lease_bundle(["hb__test"]):
                pytest.fail("work admitted without enough disk space")
    assert state["loads"] == 0


@pytest.mark.asyncio
async def test_corrupt_archive_is_rejected_before_load(tmp_path, monkeypatch):
    c, state = cache(tmp_path, monkeypatch)
    (tmp_path / "image.tar").write_bytes(b"corrupt")
    async with c.session():
        with pytest.raises(ImageCacheError, match="checksum"):
            async with c.lease_bundle(["hb__test"]):
                pytest.fail("corrupt image admitted")
    assert state["loads"] == 0


@pytest.mark.asyncio
async def test_live_container_prevents_deletion(tmp_path, monkeypatch):
    c, state = cache(tmp_path, monkeypatch)
    async with c.session():
        with pytest.raises(ImageCacheError, match="container"):
            async with c.lease_bundle(["hb__test"]):
                state["containers"] = "live-container"
    assert state["present"]


@pytest.mark.asyncio
async def test_second_manager_cannot_use_same_daemon_concurrently(
    tmp_path, monkeypatch
):
    c, _ = cache(tmp_path, monkeypatch)
    other = ImageCache(
        tmp_path, tmp_path, min_free_bytes=100, runtime_reserve_bytes=100
    )
    monkeypatch.setattr(other, "_docker", AsyncMock(return_value="test-daemon"))
    async with c.session():
        with pytest.raises(ImageCacheError, match="manager"):
            async with other.session():
                pytest.fail("second manager admitted")


@pytest.mark.asyncio
async def test_exception_releases_bundle_and_allows_next_trial(tmp_path, monkeypatch):
    c, state = cache(tmp_path, monkeypatch)
    async with c.session():
        with pytest.raises(ValueError, match="trial failed"):
            async with c.lease_bundle(["hb__test"]):
                raise ValueError("trial failed")
        assert not state["present"]
        async with c.lease_bundle(["hb__test"]):
            assert state["loads"] == 2


@pytest.mark.asyncio
async def test_unknown_alias_is_not_deleted(tmp_path, monkeypatch):
    c, state = cache(tmp_path, monkeypatch)
    old = c._docker

    async def docker(*args):
        if args[:2] == ("image", "inspect"):
            return json.dumps(
                [
                    {
                        "Id": "sha256:test",
                        "RepoTags": ["hb__test:latest", "someone-elses-image:latest"],
                    }
                ]
            )
        return await old(*args)

    monkeypatch.setattr(c, "_docker", docker)
    async with c.session():
        with pytest.raises(ImageCacheError, match="aliases"):
            async with c.lease_bundle(["hb__test"]):
                pass
    assert state["present"]


@pytest.mark.asyncio
async def test_cancellation_waits_for_image_cleanup(tmp_path, monkeypatch):
    import asyncio

    c, state = cache(tmp_path, monkeypatch)
    entered = asyncio.Event()

    async def work():
        async with c.session():
            async with c.lease_bundle(["hb__test"]):
                entered.set()
                await asyncio.Event().wait()

    task = asyncio.create_task(work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not state["present"]


@pytest.mark.asyncio
async def test_start_guard_failure_cannot_be_silently_scored_zero(
    tmp_path, monkeypatch
):
    c, _ = cache(tmp_path, monkeypatch)
    async with c.session():
        with pytest.raises(ImageCacheError, match="outside active bundle"):
            async with c.lease_bundle(["hb__test"]):
                # Harbor may catch trial exceptions. The bundle must still fail.
                with pytest.raises(ImageCacheError):
                    await c.require_image("unexpected-verifier-image")


@pytest.mark.asyncio
async def test_cancelling_load_waits_for_docker_to_finish_before_returning(
    tmp_path, monkeypatch
):
    import asyncio

    c, _ = cache(tmp_path, monkeypatch)
    entered, finish = asyncio.Event(), asyncio.Event()

    class Process:
        returncode = None
        killed = False

        async def communicate(self):
            entered.set()
            await finish.wait()
            self.returncode = 0
            return b"loaded", b""

        def kill(self):
            self.killed = True

        async def wait(self):
            return 0

    process = Process()
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    task = asyncio.create_task(
        ImageCache._docker(c, "image", "load", "--input", "archive.tar")
    )
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not process.killed
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode == 0


@pytest.mark.asyncio
async def test_overlapping_trials_share_images_until_last_release(
    tmp_path, monkeypatch
):
    c, state = cache(tmp_path, monkeypatch)
    first = c.lease_bundle(["hb__test"])
    second = c.lease_bundle(["hb__test"])
    async with c.session():
        await first.__aenter__()
        try:
            await second.__aenter__()
        except BaseException:
            await first.__aexit__(None, None, None)
            raise
        await first.__aexit__(None, None, None)
        assert state["present"] and state["loads"] == 1
        await c.require_image("hb__test")
        await second.__aexit__(None, None, None)
        assert not state["present"]


@pytest.mark.asyncio
async def test_cancelled_admission_does_not_evict_another_trials_image(
    tmp_path, monkeypatch
):
    c, state = cache(tmp_path, monkeypatch)
    entered = asyncio.Event()

    async def work():
        async with c.lease_bundle(["hb__test"]):
            entered.set()
            await asyncio.Event().wait()

    async with c.session():
        async with c.lease_bundle(["hb__test"]):
            worker = asyncio.create_task(work())
            try:
                await asyncio.wait_for(entered.wait(), 1)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            assert state["present"] and state["loads"] == 1
        assert not state["present"]


@pytest.mark.asyncio
async def test_disk_waiter_starts_when_one_trial_releases_while_another_runs(
    tmp_path, monkeypatch
):
    c, _ = cache(tmp_path, monkeypatch)
    c.images = {name: {"size_bytes": 100} for name in ("a", "b", "c")}
    present = set()
    waiting, started = asyncio.Event(), asyncio.Event()
    finish = asyncio.Event()

    def free():
        if present == {"a", "b"}:
            waiting.set()
        return 600 - len(present) * 200

    async def docker(*args):
        if args[0] == "info":
            return "test-daemon"
        assert args[:2] == ("image", "load")
        present.add(args[-1])
        return ""

    async def evict(name):
        present.discard(name)

    monkeypatch.setattr(c, "_docker", docker)
    monkeypatch.setattr(c, "_free_bytes", free)
    monkeypatch.setattr(
        c, "_present", AsyncMock(side_effect=lambda name: name in present)
    )
    monkeypatch.setattr(c, "_inspect", AsyncMock())
    monkeypatch.setattr(c, "_verify_archive", AsyncMock(side_effect=lambda name: name))
    monkeypatch.setattr(c, "_evict", evict)

    async def next_trial():
        async with c.lease_bundle(["c"]):
            started.set()
            await finish.wait()

    async with c.session():
        # B remains pinned throughout A's completion and C's admission.
        async with c.lease_bundle(["b"]):
            async with c.lease_bundle(["a"]):
                waiter = asyncio.create_task(next_trial())
                try:
                    await asyncio.wait_for(waiting.wait(), 1)
                    assert not started.is_set()
                except BaseException:
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)
                    raise
            try:
                await asyncio.wait_for(started.wait(), 1)
                assert present == {"b", "c"}
            finally:
                finish.set()
                await asyncio.wait_for(waiter, 1)
        assert not present
