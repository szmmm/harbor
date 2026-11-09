"""Archive-backed, daemon-exclusive image residency for concurrent trials.

Each trial leases its agent and verifier images until cleanup completes.
Overlapping trials share reference-counted leases. Only manifest-owned images are evicted. No build/pull is permitted
inside a leased job: an incomplete archive is an infrastructure error.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, AsyncIterator

active_image_cache: ContextVar[ImageCache | None] = ContextVar(
    "image_cache", default=None
)


class ImageCacheError(RuntimeError):
    """Infrastructure failure; must abort rollout rather than become reward zero."""


class ImageCacheSpaceError(ImageCacheError):
    """The requested trial images cannot fit within the configured disk budget."""


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def task_image_names(task_dir: Path) -> list[str]:
    """Resolve the two Dockerfile contexts used by our R2E-Gym profile.

    Reject unsupported task layouts instead of silently missing sidecar/step
    images. The ordinary, non-cache execution path remains unrestricted.
    """
    import tomllib

    from harbor.environments.definition import environment_content_hash

    config = tomllib.loads((task_dir / "task.toml").read_text())
    verifier = config.get("verifier", {})
    if config.get("steps") or verifier.get("steps"):
        raise ImageCacheError(
            f"Step-based tasks are not supported by image cache: {task_dir}"
        )
    contexts = [(task_dir / "environment", config.get("environment", {}))]
    if verifier.get("environment_mode") == "separate":
        contexts.append((task_dir / "tests", verifier.get("environment", {})))
    names = []
    for directory, env in contexts:
        if env.get("docker_image") or (directory / "docker-compose.yaml").exists():
            raise ImageCacheError(
                f"Archive mode requires a simple Dockerfile: {directory}"
            )
        if not (directory / "Dockerfile").is_file():
            raise ImageCacheError(f"Missing Dockerfile: {directory}")
        names.append("hb__" + environment_content_hash(directory))
    return names


class ImageCache:
    def __init__(
        self,
        archive_dir: Path,
        disk_probe_path: Path,
        *,
        min_free_bytes: int,
        runtime_reserve_bytes: int,
    ):
        self.archive_dir = archive_dir.resolve()
        self.disk_probe_path = disk_probe_path.resolve(strict=True)
        if min_free_bytes < 0 or runtime_reserve_bytes < 0:
            raise ValueError("Disk budgets must be nonnegative")
        self.min_free_bytes = min_free_bytes
        self.runtime_reserve_bytes = runtime_reserve_bytes
        manifest = json.loads((self.archive_dir / "manifest.json").read_text())
        if manifest.get("version") != 1 or not isinstance(manifest.get("images"), dict):
            raise ImageCacheError("Unsupported image archive manifest")
        self.images = manifest["images"]
        self._leased: dict[str, int] = {}
        self._condition = asyncio.Condition()
        self._session_open = False
        self._verified: dict[str, tuple[int, int]] = {}
        self.failure: str | None = None

    async def _docker(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def complete():
            try:
                return await asyncio.wait_for(process.communicate(), timeout=1800)
            except BaseException:
                if process.returncode is None:
                    process.kill()
                await process.wait()
                raise

        # Cancelling the HTTP client does not reliably cancel an operation in
        # dockerd. Drain it before releasing the lease or removing its image.
        operation = asyncio.create_task(complete())
        try:
            stdout, stderr = await asyncio.shield(operation)
        except asyncio.CancelledError:
            await operation
            raise
        if process.returncode:
            raise ImageCacheError(
                f"Docker {args[:2]} failed: {stderr.decode(errors='replace')[-2000:]}"
            )
        return stdout.decode()

    def _free_bytes(self) -> int:
        return shutil.disk_usage(self.disk_probe_path).free

    async def _present(self, name: str) -> bool:
        return bool(
            (await self._docker("image", "ls", "--no-trunc", "--quiet", name)).strip()
        )

    async def _verify_archive(self, name: str) -> Path:
        if name not in self.images:
            raise ImageCacheError(f"Image missing from archive manifest: {name}")
        record = self.images[name]
        archive = (self.archive_dir / record["archive"]).resolve()
        if not archive.is_relative_to(self.archive_dir):
            raise ImageCacheError(f"Archive path escapes archive directory: {name}")
        stat = archive.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if self._verified.get(name) != fingerprint:
            if await asyncio.to_thread(file_sha256, archive) != record["sha256"]:
                raise ImageCacheError(f"Archive checksum mismatch: {name}")
            self._verified[name] = fingerprint
        return archive

    async def _inspect(self, name: str) -> dict[str, Any]:
        info = json.loads(await self._docker("image", "inspect", name))[0]
        if info["Id"] != self.images[name]["image_id"]:
            raise ImageCacheError(f"Image identity changed: {name}")
        return info

    async def require_image(self, name: str) -> None:
        """Guard DockerEnvironment.start before any build, pull, or container creation."""
        if name not in self._leased:
            self.failure = f"Image outside active bundle: {name}"
            raise ImageCacheError(self.failure)
        try:
            await self._inspect(name)
        except ImageCacheError as exc:
            self.failure = str(exc)
            raise

    async def _evict(self, name: str) -> None:
        if not await self._present(name):
            return
        await self._verify_archive(name)
        info = await self._inspect(name)
        refs = (
            await self._docker("ps", "-aq", "--filter", f"ancestor={info['Id']}")
        ).strip()
        if refs:
            raise ImageCacheError(
                f"Cannot evict {name}: container references remain: {refs}"
            )
        tags = info.get("RepoTags") or []
        allowed = {name, name + ":latest", *self.images[name].get("tags", [])}
        if set(tags) - allowed:
            raise ImageCacheError(
                f"Unmanaged aliases prevent eviction of {name}: {tags}"
            )
        await self._docker("image", "rm", *(tags or [info["Id"]]))
        if await self._present(name):
            raise ImageCacheError(f"Image still present after eviction: {name}")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ImageCache]:
        # All managers of a daemon must use this same archive root. flock is
        # nonblocking: a duplicate runner must fail rather than deadlock a job.
        import fcntl

        daemon = (await self._docker("info", "--format", "{{.ID}}")).strip()
        if not daemon:
            raise ImageCacheError("Docker daemon identity is unavailable")
        locks = self.archive_dir / ".locks"
        locks.mkdir(exist_ok=True)
        lock_path = locks / (hashlib.sha256(daemon.encode()).hexdigest() + ".lock")
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ImageCacheError(
                    "Another image manager owns this Docker daemon"
                ) from exc
            self._session_open = True
            self.failure = None
            token = active_image_cache.set(self)
            try:
                yield self
            finally:
                active_image_cache.reset(token)
                self._session_open = False
                fcntl.flock(lock, fcntl.LOCK_UN)

    @asynccontextmanager
    async def lease_bundle(self, names: list[str]) -> AsyncIterator[None]:
        """Pin one trial's images without waiting for unrelated trials to finish."""
        if not self._session_open:
            raise ImageCacheError("Bundles require an exclusive session")
        names = list(dict.fromkeys(names))
        acquired = False
        try:
            # Serialize image mutations and disk admission, not trial execution.
            async with self._condition:
                for name in names:
                    await self._verify_archive(name)
                while True:
                    if self.failure:
                        raise ImageCacheError(self.failure)
                    missing = [name for name in names if not await self._present(name)]
                    needed = (
                        self.min_free_bytes
                        + self.runtime_reserve_bytes
                        + sum(
                            int(self.images[name]["size_bytes"]) * 2 for name in missing
                        )
                    )
                    if self._free_bytes() < needed:
                        # Reclaim only idle, manifest-owned images, including
                        # leftovers from an interrupted run. Leased images stay.
                        for name in self.images:
                            if name not in self._leased and name not in names:
                                await self._evict(name)
                    if self._free_bytes() >= needed:
                        break
                    if not self._leased:
                        raise ImageCacheSpaceError(
                            f"Insufficient disk space: need {needed} bytes, free {self._free_bytes()}"
                        )
                    # A finishing trial can free disk independently of the rest
                    # of the Job. Release the lock so its cleanup can proceed.
                    await self._condition.wait()

                for name in names:
                    self._leased[name] = self._leased.get(name, 0) + 1
                acquired = True
                for name in names:
                    if name in missing:
                        archive = await self._verify_archive(name)
                        await self._docker("image", "load", "--input", str(archive))
                    await self._inspect(name)
                    if (
                        self._free_bytes()
                        < self.min_free_bytes + self.runtime_reserve_bytes
                    ):
                        raise ImageCacheSpaceError(
                            "Insufficient disk space after loading image"
                        )
            yield
            if self.failure:
                raise ImageCacheError(self.failure)
        finally:
            if acquired:
                cleanup = asyncio.create_task(self._release_bundle(names))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise

    async def _release_bundle(self, names: list[str]) -> None:
        async with self._condition:
            errors = []
            try:
                for name in names:
                    self._leased[name] -= 1
                    if self._leased[name]:
                        continue
                    del self._leased[name]
                    try:
                        await self._evict(name)
                    except Exception as exc:
                        errors.append(str(exc))
                if errors:
                    self.failure = "; ".join(errors)
                    raise ImageCacheError(self.failure)
            finally:
                self._condition.notify_all()
