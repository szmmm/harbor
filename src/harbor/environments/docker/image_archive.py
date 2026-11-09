"""Prepare immutable archives before enabling bounded image residency.

Usage: python -m harbor.environments.docker.image_archive --help
Existing images are exported by default. Missing images require explicit
--build-missing. --evict removes exported images only after checksum validation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from harbor.environments.docker.image_cache import (
    ImageCache,
    ImageCacheError,
    file_sha256,
    task_image_names,
)


async def prepare(args: argparse.Namespace) -> None:
    import fcntl

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    with (args.archive_dir / ".prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ImageCacheError("Another archive preparation is running") from exc
        await _prepare_locked(args)


async def _prepare_locked(args: argparse.Namespace) -> None:
    root = args.archive_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        # Exclusive creation prevents an accidental concurrent initialization
        # from truncating another preparer's completed manifest.
        try:
            with manifest_path.open("x") as stream:
                stream.write('{"version": 1, "images": {}}')
        except FileExistsError:
            pass
    cache = ImageCache(
        root,
        args.disk_probe_path,
        min_free_bytes=int(args.min_free_gib * 2**30),
        runtime_reserve_bytes=int(args.build_reserve_gib * 2**30),
    )
    task_dirs = sorted(p.parent for p in args.dataset_dir.glob("*/task.toml"))
    if args.task:
        task_dirs = [p for p in task_dirs if p.name in args.task]
        if {p.name for p in task_dirs} != set(args.task):
            raise ImageCacheError("One or more requested tasks are missing")
    if not task_dirs:
        raise ImageCacheError(
            "No task directories found; pass the directory containing the tasks"
        )
    contexts = {}
    for task in task_dirs:
        names = task_image_names(task)
        for name, directory in zip(names, [task / "environment", task / "tests"]):
            contexts[name] = directory
    async with cache.session():
        # Refresh inside the lock: another preparation may have finished since
        # this process constructed its cache object.
        cache.images = json.loads(manifest_path.read_text())["images"]
        for name, directory in contexts.items():
            if name in cache.images:
                await cache._verify_archive(name)
                if await cache._present(name):
                    await cache._inspect(name)
            else:
                if not await cache._present(name):
                    if not args.build_missing:
                        raise ImageCacheError(
                            f"{name} is not local. Prepare on its owning node or use --build-missing"
                        )
                    if (
                        cache._free_bytes()
                        < cache.min_free_bytes + cache.runtime_reserve_bytes
                    ):
                        raise ImageCacheError(
                            "Insufficient disk space to build missing image"
                        )
                    await cache._docker(
                        "build", "--no-cache", "--tag", name, str(directory)
                    )
                info = json.loads(await cache._docker("image", "inspect", name))[0]
                image_id = info["Id"]
                archive = root / (image_id.split(":", 1)[1] + ".tar")
                temporary = archive.with_suffix(".tar.partial")
                try:
                    await cache._docker(
                        "image", "save", "--output", str(temporary), name
                    )
                    checksum = await asyncio.to_thread(file_sha256, temporary)
                    temporary.replace(archive)
                finally:
                    temporary.unlink(missing_ok=True)
                cache.images[name] = {
                    "image_id": image_id,
                    "archive": archive.name,
                    "sha256": checksum,
                    "size_bytes": info["Size"],
                    # Aliases are NOT automatically adopted: unrelated tags
                    # must continue to prevent destructive eviction.
                    "tags": [name + ":latest"],
                    "platform": f"{info['Os']}/{info['Architecture']}",
                    "context": str(directory),
                }
                temporary_manifest = root / "manifest.json.partial"
                temporary_manifest.write_text(
                    json.dumps({"version": 1, "images": cache.images}, indent=2)
                )
                temporary_manifest.replace(manifest_path)
            if args.evict:
                await cache._evict(name)
            print(
                json.dumps({"image": name, "archived": True, "evicted": args.evict}),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument(
        "--disk-probe-path",
        required=True,
        type=Path,
        help="Path backed by the Docker daemon disk, as visible to this process",
    )
    parser.add_argument(
        "--task", action="append", help="Prepare only these task directory names"
    )
    parser.add_argument("--min-free-gib", type=float, default=25)
    parser.add_argument("--build-reserve-gib", type=float, default=16)
    parser.add_argument("--build-missing", action="store_true")
    parser.add_argument("--evict", action="store_true")
    asyncio.run(prepare(parser.parse_args()))


if __name__ == "__main__":
    main()
