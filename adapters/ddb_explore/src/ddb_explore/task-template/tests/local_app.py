"""Start downloaded workspaces with bounded installation and process cleanup."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class LocalAppError(RuntimeError):
    pass


def stop_process(process: subprocess.Popen[Any]) -> None:
    """Stop the whole session, including children after the launcher exits."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def package_commands(
    root: Path, package: dict[str, Any]
) -> tuple[list[str], list[str]]:
    declared = str(package.get("packageManager", ""))
    manager = declared.split("@", 1)[0] if declared else ""
    if not manager:
        locks = [
            ("pnpm", "pnpm-lock.yaml"),
            ("yarn", "yarn.lock"),
            ("npm", "package-lock.json"),
            ("npm", "npm-shrinkwrap.json"),
        ]
        found = {name for name, lock in locks if (root / lock).exists()}
        if len(found) > 1:
            raise LocalAppError(
                "Multiple package-manager lockfiles without packageManager"
            )
        manager = next(iter(found), "npm")
    if manager not in {"npm", "pnpm", "yarn"}:
        raise LocalAppError(f"Unsupported package manager: {manager}")
    # Corepack honors packageManager (including its pinned version).
    if declared and manager != "npm" and shutil.which("corepack") is not None:
        prefix = ["corepack", manager]
    elif shutil.which(manager) is not None:
        if declared and "@" in declared:
            expected = declared.split("@", 1)[1].split("+", 1)[0]
            actual = subprocess.check_output(
                [manager, "--version"], text=True, timeout=30
            ).strip()
            if actual != expected:
                raise LocalAppError(
                    f"{manager} version mismatch: expected {expected}, installed {actual}"
                )
        prefix = [manager]
    else:
        raise LocalAppError(f"Required package manager {manager} is not installed")
    if manager == "pnpm":
        install = (
            ["install", "--frozen-lockfile"]
            if (root / "pnpm-lock.yaml").exists()
            else ["install"]
        )
    elif manager == "yarn":
        # Yarn 1 and modern Yarn use different immutable-install flags.
        version = subprocess.check_output(
            [*prefix, "--version"], text=True, timeout=30
        ).strip()
        install = [
            "install",
            "--frozen-lockfile" if version.startswith("1.") else "--immutable",
        ]
    else:
        locked = any(
            (root / name).exists()
            for name in ("package-lock.json", "npm-shrinkwrap.json")
        )
        install = ["ci" if locked else "install", "--no-audit", "--no-fund"]
    scripts = package.get("scripts") or {}
    script = next((name for name in ("dev", "start") if name in scripts), None)
    if script is None:
        raise LocalAppError("package.json has no dev or start script")
    return [*prefix, *install], [*prefix, "run", script]


def validate_local_archives(root: Path, package: dict[str, Any]) -> None:
    for section in ("dependencies", "devDependencies", "optionalDependencies"):
        for name, spec in (package.get(section) or {}).items():
            if not isinstance(spec, str) or not spec.startswith("file:"):
                continue
            relative = spec[5:]
            if not relative.endswith((".tgz", ".tar.gz")):
                continue
            archive_path = (root / relative).resolve()
            if not archive_path.is_relative_to(root.resolve()):
                raise LocalAppError(
                    f"Local archive for {name} is outside the workspace"
                )
            try:
                with tarfile.open(archive_path, "r:gz") as archive:
                    members = archive.getmembers()
                    if not any(
                        m.name.lstrip("./") == "package/package.json" for m in members
                    ):
                        raise LocalAppError(
                            f"Local archive for {name} has no package/package.json"
                        )
            except (OSError, EOFError, tarfile.TarError) as exc:
                raise LocalAppError(
                    f"Invalid or missing local dependency archive: {relative}"
                ) from exc


def start_local_app(
    root: Path,
    log_path: Path,
    readiness: Callable[[str, float], str | None],
    startup_timeout: float = 60,
    install_timeout: float = 300,
) -> tuple[subprocess.Popen[str], str]:
    try:
        package = json.loads((root / "package.json").read_text())
    except (OSError, ValueError) as exc:
        raise LocalAppError("Workspace package.json is missing or invalid") from exc
    validate_local_archives(root, package)
    install, command = package_commands(root, package)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        installer = subprocess.Popen(
            install,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "CI": "true"},
        )
        try:
            code = installer.wait(timeout=install_timeout)
        except subprocess.TimeoutExpired as exc:
            raise LocalAppError(
                f"Dependency installation timed out after {install_timeout}s; see local-app.log"
            ) from exc
        finally:
            stop_process(installer)
        if code:
            raise LocalAppError(
                f"Dependency installation exited {code}; see local-app.log"
            )
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    ready = threading.Event()
    urls: list[str] = []

    def read_output() -> None:
        if process.stdout is None:
            return
        with log_path.open("a") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                match = re.search(
                    r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d+)", clean
                )
                if match and not urls:
                    urls.append(f"http://127.0.0.1:{match.group(1)}")
                    ready.set()

    threading.Thread(target=read_output, daemon=True).start()
    try:
        deadline = time.monotonic() + startup_timeout
        while not ready.wait(0.1):
            if process.poll() is not None:
                raise LocalAppError(
                    f"App server exited {process.returncode}; see local-app.log"
                )
            if time.monotonic() >= deadline:
                raise LocalAppError(
                    "App server did not publish a local URL; see local-app.log"
                )
        error = readiness(urls[0], max(1, deadline - time.monotonic()))
        if error is not None:
            raise LocalAppError(f"Local app readiness failed: {error}")
        return process, urls[0]
    except BaseException:
        stop_process(process)
        raise
