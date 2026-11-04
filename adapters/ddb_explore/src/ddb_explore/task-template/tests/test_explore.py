#!/usr/bin/env python3
"""Evaluate a running app against the task's rubric tree with the browser judge."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from browser_env import PlaywrightEnv, resolve_device
from judge import RubricJudge
from probe import ProbeConfig


APP_SPEC_PATH = Path("/tests/app_spec.json")
RUBRIC_PATH = Path("/tests/rubric.json")
APP_DIR = Path("/app/workspace")
REPORT_PATH = Path("/logs/verifier/explore-report.json")
TRACE_PATH = Path("/logs/verifier/explore.trace.jsonl")
APP_READY_TIMEOUT = float(os.environ.get("APP_READY_TIMEOUT", "120"))
DEV_SERVER_STARTUP_TIMEOUT = 60.0

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


def _trace_writer(
    path: Path,
) -> tuple[Callable[[dict[str, Any]], None], Callable[[], None]]:
    """Append and flush one JSON object for each multiturn probe event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")

    seen_messages = 0

    def sink(event: dict[str, Any]) -> None:
        nonlocal seen_messages
        if event.get("kind") == "judge_request":
            messages = event["request"]["messages"]
            incoming = copy.deepcopy(
                [m for m in messages[seen_messages:] if m["role"] != "assistant"]
            )
            for message in incoming:
                if not isinstance(message.get("content"), list):
                    continue
                evidence_id = None
                for item in message["content"]:
                    if item.get("type") == "text" and item["text"].startswith("{"):
                        evidence_id = json.loads(item["text"]).get("evidence_id", evidence_id)
                    elif item.get("type") == "image_url":
                        item.clear()
                        item.update(type="image_reference", path=f"evidence/{evidence_id}.png")
            event = {k: v for k, v in event.items() if k != "request"}
            event["messages"] = incoming
            seen_messages = len(messages)
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()

    return sink, handle.close


def _write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def start_dev_server() -> tuple[subprocess.Popen[str], str] | None:
    """Start the generated app and return its process and local URL."""
    package_json = APP_DIR / "package.json"
    if not package_json.is_file():
        print(f"ERROR: {package_json} not found")
        return None
    try:
        package = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: cannot read package.json: {exc}")
        return None
    scripts = package.get("scripts") or {}
    if "dev" in scripts:
        command = ["npm", "run", "dev"]
    elif "start" in scripts:
        command = ["npm", "run", "start"]
    else:
        print("ERROR: package.json has no dev or start script")
        return None

    subprocess.run(["npm", "install"], cwd=APP_DIR, check=True, timeout=120)
    process = subprocess.Popen(
        command,
        cwd=APP_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    ready = threading.Event()
    detected: list[str] = []
    url_pattern = re.compile(r"https?://(?:localhost|127\.0\.0\.1):(\d+)")

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            match = url_pattern.search(line)
            if match and not ready.is_set():
                detected.append(f"http://localhost:{match.group(1)}")
                ready.set()

    threading.Thread(target=read_output, daemon=True).start()
    if not ready.wait(DEV_SERVER_STARTUP_TIMEOUT):
        process.terminate()
        print("ERROR: app dev server did not publish a local URL")
        return None
    return process, detected[0]


def wait_for_app_ready(url: str, timeout: float) -> str | None:
    """Return the readiness failure reason, or None when ready."""
    deadline = time.monotonic() + timeout
    last_error = "No readiness attempt completed"
    while time.monotonic() < deadline:
        try:
            request = Request(url, headers={"User-Agent": "ddb-exploratory-probe"})
            with urlopen(request, timeout=15) as response:
                body = response.read(65536).decode("utf-8", errors="replace")
                placeholder = (
                    "__deepdiver_ready" in body or "Finalizing your app" in body
                )
                if response.status == 200 and body and not placeholder:
                    return None
                reason = "publishing placeholder" if placeholder else "empty response body"
                last_error = f"HTTP {response.status}: {reason}"
        except Exception as exc:
            # URLs can contain preview tokens; keep them out of failure reports.
            detail = re.sub(r"https?://[^\s<>\"']+", "<URL redacted>", str(exc))
            last_error = f"{type(exc).__name__}: {detail[:300]}"
        time.sleep(min(3, max(0, deadline - time.monotonic())))
    error = f"Preview readiness timed out after {timeout:.0f}s; last check: {last_error}"
    print(error, flush=True)
    return error


async def run_probe(
    app_url: str, app_spec: dict[str, Any], rubric: dict[str, Any]
) -> dict[str, Any]:
    """Judge every leaf of the task's rubric tree on the running app."""
    model = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct").strip()
    config = ProbeConfig(
        model=model,
        api_key=os.environ.get("VLM_API_KEY", "").strip(),
        base_url=os.environ.get("VLM_API_BASE_URL", "").strip(),
        max_steps=int(os.environ.get("VLM_EXPLORE_MAX_STEPS", "40")),
        request_timeout_s=float(os.environ.get("VLM_REQUEST_TIMEOUT_S", "120")),
        max_retries=1,
        temperature=0,
        omit_temperature=_truthy("VLM_OMIT_TEMPERATURE", "1"),
    )
    deadline_s = float(os.environ.get("VLM_EXPLORE_DEADLINE_S", "1500"))
    min_coverage = float(os.environ.get("VLM_RUBRIC_MIN_COVERAGE", "0.9"))
    device_name = (
        os.environ.get("VLM_EXPLORE_DEVICE", "").strip() or app_spec["default_device"]
    )
    device = (
        None if device_name.casefold() == "desktop" else resolve_device(device_name)
    )

    sink, close_trace = _trace_writer(TRACE_PATH)
    env = PlaywrightEnv(
        headless=True,
        has_touch=device is not None,
        device=device,
        viewport=None if device else {"width": 1024, "height": 1000},
    )
    try:
        await env.start()
        try:
            result = await RubricJudge(
                env,
                config,
                REPORT_PATH.parent / "evidence",
                deadline_s=deadline_s,
                source_root=APP_DIR,
                trace_sink=sink,
            ).evaluate(app_url, rubric, device=device, min_coverage=min_coverage)
        finally:
            await env.stop()
    finally:
        close_trace()

    return {
        **result,
        "model": model,
        "app_url": app_url,
        "device": asdict(device)
        if device
        else {"name": "desktop", "width": 1024, "height": 1000},
        "config": {
            "max_steps": config.max_steps,
            "deadline_s": deadline_s,
            "temperature": None if config.omit_temperature else config.temperature,
        },
        "trace_path": str(TRACE_PATH),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not _truthy("VLM_EXPLORE_ENABLED", "1"):
        _write_report(
            {
                "status": "skipped",
                "scoring": "disabled",
                "reason": "VLM_EXPLORE_ENABLED is false",
                "findings": [],
            }
        )
        return 0
    if not os.environ.get("VLM_API_KEY", "").strip():
        _write_report(
            {
                "status": "skipped",
                "scoring": "disabled",
                "reason": "VLM_API_KEY is not set",
                "findings": [],
            }
        )
        print("=== Exploratory probe skipped: VLM_API_KEY is not set ===")
        return 0
    for path in (APP_SPEC_PATH, RUBRIC_PATH):
        if not path.is_file():
            _write_report(
                {
                    "status": "evaluator_failed",
                    "scoring": "disabled",
                    "error": f"{path} not found",
                    "findings": [],
                }
            )
            return 1

    app_spec = json.loads(APP_SPEC_PATH.read_text(encoding="utf-8"))
    rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    override_url = os.environ.get("APP_BASE_URL", "").strip()
    process = None
    if override_url:
        app_url = override_url
        readiness_error = wait_for_app_ready(app_url, APP_READY_TIMEOUT)
        if readiness_error is not None:
            _write_report(
                {
                    "status": "app_unavailable",
                    "score": None,
                    "error": readiness_error,
                }
            )
            return 1
    else:
        started = start_dev_server()
        if started is None:
            _write_report(
                {
                    "status": "app_unavailable",
                    "scoring": "disabled",
                    "error": "dev server did not start",
                    "findings": [],
                }
            )
            return 1
        process, app_url = started

    try:
        try:
            report = asyncio.run(run_probe(app_url, app_spec, rubric))
        except Exception as exc:
            report = {
                "status": "evaluator_failed",
                "scoring": "disabled",
                "error": f"{type(exc).__name__}: {exc}",
                "findings": [],
                "trace_path": str(TRACE_PATH),
            }
        _write_report(report)
        status = str(report.get("status") or "evaluator_failed")
        print(f"=== Exploratory probe report: {REPORT_PATH} ({status}) ===")
        return 0 if status == "completed" else 1
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
