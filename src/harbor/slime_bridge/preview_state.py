"""Best-effort HTTP readiness observation; never a verifier or quality score."""

import asyncio
import time
from datetime import UTC, datetime

import httpx


async def observe_output_state(task, public_url, *, timeout=10.0):
    build = next(
        (
            e.get("data") or {}
            for e in reversed(task.get("events") or [])
            if e.get("type") == "build_complete"
        ),
        {},
    )
    result = task.get("result") or {}
    state = {
        "observed_at": datetime.now(UTC).isoformat(),
        "task_status": task.get("status"),
        "generation_success": result.get("success"),
        "build_mode": build.get("mode"),
        "build_success": build.get("success"),
        "preview_status": "not_observed",
        "probe_timeout_seconds": timeout,
    }
    if result.get("success") is False or build.get("success") is False:
        state["preview_status"] = "generation_or_build_failed"
        return state
    if build.get("mode") == "none":
        state["preview_status"] = "no_artifact"
        return state
    url = public_url(result.get("preview_url"))
    if not url:
        state["preview_status"] = "no_preview_url"
        return state
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True, trust_env=False
            ) as client:
                async with client.stream(
                    "GET", url, headers={"Accept": "text/html"}
                ) as response:
                    state["http_status"] = response.status_code
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk[: max(0, 32768 - len(content))])
                        if len(content) >= 32768:
                            break
                    body = content.decode("utf-8", errors="replace").lower()
                    pending = any(
                        s in body
                        for s in (
                            "starting preview",
                            "preview is starting",
                            "preview is being prepared",
                            "preparing preview",
                            "live preview is starting",
                        )
                    )
                    if response.status_code == 202 or pending:
                        state["preview_status"] = "pending"
                    elif response.status_code == 200 and body.strip():
                        state["preview_status"] = "ready"
                    else:
                        state["preview_status"] = "unavailable"
    except (TimeoutError, httpx.TimeoutException):
        state["preview_status"] = "probe_timed_out"
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail a rollout
        # Never serialize exception text: HTTP exceptions can contain URL tokens.
        state["preview_status"] = "probe_error"
        state["probe_error_type"] = type(exc).__name__
    state["probe_seconds"] = round(time.monotonic() - started, 3)
    return state
