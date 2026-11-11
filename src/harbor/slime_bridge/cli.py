"""The harbor rollout service: how ``harbor.slime_bridge.rollout`` runs trials.

This is the only path.  Harbor needs a Docker daemon, the task dataset,
~1 workspace + 1 vite dev server per concurrent trial and a Chromium verifier
container; slime places the Ray rollout actor on a GPU node, which is the wrong
host for all of that.  So the actor holds no harbor code at all -- it POSTs here
and waits (``harbor.slime_bridge.rollout._run_harbor``).

Run this on the CPU host that already owns Docker and the DDB stack::

    source env.sh
    python -m harbor.slime_bridge.cli --port 9500 --n-concurrent-trials 8

then launch training with ``--harbor-service-url http://<host>:9500``.

**Requests are serialized; trials are not.**  One ``/rollout`` runs
``len(task_ids) * n_attempts`` trials with ``n_concurrent_trials`` of them in
flight at once, each in its own container -- that is the whole point of the
service, and the lock below does not restrict it.  What is serialized is the
*request*, i.e. one slime training step, for three reasons:

* Every trial of a step must sample from one policy version.  slime's own loop
  is synchronous so a healthy trainer never overlaps; the lock is for the rest
  -- a client retry after a network blip, a restarted trainer resuming while the
  previous job's trials are still live, a second trainer or a hand-run curl.
  Each of those straddles an ``update_weights()`` boundary and mixes two
  policies into one batch, silently.
* ``n_concurrent_trials`` is a **per-Job** semaphore
  (``harbor/trial/queue.py``), not a service-wide one, so two overlapping
  requests mean two semaphores: double the containers, workspaces, vite servers
  and Chromium instances on a host sized for one.
* ``jobs_dir`` is keyed by ``run_label``, so two requests sharing a label (a
  retried ``rollout_7``, or two unlabelled calls) would write into the same
  directory and harbor would try to resume one as the other.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hmac
import logging
import os
import re
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from harbor.slime_bridge.harbor_client import HarborConfig, HarborRolloutClient

logger = logging.getLogger(__name__)


class RolloutRequest(BaseModel):
    """Body of ``POST /rollout``.

    Module scope, not nested inside ``build_app``, and that is load-bearing:
    this module uses postponed annotations, so FastAPI sees the *string*
    ``"RolloutRequest"`` and resolves it through the endpoint's module globals.
    A class defined in ``build_app``'s local scope is not there, so resolution
    fails, FastAPI falls back to treating the parameter as a **query** param.
    """

    task_ids: list[str] = Field(min_length=1)
    n_attempts: int = Field(default=1, ge=1)
    run_label: str | None = None
    #: Per-task early stop: task id -> usable trials wanted.  Once a task has
    #: that many usable trials, its remaining attempts are cancelled and the
    #: cancelled trials are dropped from the response.  Tasks not listed run
    #: all ``n_attempts``.  Validated against ``task_ids`` by the client.
    n_targets: dict[str, int] | None = None


_RUNNER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def runner_jobs_dir(base: Path, runner_id: str) -> Path:
    """Namespace Harbor job output without allowing runner ids to escape it."""
    if runner_id in {".", ".."} or not _RUNNER_ID_PATTERN.fullmatch(runner_id):
        raise ValueError(
            f"invalid runner id {runner_id!r}; use only letters, digits, '.', '_', or '-'"
        )
    return base / runner_id


def build_app(
    client: HarborRolloutClient,
    *,
    runner_id: str = "local",
    profile_id: str = "ddb",
    auth_token: str | None = None,
    manage_callback: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not manage_callback:
            yield
            return
        # Hold one reference for the service lifetime. Trials share this exact
        # listener, so callback conflicts fail startup instead of the first batch.
        from harbor.agents.installed.ddb import (
            DEFAULT_CALLBACK_PORT,
            _acquire_listener,
            _default_callback_secret,
            _release_listener,
        )

        listener = await _acquire_listener(
            bind_host=os.getenv("DDB_CALLBACK_HOST") or "0.0.0.0",
            bind_port=int(os.getenv("DDB_CALLBACK_PORT") or DEFAULT_CALLBACK_PORT),
            advertised_url=os.getenv("DDB_CALLBACK_URL") or None,
            secret=os.getenv("DDB_CALLBACK_SECRET") or _default_callback_secret(),
            probe_host=urlsplit(os.getenv("DDB_BASE_URL", "")).hostname or "127.0.0.1",
        )
        app.state.callback_listener = listener
        try:
            yield
        finally:
            app.state.callback_listener = None
            await _release_listener()

    app = FastAPI(title="harbor.slime_bridge rollout service", lifespan=lifespan)
    lock = asyncio.Lock()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "busy": lock.locked(),
            "dataset": str(client.config.dataset_dir),
            "n_concurrent_trials": client.config.n_concurrent_trials,
            "model_name": client.config.model_name,
            "runner_id": runner_id,
            "profile_id": profile_id,
        }

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        listener = getattr(app.state, "callback_listener", None)
        if (
            listener is None
            or listener._server is None
            or not listener._server.is_serving()
        ):
            raise HTTPException(
                status_code=503, detail="callback listener is not ready"
            )
        return {
            **await health(),
            "status": "ready",
            "pid": os.getpid(),
            "jobs_dir": str(client.config.jobs_dir),
            "callback_url": listener.advertised_url,
        }

    @app.post("/rollout")
    async def rollout(
        request: RolloutRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if (
            auth_token is not None
            and auth_token != ""
            and not hmac.compare_digest(authorization or "", f"Bearer {auth_token}")
        ):
            raise HTTPException(status_code=401, detail="invalid bearer token")
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail="a rollout is already running; slime steps must not overlap",
            )
        async with lock:
            try:
                results = await client.run(
                    request.task_ids,
                    n_attempts=request.n_attempts,
                    run_label=request.run_label,
                    n_targets=request.n_targets,
                )
            except (
                ValueError
            ) as exc:  # bad task id / unknown task directory / bad n_targets
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            task_id: [dataclasses.asdict(trial) for trial in trials]
            for task_id, trials in results.items()
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9500)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=None)
    parser.add_argument("--n-concurrent-trials", type=int, default=None)
    parser.add_argument("--timeout-multiplier", type=float, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--runner-id",
        default=os.getenv("HARBOR_RUNNER_ID") or socket.gethostname(),
    )
    parser.add_argument(
        "--profile-id",
        default=os.getenv("HARBOR_PROFILE_ID", "ddb"),
        choices=["ddb"],
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = HarborConfig.from_env(
        dataset_dir=args.dataset_dir,
        jobs_dir=args.jobs_dir,
        n_concurrent_trials=args.n_concurrent_trials,
        timeout_multiplier=args.timeout_multiplier,
        model_name=args.model_name,
    )
    config = dataclasses.replace(
        config,
        jobs_dir=runner_jobs_dir(config.jobs_dir, args.runner_id),
    )
    client = HarborRolloutClient(config)
    auth_token = os.getenv("HARBOR_RUNNER_TOKEN")
    if auth_token is None or auth_token == "":
        logger.warning(
            "$HARBOR_RUNNER_TOKEN is unset; POST /rollout is unauthenticated"
        )

    logger.info(
        "serving harbor runner %s (profile %s) on %s:%d",
        args.runner_id,
        args.profile_id,
        args.host,
        args.port,
    )
    # timeout_keep_alive is generous: a rollout step legitimately takes hours.
    uvicorn.run(
        build_app(
            client,
            runner_id=args.runner_id,
            profile_id=args.profile_id,
            auth_token=auth_token,
            manage_callback=True,
        ),
        host=args.host,
        port=args.port,
        timeout_keep_alive=3600,
        timeout_graceful_shutdown=20,
    )


if __name__ == "__main__":
    main()
