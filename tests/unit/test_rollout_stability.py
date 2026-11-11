"""Rollout-only failures remain observable without rewards or refills."""

from harbor.slime_bridge.tests import conftest as bridge_test_support  # noqa: F401

# isort: split

import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from slime.utils.types import Sample

from harbor.slime_bridge import preview_state, stability
from harbor.slime_bridge.harbor_client import HarborConfig, HarborRolloutClient
from harbor.slime_bridge.rollout import generate_rollout
from harbor.slime_bridge.trial import TrialRollout


@pytest.mark.parametrize(
    "code,body,expected",
    [
        (200, "<html>app</html>", "ready"),
        (202, "<html>pending</html>", "pending"),
        (200, "Starting preview", "pending"),
        (404, "missing", "unavailable"),
        (200, "", "unavailable"),
    ],
)
def test_preview_observation(monkeypatch, code, body, expected):
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(code, text=body))
    monkeypatch.setattr(
        preview_state.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=transport, **kw),
    )
    task = {
        "status": "completed",
        "result": {"success": True, "preview_url": "https://test/preview?token=secret"},
    }
    result = asyncio.run(preview_state.observe_output_state(task, lambda url: url))
    assert result["preview_status"] == expected
    assert "secret" not in json.dumps(result)


def test_timeout_is_unknown_not_app_failure(monkeypatch):
    real_client = httpx.AsyncClient

    def timeout(request):
        raise httpx.ReadTimeout("private URL token")

    transport = httpx.MockTransport(timeout)
    monkeypatch.setattr(
        preview_state.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=transport, **kw),
    )
    result = asyncio.run(
        preview_state.observe_output_state(
            {"result": {"preview_url": "https://test"}}, lambda url: url
        )
    )
    assert result["preview_status"] == "probe_timed_out"
    assert "private" not in json.dumps(result)


def test_no_artifact_never_requests_preview(monkeypatch):
    monkeypatch.setattr(
        preview_state.httpx, "AsyncClient", lambda **kw: pytest.fail("must not probe")
    )
    task = {
        "result": {"success": True},
        "events": [{"type": "build_complete", "data": {"mode": "none"}}],
    }
    assert (
        asyncio.run(preview_state.observe_output_state(task, lambda url: url))[
            "preview_status"
        ]
        == "no_artifact"
    )


def test_all_failed_batch_is_recorded_and_returns_without_refill(monkeypatch, tmp_path):
    args = Namespace(
        rollout_batch_size=2, n_samples_per_prompt=2, harbor_stability_dir=str(tmp_path)
    )
    groups = [
        [Sample(metadata={"task_id": task}) for _ in range(2)]
        for task in ["one", "two"]
    ]
    source = SimpleNamespace(
        get_samples=lambda n: groups, add_samples=lambda groups: None
    )
    monkeypatch.setattr(stability, "_sync_sampling_overrides", AsyncMock())
    invoke = AsyncMock(
        return_value={
            "one": [
                TrialRollout(
                    task_id="one",
                    error="task failed",
                    output_state={"preview_status": "generation_or_build_failed"},
                )
            ],
            "two": [],
        }
    )
    monkeypatch.setattr(stability, "_run_harbor_distributed", invoke)
    result = asyncio.run(stability._rollout(args, 1, source))
    invoke.assert_awaited_once()
    assert len(result.samples) == 4
    assert all(s.reward is None for s in result.samples)
    assert result.metrics["stability/errors"] == 1
    assert result.metrics["stability/missing"] == 3
    report = json.loads((tmp_path / "step-0001.json").read_text())
    assert report["optimizer_updates"] is False
    assert (
        report["trials"][0]["output_state"]["preview_status"]
        == "generation_or_build_failed"
    )


def test_disabled_verification_cannot_enter_training():
    with pytest.raises(RuntimeError, match="rollout-only"):
        generate_rollout(Namespace(harbor_disable_verification=True), 1, None)


@pytest.mark.parametrize("disable", [False, True])
def test_job_verification_is_opt_in_disabled(monkeypatch, tmp_path, disable):
    from harbor.slime_bridge import harbor_client

    (tmp_path / "one").mkdir()
    cfg = HarborConfig.from_env(
        dataset_dir=tmp_path,
        proxy_url="http://proxy",
        ddb_base_url="http://ddb",
        ddb_api_key="test",
        ddb_preview_base_url="http://preview",
        vlm_api_base_url="http://judge",
        vlm_api_key="test",
        vlm_model="test",
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            run=AsyncMock(return_value=SimpleNamespace(trial_results=[]))
        )
    )
    monkeypatch.setattr(harbor_client.Job, "create", create)
    asyncio.run(HarborRolloutClient(cfg).run(["one"], disable_verification=disable))
    job = create.call_args.args[0]
    assert job.verifier.disable is disable
    assert job.agents[0].kwargs["log_preview_state"] is disable
