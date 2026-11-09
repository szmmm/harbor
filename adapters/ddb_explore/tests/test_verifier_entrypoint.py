"""Verify reward switches without browsers or model calls."""

import json
import os
import subprocess
from pathlib import Path

import combine_rewards
import pytest
import test_trajectory as trajectory

TEMPLATE = Path(__file__).resolve().parents[1] / "src/ddb_explore/task-template/tests"


@pytest.mark.parametrize("flag,expected", [("0", False), ("1", True)])
def test_shell_trajectory_switch(tmp_path, flag, expected):
    calls = tmp_path / "calls"
    # Replace only the absolute log/output paths; intercept child executables.
    script = (
        (TEMPLATE / "test.sh")
        .read_text()
        .replace("/logs/verifier", str(tmp_path / "logs"))
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("timeout", "python3"):
        exe = bindir / name
        exe.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
        exe.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bindir) + ":" + os.environ["PATH"],
        "CALLS": str(calls),
        "TRAJECTORY_REWARD_ENABLED": flag,
    }
    subprocess.run(["bash", "-c", script], env=env, check=True)
    text = calls.read_text()
    assert ("test_trajectory.py" in text) is expected
    assert "test_explore.py" in text
    assert "combine_rewards.py" in text
    assert "test_baseline.py" not in text


def test_disabled_trajectory_does_not_read_input(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAJECTORY_REWARD_ENABLED", "0")
    monkeypatch.setattr(trajectory, "TRAJECTORY_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(trajectory, "REPORT_PATH", tmp_path / "report.json")
    trajectory.main()
    assert not (tmp_path / "report.json").exists()


def test_enabled_trajectory_writes_reward(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAJECTORY_REWARD_ENABLED", "1")
    payload = tmp_path / "input.json"
    payload.write_text(
        json.dumps(
            {"schema_version": 1, "trajectories": {"ddt-agent": {"messages": []}}}
        )
    )
    monkeypatch.setattr(trajectory, "TRAJECTORY_PATH", payload)
    monkeypatch.setattr(trajectory, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(trajectory, "REWARD_PATH", tmp_path / "reward.txt")
    trajectory.main()
    assert json.loads((tmp_path / "report.json").read_text())["reward"] == 0


def test_disabled_component_cannot_affect_combined_reward(monkeypatch):
    monkeypatch.setenv("VLM_REWARD_ENABLED", "1")
    monkeypatch.setenv("TRAJECTORY_REWARD_ENABLED", "0")
    result = combine_rewards.build_rewards(
        {"status": "completed", "score": 0.75}, {"reward": 1}
    )
    assert result["reward"] == 0.75
    assert "trajectory_reward" not in result


def test_enabled_missing_trajectory_fails(monkeypatch):
    monkeypatch.setenv("VLM_REWARD_ENABLED", "1")
    monkeypatch.setenv("TRAJECTORY_REWARD_ENABLED", "1")
    with pytest.raises(ValueError):
        combine_rewards.build_rewards(
            {"status": "completed", "score": 0.75}, {"status": "missing"}
        )


def test_trace_retains_new_observations_when_old_table_is_removed(tmp_path):
    from test_explore import _trace_writer

    sink, close = _trace_writer(tmp_path / "trace.jsonl")
    system = {"role": "system", "content": "rules"}
    old_table = {"role": "user", "content": "Current verdicts: old"}
    assistant = {"role": "assistant", "content": None}
    observation = {"role": "tool", "tool_call_id": "c1", "content": "e0001"}
    new_table = {"role": "user", "content": "Current verdicts: new"}
    first = [system, old_table]
    second = [system, assistant, observation, new_table]
    try:
        for messages in (first, second, second):
            sink({"kind": "judge_request", "request": {"messages": messages}})
    finally:
        close()
    events = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert events[0]["messages"] == first
    assert events[1]["messages"] == [observation, new_table]
    assert events[2]["messages"] == []  # Retry must not duplicate evidence.


@pytest.mark.parametrize(
    "preview_url,readiness_error",
    [
        ("https://preview.example.test", "Preview readiness timed out"),
        ("", None),
        ("https://preview.example.test", None),
    ],
)
def test_preview_required_without_local_fallback(
    tmp_path, monkeypatch, preview_url, readiness_error
):
    from unittest.mock import AsyncMock, Mock
    import test_explore as explore
    import local_app

    monkeypatch.setenv("VLM_EXPLORE_ENABLED", "1")
    monkeypatch.setenv("VLM_API_KEY", "test-key")
    monkeypatch.setenv("APP_BASE_URL", preview_url)
    for name in ("APP_SPEC_PATH", "RUBRIC_PATH"):
        path = tmp_path / name
        path.write_text("{}")
        monkeypatch.setattr(explore, name, path)
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(explore, "REPORT_PATH", report_path)
    ready = Mock(return_value=readiness_error)
    probe = AsyncMock(return_value={"status": "completed", "score": 1.0})
    local = Mock(side_effect=AssertionError("Local fallback must not run"))
    monkeypatch.setattr(explore, "wait_for_app_ready", ready)
    monkeypatch.setattr(explore, "run_probe", probe)
    monkeypatch.setattr(local_app, "start_local_app", local)
    monkeypatch.setattr(explore, "start_dev_server", local)
    success = bool(preview_url) and readiness_error is None
    assert explore.main() == (0 if success else 1)
    report = json.loads(report_path.read_text())
    assert report["status"] == ("completed" if success else "app_unavailable")
    assert report["execution_source"] == "remote_preview"
    local.assert_not_called()
    if success:
        probe.assert_awaited_once()
    else:
        probe.assert_not_awaited()
        assert report["score"] is None
    if not preview_url:
        ready.assert_not_called()
