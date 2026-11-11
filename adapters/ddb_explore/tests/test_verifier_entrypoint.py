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
    "url,readiness_error",
    [("http://preview", "HTTP 202: not ready"), ("", None), ("http://preview", None)],
)
def test_readiness_reaches_normal_probe_without_local_fallback(
    tmp_path, monkeypatch, url, readiness_error
):
    import test_explore as explore

    monkeypatch.setenv("VLM_API_KEY", "test")
    monkeypatch.setenv("APP_BASE_URL", url)
    monkeypatch.setenv("VLM_EXPLORE_ENABLED", "1")
    spec = tmp_path / "app_spec.json"
    rubric = tmp_path / "rubric.json"
    spec.write_text('{"default_device":"desktop"}')
    rubric.write_text('{"prompt":"Build an app"}')
    monkeypatch.setattr(explore, "APP_SPEC_PATH", spec)
    monkeypatch.setattr(explore, "RUBRIC_PATH", rubric)
    monkeypatch.setattr(explore, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(explore, "wait_for_app_ready", lambda *args: readiness_error)
    monkeypatch.setattr(
        explore, "start_dev_server", lambda: pytest.fail("Local fallback must not run")
    )
    calls = []

    async def run_probe(app_url, app_spec, rubric, *, preview_error=None, handoff=None):
        calls.append((app_url, preview_error))
        return {"status": "completed", "score": 0}

    monkeypatch.setattr(explore, "run_probe", run_probe)
    assert explore.main() == 0
    assert len(calls) == 1 and calls[0][0] == url
    assert bool(calls[0][1]) is bool(not url or readiness_error)
    assert (
        json.loads(explore.REPORT_PATH.read_text())["preview_readiness_error"]
        == calls[0][1]
    )
