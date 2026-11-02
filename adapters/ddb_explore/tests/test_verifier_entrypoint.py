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
