"""Combine enabled training rewards; keep the benchmark baseline separate."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

REWARD_DIR = Path("/logs/verifier")


def enabled(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def read_report(name: str) -> dict:
    path = REWARD_DIR / name
    if not path.is_file():
        return {"status": "missing", "score": None}
    try:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError("Report must be an object")
        return result
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "score": None, "error": str(exc)}


def valid_score(value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"Expected finite score in [0, 1], got {value!r}")
    return float(value)


def rubric_diagnostics(probe: dict) -> dict[str, float]:
    """Numeric rubric results for monitoring. The bridge logs them; only "reward" trains."""
    out = {}
    for key in ("coverage", "pass_rate"):
        value = probe.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"vlm_{key}"] = float(value)
    if probe.get("status") in {"completed", "incomplete"}:
        out["vlm_critical_failed"] = float(bool(probe.get("failed_critical")))
    for branch, value in (probe.get("branches") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"vlm_branch_{branch}"] = float(value)
    return out


def build_rewards(probe: dict, trajectory: dict) -> dict[str, float]:
    selected = []
    rewards = {}
    for name, report, value in (
        ("vlm", probe, probe.get("score")),
        ("trajectory", trajectory, trajectory.get("reward")),
    ):
        active = enabled(f"{name.upper()}_REWARD_ENABLED")
        if not active:
            continue
        try:
            if report.get("status") not in {None, "completed"}:
                raise ValueError(f"{name} evaluation status: {report.get('status')}")
            score = valid_score(value)
        except ValueError:
            if active:
                raise
        else:
            rewards[f"{name}_reward"] = score
            if active:
                selected.append(score)
    if not selected:
        raise ValueError("Enable VLM_REWARD_ENABLED or TRAJECTORY_REWARD_ENABLED")
    rewards["reward"] = sum(selected) / len(selected)
    rewards.update(rubric_diagnostics(probe))
    return rewards


def main() -> None:
    for name in ("reward.json", "reward.txt"):
        (REWARD_DIR / name).unlink(missing_ok=True)
    probe = read_report("explore-report.json")
    trajectory = (
        read_report("trajectory-score.json")
        if enabled("TRAJECTORY_REWARD_ENABLED")
        else {"status": "disabled"}
    )
    result = {
        "schema_version": 2,
        "reward_mode": "enabled_component_mean",
        "probe": probe,
        "trajectory": trajectory,
        "reward_switches": {
            key: os.environ.get(key, "1")
            for key in ("VLM_REWARD_ENABLED", "TRAJECTORY_REWARD_ENABLED")
        },
    }
    try:
        rewards = build_rewards(probe, trajectory)
    except ValueError as exc:
        result.update(status="evaluator_failed", error=str(exc), rewards=None)
        (REWARD_DIR / "evaluation-result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        raise
    result.update(status="completed", rewards=rewards)
    (REWARD_DIR / "evaluation-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (REWARD_DIR / "reward.json").write_text(json.dumps(rewards, indent=2) + "\n")
    (REWARD_DIR / "reward.txt").write_text(f"{rewards['reward']:.8f}\n")


if __name__ == "__main__":
    main()
