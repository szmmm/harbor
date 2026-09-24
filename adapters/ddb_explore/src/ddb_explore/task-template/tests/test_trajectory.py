#!/usr/bin/env python3
"""Use one selected DDB agent trajectory as the trial reward."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from trajectory_utils import DEFAULT_TRAJECTORY_SCORE_TYPE, get_trajectory_score

TRAJECTORY_PATH = Path("/tmp/harbor_trajectory.json")
REWARD_PATH = Path(
    os.environ.get("TRAJECTORY_REWARD_PATH", "/logs/verifier/reward.txt")
)
REPORT_PATH = Path("/logs/verifier/trajectory-score.json")
TARGET_AGENT = os.environ.get("DDB_TRAJECTORY_AGENT", "ddt-agent")
TRAJECTORY_SCORE_TYPE = os.environ.get(
    "TRAJECTORY_SCORE_TYPE", DEFAULT_TRAJECTORY_SCORE_TYPE
).strip().lower()
REWARD_CAP = 1000
_UUID_SUFFIX = re.compile(r"[0-9a-f]{32}$", re.IGNORECASE)


def _normalized_role(value: str) -> str:
    return re.sub(r"[-_]", "", _UUID_SUFFIX.sub("", value)).lower()


def select_trajectory(
    trajectories: dict[str, Any], target_agent: str
) -> tuple[str, list[dict[str, Any]]]:
    target_role = _normalized_role(target_agent)
    matches = []
    for agent_id, item in trajectories.items():
        if _normalized_role(agent_id) != target_role or not isinstance(item, dict):
            continue
        messages = item.get("messages")
        if isinstance(messages, list):
            matches.append((agent_id, messages))

    if len(matches) != 1:
        raise ValueError(
            f"expected one {target_agent!r} trajectory, found "
            f"{len(matches)}: {[agent_id for agent_id, _ in matches]}"
        )
    return matches[0]


def main() -> None:
    payload = json.loads(TRAJECTORY_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported trajectory schema")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, dict):
        raise ValueError("trajectory payload has no trajectories object")

    agent_id, messages = select_trajectory(trajectories, TARGET_AGENT)
    result = get_trajectory_score(
        messages,
        trajectory_score_type=TRAJECTORY_SCORE_TYPE,
        reward_cap=REWARD_CAP,
    )
    reward = float(result["score"])
    report = {
        "reward_mode": "trajectory",
        "selected_agent": TARGET_AGENT,
        "selected_agent_id": agent_id,
        "message_count": len(messages),
        **result,
        "reward": reward,
    }

    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(f"{reward:.4f}\n", encoding="utf-8")
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Trajectory reward: {reward:.4f} (agent={agent_id})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: trajectory scoring failed: {exc}", file=sys.stderr)
        raise
