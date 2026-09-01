#!/usr/bin/env python3
"""Combine VLM and trajectory component scores into Harbor reward.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

REWARD_DIR = Path("/logs/verifier")
REWARD_JSON_PATH = REWARD_DIR / "reward.json"
VLM_REWARD_PATH = REWARD_DIR / "vlm-reward.txt"
TRAJECTORY_REWARD_PATH = REWARD_DIR / "trajectory-reward.txt"
VALID_MODES = {"trajectory", "vlm", "both"}


def _read_score(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").strip())


def build_rewards(mode: str) -> dict[str, float]:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported VLM_STATIC_EVAL mode: {mode}")

    vlm_reward = _read_score(VLM_REWARD_PATH) if mode in {"vlm", "both"} else -1.0
    trajectory_reward = (
        _read_score(TRAJECTORY_REWARD_PATH)
        if mode in {"trajectory", "both"}
        else -1.0
    )

    if mode == "vlm":
        reward = vlm_reward
    elif mode == "trajectory":
        reward = trajectory_reward
    else:
        reward = vlm_reward * trajectory_reward

    return {
        "reward": round(reward, 4),
        "vlm_reward": round(vlm_reward, 4),
        "trajectory_reward": round(trajectory_reward, 4),
    }


def main() -> None:
    mode = os.environ.get("VLM_STATIC_EVAL", "trajectory").strip().lower()
    rewards = build_rewards(mode)
    REWARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON_PATH.write_text(
        json.dumps(rewards, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rewards, sort_keys=True))


if __name__ == "__main__":
    main()
