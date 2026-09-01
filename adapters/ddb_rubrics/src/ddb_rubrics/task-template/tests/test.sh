#!/bin/bash
#
# VLM_STATIC_EVAL selects: trajectory, vlm, or both (default: trajectory).
# DDB_TRAJECTORY_AGENT selects the trajectory role (default: ddt-agent).
# VLM modes also require VLM_API_KEY and optional VLM_API_BASE_URL/VLM_MODEL.

set -uo pipefail

reward_dir=/logs/verifier
vlm_reward_path="$reward_dir/vlm-reward.txt"
trajectory_reward_path="$reward_dir/trajectory-reward.txt"
reward_mode="${VLM_STATIC_EVAL:-trajectory}"

mkdir -p "$reward_dir"
rm -f \
    "$reward_dir/reward.txt" \
    "$reward_dir/reward.json" \
    "$vlm_reward_path" \
    "$trajectory_reward_path"

echo "=== DDB Rubrics Test Runner (mode: $reward_mode) ==="

# Use the preview URL published by DDB. A manual APP_BASE_URL wins.
if [ -z "${APP_BASE_URL:-}" ] && [ -f /tmp/harbor_preview_url ]; then
    APP_BASE_URL="$(cat /tmp/harbor_preview_url)"
    export APP_BASE_URL
    echo "=== Using agent-published preview URL: $APP_BASE_URL ==="
fi

run_vlm() {
    VLM_REWARD_PATH="$vlm_reward_path" python3 -u /tests/test_ui.py 2>&1
    local code=$?
    if [ ! -s "$vlm_reward_path" ]; then
        echo "WARNING: VLM reward was not written; defaulting to 0"
        echo 0 > "$vlm_reward_path"
    fi
    return "$code"
}

run_trajectory() {
    TRAJECTORY_REWARD_PATH="$trajectory_reward_path" \
        python3 -u /tests/test_trajectory.py 2>&1
    local code=$?
    if [ ! -s "$trajectory_reward_path" ]; then
        echo "WARNING: trajectory reward was not written; defaulting to 0"
        echo 0 > "$trajectory_reward_path"
    fi
    return "$code"
}

set +e
case "$reward_mode" in
    trajectory)
        run_trajectory
        exit_code=$?
        ;;
    vlm)
        run_vlm
        exit_code=$?
        ;;
    both)
        run_trajectory
        trajectory_exit_code=$?
        run_vlm
        vlm_exit_code=$?
        if [ "$trajectory_exit_code" -ne 0 ]; then
            exit_code=$trajectory_exit_code
        else
            exit_code=$vlm_exit_code
        fi
        ;;
    *)
        echo "ERROR: VLM_STATIC_EVAL must be trajectory, vlm, or both"
        exit_code=2
        ;;
esac

VLM_STATIC_EVAL="$reward_mode" python3 -u /tests/combine_rewards.py 2>&1
combine_exit_code=$?
set -e

if [ "$combine_exit_code" -ne 0 ] || [ ! -s "$reward_dir/reward.json" ]; then
    echo "WARNING: reward aggregation failed; writing zero reward"
    echo '{"reward": 0, "vlm_reward": -1, "trajectory_reward": -1}' \
        > "$reward_dir/reward.json"
fi

echo "=== Test runner exit code: $exit_code ==="
echo "=== Rewards: $(cat "$reward_dir/reward.json") ==="
exit 0
