#!/bin/bash
# Each enabled reward component is required. The original rubric is diagnostic.
set -uo pipefail
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json \
    /logs/verifier/trajectory-reward.txt /logs/verifier/trajectory-score.json \
    /logs/verifier/explore-report.json /logs/verifier/explore.trace.jsonl \
    /logs/verifier/webgen-report.json /logs/verifier/evaluation-result.json

if [ -z "${APP_BASE_URL:-}" ] && [ -f /tmp/harbor_preview_url ]; then
    APP_BASE_URL="$(cat /tmp/harbor_preview_url)"
    export APP_BASE_URL
fi

TRAJECTORY_REWARD_PATH=/logs/verifier/trajectory-reward.txt \
    timeout 60 python3 -u /tests/test_trajectory.py || true

# Bound sidecars independently, leaving time to publish the final result.
timeout 900 python3 -u /tests/test_explore.py || true

if [ -f /tests/ui_instruct.json ] && [ "${WEBGEN_BASELINE_ENABLED:-1}" = "1" ]; then
    timeout 1800 python3 -u /tests/test_baseline.py || true
fi

# Missing evidence must not become a zero-valued training example.
python3 -u /tests/combine_rewards.py
