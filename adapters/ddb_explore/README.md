# DDB rubric-tree adapter

Create Harbor tasks from the rubric trees of
[rubrics-tree-generation](../../../rubrics-tree-generation). Each task has the
user's query as its instruction and the query's own rubric tree in
`tests/rubric.json`. A browser judge tests the built app against every leaf of
the tree. The builder never sees the rubric.
The verifier uses a rubric judge with shared browser helpers. Trajectory
scoring is optional and lives in `test_trajectory.py`; setting
`TRAJECTORY_REWARD_ENABLED=0` skips it entirely.

## Generate and run

From the Harbor repository root:

```bash
bash adapters/ddb_explore/generate_dataset.sh \
  --rubrics /mnt/workspace/liurenxi3/szm/rubrics-tree-generation/runs/dsv4-rand100-max-t1/rubrics.jsonl

bash adapters/ddb_explore/run_ddb_explore.sh
```

Tasks are written to `datasets/ddb-explore`. The script also writes the slime
prompt data to `datasets/ddb-explore.prompts.jsonl`: one row
`{"instruction": ..., "metadata": {"task_id": ...}}` per task, where `task_id`
is the task's directory name. Use `--limit 5` for a smaller dataset and
`--overwrite` to write the tasks again. The script uses `uv` when it is
installed, and `.venv/bin/python` when it is not.

On h20 and h20-2, Docker Hub is not reachable. Add `--registry-mirror
dockerproxy.net` so that the task Dockerfile pulls its base image through the
mirror, as the r2egym runtime datasets do. The image itself installs Node.js,
Playwright and Chromium with its system libraries.

The input is the `rubrics.jsonl` of a generator run. Each row has `id`,
`prompt`, `as_of`, `spec` and `tree`. Duplicate IDs, IDs that collide after
normalization, blank prompts and trees without the four branches are rejected.

The launcher uses the same DDB and VLM environment file as the comparison
adapter. It does not change existing jobs or environment files.
The browser judge uses `VLM_MODEL`, `VLM_API_BASE_URL`, and `VLM_API_KEY`.
The model must accept images and tool calls.

## Judge screen

`spec.delivery` selects the judge's screen for each task: `mobile_app` and
`unspecified` use HUAWEI Mate 80 Pro Max (440×949 CSS pixels, portrait, touch);
`single_page` uses `desktop` (1024×1000, no touch). Set `VLM_EXPLORE_DEVICE` to
use one screen for all tasks. An unset or empty value uses the task's screen.

## Reward switches

```bash
# Both enabled: reward = (vlm_reward + trajectory_reward) / 2
export VLM_REWARD_ENABLED=1
export TRAJECTORY_REWARD_ENABLED=1

# Rubric judge only
export VLM_REWARD_ENABLED=1
export TRAJECTORY_REWARD_ENABLED=0

# Trajectory only; optionally skip the browser judge too
export VLM_REWARD_ENABLED=0
export TRAJECTORY_REWARD_ENABLED=1
export VLM_EXPLORE_ENABLED=0
```

Both reward switches default to 1. Both set to 0 is invalid. Disabling a reward
component alone does not disable collection of its diagnostic report.
Missing enabled scores invalidate the evaluation; they do not become zero.

For GRPO through the slime bridge, use the `ddb` profile with
`--dataset-dir datasets/ddb-explore` on the runner, the prompt data above on
the trainer, and `harbor.slime_bridge.reward.harbor_verifier_reward`.

See the [verifier documentation](VERIFIER.md) for the scoring rules, evidence
artifacts, time limits, and current limitations.
