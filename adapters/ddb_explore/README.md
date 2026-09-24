# DDB prompt-only evaluation adapter

Create Harbor tasks from arbitrary prompt JSONL. Apps are evaluated with the
browser judge and builder trajectory scorer included in `ddb_explore`.
No original benchmark rubrics or WebVoyager baseline are required.
The adapter has its own prompt record and dataset classes. It does not import
WebGen classes or depend on the WebGen adapter or Hugging Face datasets.
`webgen_explore` imports the base adapter from `ddb_explore.base` and reuses
the verifier files in `ddb_explore/task-template`.

## Generate and run

From the Harbor repository root:

```bash
bash adapters/ddb_explore/generate_dataset.sh \
  --jsonl /home/szmmatt/data/ddb-queries/variety100_02.jsonl

bash adapters/ddb_explore/run_ddb_explore.sh
```

Tasks are written to `datasets/ddb-explore`. The provided file contains
108 prompts. Use `--limit 5` for a smaller dataset.

The default prompt field is `query`. Use `--prompt-field original_query`,
`--prompt-field generalized_query`, or `--prompt-field instruction` for other
schemas. The selected prompt is passed unchanged to both builder and judge.
Blank prompts, duplicate IDs, and IDs that collide after normalization are
rejected. If `id` is absent, a stable ID is derived from the prompt content.
The complete source row is retained in each task's `tests/prompt_source.json`.

The launcher uses the same DDB and VLM environment file as the comparison
adapter. It does not change existing jobs or environment files.
The browser judge uses `VLM_MODEL`, `VLM_API_BASE_URL`, and `VLM_API_KEY`.
The `VLM_EXPLORE_*` settings control probe behavior, not its model or connection.

The judge defaults to HUAWEI Mate 80 Pro Max (440×949 CSS pixels, portrait).
Set `VLM_EXPLORE_DEVICE` to select another supported phone, or `desktop` for a
1024×1000 viewport with touch disabled. An unset or empty value uses the default.

## Reward switches

```bash
# Both enabled: reward = (vlm_reward + trajectory_reward) / 2
export VLM_REWARD_ENABLED=1
export TRAJECTORY_REWARD_ENABLED=1

# VLM only
export VLM_REWARD_ENABLED=1
export TRAJECTORY_REWARD_ENABLED=0

# Trajectory only; optionally skip the browser probe too
export VLM_REWARD_ENABLED=0
export TRAJECTORY_REWARD_ENABLED=1
export VLM_EXPLORE_ENABLED=0
```

Both reward switches default to 1. Both set to 0 is invalid. Disabling a reward
component alone does not disable collection of its diagnostic report.
Missing enabled scores invalidate the evaluation; they do not become zero.

For GRPO through the slime bridge, consume the final score with
`harbor.slime_bridge.reward.harbor_verifier_reward`.

See the [verifier documentation](VERIFIER.md) for
scoring anchors, evidence artifacts, time limits, and current limitations.
