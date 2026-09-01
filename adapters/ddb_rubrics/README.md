# DDB Rubrics Adapter

This adapter converts `luzimu/WebGen-Bench` records into Harbor tasks for DDB
agent rollout training. It supports trajectory reward, VLM reward, and both
rewards together.

## Requirements

- Harbor with the staged `ddb.py` trajectory publication change.
- A reachable DDB service and rollout proxy.
- Docker.
- Hugging Face access, or a local WebGen-Bench JSONL file.
- A VLM API key when using `vlm` or `both` mode.

The staged DDB change publishes complete trajectories to:

```text
/tmp/harbor_trajectory.json
```

## Generate the dataset

Run from the Harbor repository root after placing this adapter under
`adapters/ddb_rubrics`:

```bash
cd adapters/ddb_rubrics
uv sync

uv run ddb-rubrics \
  --output-dir ../../datasets/ddb-rubrics
```

Generate a small subset:

```bash
uv run ddb-rubrics \
  --output-dir ../../datasets/ddb-rubrics \
  --limit 10
```

Use a local JSONL file:

```bash
uv run ddb-rubrics \
  --output-dir ../../datasets/ddb-rubrics \
  --jsonl /path/to/test.jsonl
```

## Run DDB

Set the DDB connection variables:

```bash
export DDB_BASE_URL="http://ddb-host:8000"
export DDB_API_KEY="your-key"
export DDB_PROXY_URL="http://proxy-host:9400"
```

Start the job from the Harbor repository root:

```bash
uv run harbor job start \
  -c adapters/ddb_rubrics/run_ddb_rubrics_ddb.yaml
```

The job configuration uses `training: true`. DDB records token IDs, masks,
log probabilities, trajectories, trajectory scores, finish reasons, and weight
versions in agent metadata.

## Reward modes

`VLM_STATIC_EVAL` selects one of three modes:

```text
trajectory
vlm
both
```

The default is `trajectory`.

Trajectory mode:

```bash
export VLM_STATIC_EVAL=trajectory
```

VLM mode:

```bash
export VLM_STATIC_EVAL=vlm
export VLM_API_KEY="your-vlm-key"
export VLM_API_BASE_URL="https://example.com/v1"
export VLM_MODEL="Qwen/Qwen2.5-VL-32B-Instruct"
```

Both modes together:

```bash
export VLM_STATIC_EVAL=both
```

The canonical reward rules are:

```text
trajectory: reward = trajectory_reward
vlm:        reward = vlm_reward
both:       reward = trajectory_reward * vlm_reward
```

## Result format

The verifier writes `/logs/verifier/reward.json`. Harbor stores it under
`verifier_result.rewards` in `result.json`:

```json
{
  "reward": 798.4,
  "vlm_reward": 0.8,
  "trajectory_reward": 998.0
}
```

An inactive component is `-1`. Harbor uses the canonical `reward` field as the
trial training reward. The two component fields remain available for inspection
and metrics.

## Trajectory selection

The default selected role is:

```text
ddt-agent
```

Override it with:

```bash
export DDB_TRAJECTORY_AGENT="another-agent-role"
```

Exactly one matching trajectory must exist in the DDB metadata.

## VLM result aggregation

`compute_acc.py` aggregates WebVoyager VLM results from a DDB job:

```bash
python adapters/ddb_rubrics/compute_acc.py \
  --job-dir jobs/ddb-rubrics-ddb \
  --datasets-dir datasets/ddb-rubrics
```
