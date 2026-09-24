# WebGen exploration and comparison adapter

Build apps from the original `luzimu/WebGen-Bench` dataset and evaluate each app
with three separate signals:

- The original dataset `ui_instruct` tasks, expected results, and WebVoyager evaluator.
- Our browser judge, which receives only the original instruction and running app.
- The DDB builder's trajectory score.

The original rubric score is a comparison metric. It never enters the training
reward. The new prompt-only adapter is in [../ddb_explore](../ddb_explore).
This adapter imports its base classes and verifier files from `ddb_explore`.

## Generate and run

From the Harbor repository root:

```bash
bash adapters/webgen_explore/generate_dataset.sh --split test
bash adapters/webgen_explore/run_webgen_explore.sh
```

The generated dataset is `datasets/webgen-explore`. Use `--limit 5` for a
small dataset. Use `--overwrite` only when replacing generated tasks is intended.

`--jsonl` accepts original WebGen-Bench rows with `id`, `instruction`, and
nonempty `ui_instruct`. Wrapped prompt-only training files do not contain the
rubrics and are rejected. Generation copies the original evaluator from the
sibling `webgen_bench` adapter in this repository.

The launcher reads `utils/harbor-rollout/ddb_rubrics_env.sh`, or the file selected
by `EXPLORE_ENV_FILE`. DDB requires `DDB_BASE_URL`, `DDB_API_KEY`,
and `DDB_PROXY_URL` or `PROXY_URL`. The judge and baseline require
`VLM_API_KEY`, `VLM_API_BASE_URL`, and `VLM_MODEL`.
Both verifiers share these model and connection settings.
The `VLM_EXPLORE_*` settings control probe behavior, not its model or connection.

## Reward switches

Set these environment variables before launching. Both default to 1.

| VLM_REWARD_ENABLED | TRAJECTORY_REWARD_ENABLED | Training reward |
| --- | --- | --- |
| 1 | 1 | Mean of VLM and trajectory scores |
| 1 | 0 | VLM score |
| 0 | 1 | Trajectory score |
| 0 | 0 | Configuration error |

Only literal `0` and `1` are accepted. Scores must be finite and in [0, 1].
A missing, failed, or incomplete enabled component produces no reward artifact.
It is not replaced by zero or by a score from a previous run.

`VLM_EXPLORE_ENABLED=0` skips browser evaluation. Use this with
`VLM_REWARD_ENABLED=0`; otherwise the enabled reward component is missing.
`WEBGEN_BASELINE_ENABLED=0` skips the original rubric evaluation.
Disabling a reward component alone does not prevent collecting it as a metric.

For the slime bridge, select
`--custom-rm-path harbor.slime_bridge.reward.harbor_verifier_reward` to consume
the combined `reward`. Selecting `harbor_trajectory_reward` bypasses the
verifier reward and uses the separate builder score.

## Judge contract

The judge reuses the multiturn agent's browser tools, but uses a new neutral
evaluation loop. It does not use the old follow-up-generation or survey prompts.

Version `app-outcomes-v1` uses four equally weighted dimensions:
prompt fidelity, functionality, robustness, and visual usability. Each dimension
is grounded in the exact input prompt. The dimensions and weights are fixed
before the app opens; their hash includes the prompt and is stored in the report.

The model selects concrete checks during exploration and reports their outcomes.
Ratings are 0 (confirmed failure), 1 (major gaps), 2 (partial), 3 (mostly works),
or 4 (fully meets the dimension). The VLM score is the mean rating divided by 4.
PASS requires 4, FAIL requires 0, and PARTIAL requires 1–3.

UNTESTED and VERIFIER_ERROR have no numeric rating. Any unresolved dimension
makes the app's VLM score unavailable. Coverage is the fraction of resolved
dimensions, not a claim that every possible behavior was tested.

Resolved verdicts must cite existing, successful observations. Positive
functional and robustness verdicts require interaction evidence; visual
verdicts require screenshots. The harness checks evidence references and tool
success, while the model remains responsible for whether the evidence proves
the claim. This does not make the judge a ground-truth oracle.

The judge defaults to a 1024×1000 desktop viewport, with touch disabled.
This matches the original WebVoyager baseline's configured window size.
Set `VLM_EXPLORE_DEVICE` to a supported phone name to override the judge, or
to `desktop` for the desktop setting. An unset or empty value uses the default.
The override does not change the original baseline's window setting.
The default browser budget is 24 model turns
(`VLM_EXPLORE_MAX_STEPS`), followed by up to two report turns.
The final two turns can correct reporting errors, but cannot obtain new evidence.

## Artifacts

All paths are relative to the trial's `verifier/` directory:

- `explore-report.json`: ratings, checks, evidence IDs, coverage, model,
  rubric hash, token counts, and elapsed time.
- `evidence/*.json` and `evidence/*.png`: tool results, accessibility snapshots,
  runtime signals, side effects, and screenshots.
- `explore.trace.jsonl`: flushed evidence events and final verdicts.
- `trajectory-score.json`: the builder's trajectory assessment.
- `webgen-report.json`, `category_stats.json`, `webvoyager_results/`:
  original rubric results and original evaluator artifacts.
- `evaluation-result.json`: all three reports and reward switches.
- `reward.json` and `reward.txt`: training reward, only when valid.

Example reward with both components enabled:

```json
{"reward": 0.625, "vlm_reward": 0.75, "trajectory_reward": 0.5, "webgen_reward": 0.8}
```

Missing diagnostic scores are omitted instead of represented by a numeric sentinel.

## Comparison limits

The original evaluator's tasks, expected results, 15-action budget, and score
parser are copied unchanged. Our judge has a different tool interface and budget.
This compares complete verifier strategies, not only rubric wording.

Our judge runs first. The original evaluator then starts fresh browser sessions.
A shared remote backend is not reset. For comparisons where either evaluator
changes shared data, supply a separately restored copy through
`WEBGEN_BASELINE_URL`. Without that, use the comparison as exploratory evidence.

Readiness, judge, and baseline execution have separate time bounds. Probe failure
cannot consume the entire verifier budget. API and readiness failures are
measurement failures, not confirmed app-quality failures.

This version produces per-app scores. It does not yet implement pairwise scoring,
learned preference calibration, or prompt-specific rubric compilation. The
fixed dimensions are a first baseline for the human-calibration study.
