# DDB app verifier

`ddb_explore` owns the dataset-neutral task writer, browser judge, trajectory
scorer, reward combiner, Docker template, and DDB launch script.
`webgen_explore` imports the base classes from `ddb_explore.base` and reuses
these verifier files. `ddb_explore` does not depend on WebGen.

Each adapter supplies its own record and dataset classes. WebGen adds the
original benchmark instructions, rubrics, and baseline evaluator. DDB reads
prompt JSONL and preserves the selected prompt unchanged.

## Screen settings

The adapter writes its default device into each task's `app_spec.json` and
verifier configuration:

- WebGen: `desktop`, 1024×1000 CSS pixels, touch disabled.
- DDB: `HUAWEI Mate 80 Pro Max`, 440×949 CSS pixels, touch enabled.

`VLM_EXPLORE_DEVICE` overrides either default. An unset or empty value uses
the task's default. The selected device is saved in `explore-report.json`.

## Scoring and evidence

The `app-outcomes-v1` rubric has four equally weighted dimensions: prompt
fidelity, functionality, robustness, and visual usability. Each rating is 0–4;
the VLM score is their mean divided by 4. The original prompt defines what the
app must do. The judge must cite saved observations for its verdicts.

UNTESTED and VERIFIER_ERROR have no numeric rating. An unresolved dimension
makes the VLM score unavailable. Evidence validation checks references, tool
success, screenshots, and interaction evidence. The model must still decide
whether the evidence supports its verdict. Human calibration remains necessary.

The default budget is 24 browser model turns plus two report-only turns.
Set `VLM_EXPLORE_MAX_STEPS` to change the browser budget. Screenshots and tool
results are saved under `evidence/`; verdicts and checks are saved in
`explore-report.json`; the event trace is `explore.trace.jsonl`.

`VLM_REWARD_ENABLED` and `TRAJECTORY_REWARD_ENABLED` accept 0 or 1. Both default
to 1, which uses the mean of the two scores. One enabled component uses that
score alone. Both disabled is invalid. Missing enabled scores invalidate the
evaluation; they are not replaced with zero. Combined scores are written to
`reward.json` and `reward.txt`, with diagnostics in `evaluation-result.json`.
The original WebGen baseline score is diagnostic only.

This version does not implement pairwise scoring, learned preference
calibration, or prompt-specific rubric compilation.
