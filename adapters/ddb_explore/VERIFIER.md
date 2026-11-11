# DDB app verifier

`ddb_explore` owns the task writer, rubric-tree judge, trajectory scorer,
reward combiner, Docker template, and DDB launch script.
The current verifier uses `RubricJudge`; `probe.py` contains only its shared
browser tools and configuration. Legacy findings/contract judges and WebGen
baseline evaluation are not included.

## Rubric

Each task carries its rubric tree in `tests/rubric.json`, unchanged from the
generator's `rubrics.jsonl`. The tree has four branches (intention, content,
static, dynamic). Each leaf is one yes/no check with `critical`, `verify_by`
(`view`, `interaction` or `fact`) and an optional `note` for the judge. The
generator wrote the tree from the query before any app was built.

## Judge

`judge.py` drives the app in a headless browser through the probe's tools
(accessibility snapshot, click, type, swipe, long press, upload, scroll, reload,
page text, text location, screenshot). It gets the query, the user's today
(`spec.time_frame.date`), the evaluation date, and the leaves. It tests
critical leaves first.

The judge has more tools for leaves that the screen alone cannot decide:

- `set_viewport`: window size and full-screen checks.
- `set_media`: print CSS, for print checks.
- `clear_site_data`: delete cookies and browser storage, then reload. It acts
  as a new browser or device, for checks that data is stored on a server.
- `list_source_files`, `read_source_file`, `search_source`: read-only access
  to the builder's workspace in `/app/workspace`, which the DDB agent uploads
  without installed dependencies, caches and platform skill/catalog links.
  Application manifests, source, lockfiles, assets, logs and built `dist` output
  are retained. They are for checks about the
  framework, external scripts, single-file delivery, and where data is stored.
  In the 100 rubrics of `dsv4-rand100-max-t1`, 44 critical leaves need the
  source or a new session.

The judge records verdicts with `record_verdicts` while it works: PASS, FAIL or
UNKNOWN, a reason, and the evidence IDs that support it. A later verdict for a
leaf replaces the earlier one. Code rejects a verdict that cites an unknown or
failed observation, a PASS or FAIL with no evidence, and a PASS on an
`interaction` leaf without an observation after an action. The model must
still decide whether the evidence supports its verdict. Human calibration
remains necessary.

Only the 4 most recent observations keep their snapshot text in the model
context, and only the 3 most recent screenshots. Older ones keep their IDs.
All of them are saved under `evidence/`.

`finish` ends the evaluation immediately, with no finalization afterward.
Critical failures do not stop exploration. When the turn limit ends first,
the judge gets a forced verdict call and one repair call within the same
overall deadline.

## Scoring

`rubric_score.py` uses `rubric-tree-v2`: critical leaves have weight 2 and
other leaves weight 1. Reward is PASS weight divided by the weight of **all**
rubric leaves. Fully assessed scores are unchanged. Unassessed leaves earn no
credit but retain UNKNOWN (UNTESTED in generated tasks), rather than being
rewritten to FAIL. Coverage, branch pass rates and unresolved critical IDs remain
diagnostic; partial assessments have `scoring_status: partial`.

A normally completed judge evaluation with at least one accepted PASS/FAIL can
produce a reward even when critical checks remain unresolved. An assessment
with no usable verdicts, a judge/API error, or an exhausted deadline still has
no reward. This deliberately prioritizes usable conservative training rewards:
a preview outage may lower the score, and a few source-only passes cannot inflate
an otherwise untested app to a perfect score.

`reward.json` has `reward`, the enabled components (`vlm_reward`,
`trajectory_reward`), and diagnostics that the slime bridge logs but does not
train on: `vlm_coverage`, `vlm_pass_rate` (unweighted),
`vlm_critical_failed`, and `vlm_branch_<branch>` pass rates.

## Budget

| Variable | Default | Meaning |
|---|---|---|
| `VLM_EXPLORE_MAX_STEPS` | 40 | Model turns with browser tools. A turn can call several tools. |
| `VLM_EXPLORE_DEADLINE_S` | 1500 | Wall time after which the judge stops testing and records its verdicts. |
| `VLM_REQUEST_TIMEOUT_S` | 120 | Timeout of one model request. |

`test.sh` stops the judge after 1800 s. `verifier.timeout_sec` is 3600 s.

Screenshots and tool results are saved under `evidence/`. Verdicts, scores and
per-leaf results are saved in `explore-report.json`. The event trace is
`explore.trace.jsonl`.

The trace logs the initial context once, then only newly appended incoming
messages at each `judge_request`; judge outputs remain in `judge_response`.
Screenshots in logged messages reference `evidence/<id>.png` instead of inline
image data. These are readable turn logs, not full request snapshots.

`VLM_REWARD_ENABLED` and `TRAJECTORY_REWARD_ENABLED` accept 0 or 1. Both default
to 1, which uses the mean of the two scores. One enabled component uses that
score alone. Both disabled is invalid. Missing enabled scores invalidate the
evaluation; they are not replaced with zero. Combined scores are written to
`reward.json` and `reward.txt`, with diagnostics in `evaluation-result.json`.
Trajectory scoring and its helpers are contained in `test_trajectory.py`.
When `TRAJECTORY_REWARD_ENABLED=0`, `test.sh` does not launch that process,
and the combiner ignores trajectory scores. Direct invocation of the Python
entry point with that flag also exits without reading or scoring a trajectory.

## Limitations

- Single-app scoring only. Each trial is judged alone; there is no pairwise or
  per-leaf A/B comparison inside a GRPO group yet.
- `fact` leaves rely on the judge model's own knowledge. It has no web search.
- The judge uses temperature 1.0 by default (`VLM_TEMPERATURE`), matching the
  temperature used in GLM-5.3-Flash's published multimodal and tool-use evaluations
  ([model card](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes)).
  `VLM_OMIT_TEMPERATURE=1` instead uses the server default. Each trial uses one
  judge run; verdict noise is not measured yet.

The judge preserves returned `reasoning` and/or `reasoning_content` fields
verbatim under their original names in subsequent requests. The deployed vLLM
endpoint consumes `reasoning`; its tokenizer ignores `reasoning_content`. It can
use `read_evidence(evidence_id)` to retrieve saved text and screenshots without
changing the browser, and `return_to_app()` to reopen the original preview URL.
Retrieved screenshots join the same three-image context window.

`click_at(x, y)` uses viewport CSS-pixel coordinates for controls absent from
the accessibility snapshot. `open_app(mode)` opens the original preview in a
`new_tab`, closes/reopens the current tab (`reopen_tab`), or replaces the browser
context with an isolated profile (`fresh_context`). The first two retain cookies,
localStorage and IndexedDB but start empty sessionStorage. A fresh context may
represent a different anonymous user; clearing storage is not a persistence test.
Each turn receives a current table of all leaf verdicts and their evidence IDs,
including PENDING and UNKNOWN. The latest table supersedes earlier tables.


### Judge observation transport

Only an explicit `take_screenshot()` captures a screenshot. Opening the app,
clicking, scrolling, waiting, reloading, and reading page text do not capture
images. Browser actions still return fresh accessibility snapshots, action
outcomes, runtime signals, side effects, and evidence IDs. The initial app-open
snapshot is bootstrap task context; subsequent observations are tool results.

Each tool result is attached to its originating `tool_call_id`, including source
file listings/reads/searches and historical evidence retrieval. Screenshots are
image content blocks in the `take_screenshot` tool response, alongside text
metadata. No separate user-role message is added for these tool observations.
Historical evidence only includes an image if one was originally captured.
The latest three attached images remain in model context; older images remain
available through `read_evidence`.

Task instructions/rubric and harness budget/verdict reminders remain separate
from tool evidence. Reasoning effort, scoring, and exploration limits are not
changed by this transport update.


### Judge termination and finalization

`finish` ends evaluation immediately, regardless of the recorded verdicts. No
further tools or model requests execute after it. The scorer then evaluates
the recorded verdicts; an unusable score fails the evaluation run.

Only exhaustion of the exploration turn limit triggers forced finalization:
one `record_verdicts` round for pending/UNKNOWN leaves, with one repair round
for missing or rejected entries. Finalization uses existing evidence only and
caps each response at 16,384 tokens (or the lower configured output limit).
UNKNOWN is valid when evidence remains insufficient. Accepted PASS/FAIL leaves
are not automatically reconsidered. Critical FAIL never forces termination.

Exploration and finalization share the original overall deadline. There is no
reserved time split and no deadline extension. Request retries remain bounded;
an exhausted exploration request timeout fails evaluation. If finalization
cannot complete within the shared deadline, or the scorer cannot produce a
usable score, the evaluation fails with no reward. Evidence and recorded
verdicts remain available for diagnosis.

The preview readiness check remains bounded by `APP_READY_TIMEOUT` (default
120 seconds). A failed check or missing `APP_BASE_URL` is passed to the existing
rubric judge as `preview_readiness_error`, together with a not-ready note. The
judge retains the same rubric, browser/source tools, budgets and scoring rules.
Initial navigation failures and subsequent failed browser retries are saved as
failed observations, so the judge can inspect sources or retry the preview.
There is no separate judge, automatic zero, or local workspace startup fallback.
A normal evidence-supported zero remains a valid reward. Unresolved critical
checks no longer suppress partial rewards; judge failures still produce no reward. Rollout-only runs with the
verifier disabled remain unaffected.

### Completed-task handoff and local source

DDB's API lifecycle must reach `status: completed` before grading. This is
separate from the actor's self-reported `task_done.completion_status`, which
the headless API does not normally expose. Build success and preview readiness
are diagnostic outcomes, not admission gates. Failed/cancelled/running
executions still do not enter grading.

Harbor writes a pending handoff before execution. It first tries DDB's existing
saved workspace download, preserving original paths rather than guessing that a
single subdirectory is an archive wrapper. A damaged/unavailable archive falls
back to the existing file-list/read APIs. The per-file path retries every
advertised file and fails if any of those downloads remain incomplete. It also
requests `.deepdiver`, `dist`, `build` and `out` subtrees explicitly, since
the normal UI listing may hide them, and probes the artifact declaration when
it was not listed. Unavailable optional probes are recorded without claiming
that the actor failed to create those files. Read responses must match the
inventory's live/snapshot source label; stale/degraded reads are rejected.

Only after the downloads and sandbox upload finish does verifier code check
the local bytes against Harbor's receipt. These are automatic Python checks,
not judge-model calls. They verify upload integrity, not remote inventory
completeness or freshness. The existing API supplies no pinned revision or
remote hashes; saved snapshots may be older, omit dist/runtime evidence, and
listings can filter directories or cap depth. The handoff explicitly records
these limitations and sets `inventory_complete: false`. A missing file in this
local copy therefore is not sufficient evidence that the actor omitted it.

The same rubric judge receives execution, actor claims, build outcome, preview
readiness and source-transfer diagnostics. The judge-facing build context renames
`success` to `platform_reported_success` and explains the mode: `live` confirms
only a runtime/session, and `none` means the build was skipped. Neither verifies
compilation or HTTP readiness. Raw handoff values remain in the audit report.
Initial readiness errors have unknown attribution, not an infrastructure label.
The canonical and generated prompts use the same verdict policy (apart from
UNKNOWN/UNTESTED terminology): proven blockers fail dependent requirements;
missing observations or evaluator capabilities remain unassessed. Failed tool
observations can support FAIL, but do not automatically establish app fault;
PASS still requires successful observations and actual interaction evidence. It can page through file listings,
read source in chunks and search. Positive code evidence can support a verdict
for a failed app, while uncertain omissions remain UNKNOWN/UNTESTED. Source
cannot prove visual or interactive success. The prompt uses general evidence
rules rather than a separate failure workflow or delivery-specific verdict recipes.
It asks the judge to stop repeating recovery attempts that reveal no new
information, assess the available evidence, and leave time to record verdicts.
This is model guidance, not a fixed retry count or a hard tool budget. When readiness fails, available
root `package.json` and `.deepdiver/artifact.json` are supplied as cited source
observations before the first judge call, so nested app manifests do not obscure
the actual startup configuration. Uncertain infrastructure failures
may still remain UNKNOWN/UNTESTED. Configured budgets are unchanged; partial scoring follows the v2 rule above.

No dependency installation is needed to read source. Harbor retains app source,
artifact declarations, package manifests, lockfiles, vendored source, assets,
build outputs and preview logs whenever they are returned. Installed dependency
directories, caches, Git history, tool-call logs and platform skill/catalog links
are excluded. DDB's current snapshot producer itself excludes the normal root
`node_modules/`, `dist/` and `.deepdiver/preview/` before packaging. Harbor
cannot recover files that an upstream saved archive no longer contains.

The judge only reads the local copy; it has no new tools for the actor sandbox.
No DDB client/server changes or custom export endpoints are required. The
previous proposed DDB export API was reverted. Existing processes are unchanged
until Harbor is reloaded, and rollout-only runs remain verification-disabled.

The judge can inspect popup tabs explicitly: `list_tabs` returns stable tab IDs,
titles, active status and URLs without credentials/query/fragment; `switch_tab`
selects a live tab in the current browser profile without navigating or clearing
storage. Popup events include their tab ID but do not automatically change focus.
Switching refreshes element references; screenshots and actions then target the
selected page. Closed tabs and tabs from an old profile cannot be selected.
Tab selection itself is not interaction evidence for a feature PASS. These tools
do not change retry budgets, scoring, or the immediate meaning of `finish`.
