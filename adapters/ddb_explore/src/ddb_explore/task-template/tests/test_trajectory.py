#!/usr/bin/env python3
"""Use one selected DDB agent trajectory as the trial reward."""

from __future__ import annotations

import json
import math
import os
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

################ DDB Trajectory Utilities ################
TRAJECTORY_SCORE_TYPES = ["tool_call_success_rate", "validation", "validation_rubric"]
DEFAULT_TRAJECTORY_SCORE_TYPE = "validation_rubric"
if DEFAULT_TRAJECTORY_SCORE_TYPE not in TRAJECTORY_SCORE_TYPES:
    raise ValueError(
        f"Default trajectory score type {DEFAULT_TRAJECTORY_SCORE_TYPE} not in {TRAJECTORY_SCORE_TYPES}"
    )


# --- validation_rubric --------------------------------------------------------
# Each validator becomes a series of error counts for one scope. A series records
# whether the count reached zero, how much it regressed, and how many model
# actions recovery required.
#
# Scope is a closed vocabulary, not a key made from file or route names. Exact
# targets would split related work into single-reading series.
#
# Each measured scope is scored. A later, wider clean run can close a narrow
# dirty scope, but it cannot restore the efficiency that the narrow series did
# not show.
#
# This rule is necessary for runtime validation. Its scope records what the
# probe reached, so a failure can prevent the probe from reaching a wider scope.

# Blend weight for successful non-validation tool calls. At zero, the rate is
# reported in metadata but does not change the reward.
TOOL_SUCCESS_WEIGHT = 0.0
# Maximum share a family can claim when it never reached its full scope.
NON_FULL_SCOPE_COEFFICIENT = 0.25
# Discount for ending on source edits no validator has seen. Set to 1 to bypass;
# it is off until there is corpus evidence for a specific value, because a
# staleness term is only as good as the write-tool list behind it.
STALE_MULTIPLIER = 1.0

# Every validator DDB exposes, including any the rubric does not read.
VALIDATION_TOOL_NAMES = frozenset(
    {
        "static_validation",
        "build_validation",
        "runtime_validation",
        "test_validation",
    }
)

# The scope in each family that leaves nothing unmeasured. Everything else is
# narrow and claims NON_FULL_SCOPE_COEFFICIENT of what it earns.
FULL_SCOPES = frozenset({"static.project", "build.typechecked", "runtime.complete"})

# How much of its family each scope saw. Ranks are comparable only within one
# family and one project, and only strict `>` closes a series: two scopes that
# saw different slices of the same project (a file scan and a tripped integrity
# gate) vouch for nothing. Keep the top rank of each family in FULL_SCOPES.
SCOPE_COVERAGE = {
    "static.files": 0,
    "static.integrity_gate": 0,
    "static.partial": 1,
    "static.project": 2,
    "build.bundle_only": 0,
    "build.typechecked": 1,
    "runtime.load_only": 0,
    "runtime.interactive": 1,
    "runtime.complete": 2,
}

# Calls that unambiguously write source. `bash` is absent because agents use it
# for both writes and reads, and its arguments do not reliably distinguish them.
WORKSPACE_WRITE_TOOLS = frozenset(
    {
        "file_write",
        "file_edit",
        "file_insert",
        "file_delete",
        "file_multi_edit",
        "vite_init",
        "delegate_task",
    }
)
# Bookkeeping files (agent TODO lists, notes) that are not the application.
NON_SOURCE_SUFFIXES = (".md", ".txt")


@dataclass(frozen=True)
class ToolResult:
    """One tool result, joined back to the model action that asked for it.

    ``action`` is that assistant turn's ordinal. Differencing it between two
    validations is the only measure of effort the rubric uses.
    """

    name: str
    result: dict[str, Any]
    arguments: dict[str, Any]
    action: int


@dataclass(frozen=True)
class Reading:
    """One validation result, reduced to what scoring needs."""

    action: int
    errors: int
    scope: str
    project: str


def _dict(value: Any) -> dict[str, Any]:
    """Parse a field that arrives as a JSON string, an object, or neither."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _flag(value: Any) -> bool:
    """Truth of a tool argument.

    Some trajectories encode booleans as the strings ``"True"`` and ``"False"``.
    Plain truthiness reads ``"False"`` as true and mislabels the scope.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _count(value: Any) -> int:
    """A non-negative error count from a metadata field, defaulting to 0."""
    return (
        max(int(value), 0)
        if isinstance(value, int) and not isinstance(value, bool)
        else 0
    )


def _iter_tool_results(trajectory: list[dict[str, Any]]) -> list[ToolResult]:
    """Ordered tool results, each stamped with the action that requested it.

    A result carries only a ``tool_call_id``, so it is joined back to the
    assistant turn that asked for it. Unparseable results become ``{}`` rather
    than being dropped, keeping positions aligned.
    """
    calls: dict[Any, tuple[str, dict[str, Any], int]] = {}
    stream: list[ToolResult] = []
    action = 0
    for message in trajectory:
        role = message.get("role")
        if role == "assistant":
            action += 1
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                calls[call.get("id")] = (
                    function.get("name") or "",
                    _dict(function.get("arguments")),
                    action,
                )
            continue
        if role != "tool":
            continue
        # An orphaned result is attributed to the action in progress, which is
        # where it was produced.
        name, arguments, called_on = calls.get(
            message.get("tool_call_id"), ("", {}, action)
        )
        stream.append(
            ToolResult(
                name=name,
                result=_dict(message.get("content")),
                arguments=arguments,
                action=called_on,
            )
        )
    return stream


def _wrote_source(call: ToolResult) -> bool:
    """True if this call built part of the application."""
    if call.name not in WORKSPACE_WRITE_TOOLS:
        return False
    path = call.arguments.get("file_path")
    return not (isinstance(path, str) and path.lower().endswith(NON_SOURCE_SUFFIXES))


def _project(call: ToolResult) -> str:
    """The project a validation measured.

    It is a tool *argument*: only ``build_validation`` and ``test_validation``
    echo it into their metadata.
    """
    project = call.arguments.get("project_path")
    if not isinstance(project, str) or not project.strip():
        return "."
    return posixpath.normpath(project.strip().replace("\\", "/"))


def _read_static(
    metadata: dict[str, Any], arguments: dict[str, Any]
) -> tuple[int, str] | None:
    """``total_errors`` from project integrity + tsc + ESLint.

    ``early_exit`` is the integrity gate tripping before tsc/ESLint ran;
    ``file_scoped`` is ESLint alone over named files. Both measure less than a
    project run, so they are separate scopes -- as does a run that switched a
    checker off, which only the arguments record.
    """
    STATIC_SKIP_FLAGS = ("skip_typescript", "skip_eslint", "skip_integrity")

    checks_run = metadata.get("checks_run")
    if (not isinstance(checks_run, list) or not checks_run) and metadata.get(
        "overall_passed"
    ) is True:
        return None

    errors = metadata.get("total_errors")
    if not isinstance(errors, int) or isinstance(errors, bool):
        return None
    # A failed verdict with nothing counted is still a failure, not a clean bill.
    if not errors and metadata.get("overall_passed") is False:
        errors = 1
    if metadata.get("file_scoped"):
        return max(errors, 0), "static.files"
    # Checked before the skip flags: once integrity trips, tsc and ESLint never
    # ran whatever the call asked for, and the gate is the more accurate label.
    if metadata.get("early_exit"):
        return max(errors, 0), "static.integrity_gate"
    if any(_flag(arguments.get(flag)) for flag in STATIC_SKIP_FLAGS):
        return max(errors, 0), "static.partial"
    return max(errors, 0), "static.project"


def _read_build(
    metadata: dict[str, Any], arguments: dict[str, Any]
) -> tuple[int, str] | None:
    """Type errors when the ``tsc --noEmit`` gate fails, else the build's verdict.

    ``typecheck_errored`` is a toolchain failure, not a verdict on the code, so
    it yields no reading. A skipped gate means the bundler ran alone, which does
    not typecheck: its own scope.
    """
    if metadata.get("typecheck_errored"):
        return None
    gated = not (
        _flag(arguments.get("skip_typecheck"))
        or metadata.get("typecheck_skipped_no_tsconfig")
    )
    scope = "build.typechecked" if gated else "build.bundle_only"
    # The gate short-circuits the build, so this branch reports type errors and
    # `build_passed` is False without the bundler ever having run.
    if metadata.get("typecheck_passed") is False:
        return max(_count(metadata.get("type_error_count")), 1), scope
    if "build_passed" not in metadata:
        return None
    return 0 if metadata["build_passed"] else 1, scope


def _read_runtime(
    metadata: dict[str, Any], arguments: dict[str, Any]
) -> tuple[int, str] | None:
    """Faults the browser probe observed: crashes, console errors, bad requests.

    ``probe_error`` and ``environment_error`` mean the probe itself fell over,
    which is not a reading. A crawl that only loaded the page measured less than
    one that drove the UI, and one that stopped with UI undriven measured less
    than one that finished -- three scopes.
    """
    if metadata.get("probe_error") or metadata.get("environment_error"):
        return None
    if metadata.get("validation_status") != "completed":
        return None
    errors = sum(
        _count(metadata.get(key))
        for key in ("crashes", "console_errors", "critical_network_failures")
    )
    if not errors and metadata.get("overall_passed") is False:
        errors = 1
    if not metadata.get("ui_exercised"):
        scope = "runtime.load_only"
    elif metadata.get("coverage_complete"):
        scope = "runtime.complete"
    else:
        scope = "runtime.interactive"
    return errors, scope


#: The families a full score is divided by, whether or not the agent ran them.
#: Renormalising over only the families that ran would make skipping one strictly
#: better than doing it narrowly -- a clean load-only crawl claims 0.25, while no
#: crawl at all would claim the average of whatever else was measured.
SCORED_FAMILIES = ("static", "build", "runtime")

#: One entry per validation tool: the family it reports as, and its reader.
#: ``test_validation`` is deliberately out of the score. Many trajectories do
#: not call it, so making it a required family would add a frequent empty slot.
VALIDATION_FAMILIES = {
    "static_validation": ("static", _read_static),
    "build_validation": ("build", _read_build),
    "runtime_validation": ("runtime", _read_runtime),
}


def _read_validation(call: ToolResult) -> tuple[str, Reading] | None:
    """Read one successful validation result."""
    entry = VALIDATION_FAMILIES.get(call.name)
    if entry is None or call.result.get("success") is not True:
        return None
    family, read = entry
    metadata = call.result.get("metadata")
    if not isinstance(metadata, dict):
        return None
    result = read(metadata, call.arguments)
    if result is None:
        return None
    errors, scope = result
    return family, Reading(call.action, errors, scope, _project(call))


def _readings(stream: list[ToolResult]) -> dict[str, list[Reading]]:
    """Group readable validation results by family, in order.

    ``success: false`` is dropped rather than read as zero: the tool refused to
    run, so the agent got no signal back, which is not a clean bill of health.
    """
    out: dict[str, list[Reading]] = {}
    for call in stream:
        result = _read_validation(call)
        if result is None:
            continue
        family, reading = result
        out.setdefault(family, []).append(reading)
    return out


def _score_series(series: list[Reading]) -> dict[str, Any]:
    """Score one scope's readings::

        cleared    = (baseline - final) / baseline    fraction of the mess removed
        regressed  = total upward movement / baseline what the bumps put back
        actions    = model actions the recovery spent

        resolution = 1 if final == 0 else 0
        efficiency = max((cleared - regressed) / sqrt(actions), 0)
        score      = resolution * efficiency

    The square-root action penalty favors faster recovery without collapsing
    long recoveries toward zero. Efficiency cannot exceed one, and negative
    values are floored at zero. ``resolution`` gates the reward, so every
    unresolved series scores zero.

    The window closes where the count last settled, so re-running a check to
    confirm a fix neither extends the span nor dilutes the slope.
    """
    errors = [reading.errors for reading in series]
    first_dirty = next((i for i, count in enumerate(errors) if count > 0), None)
    if first_dirty is None:
        # Nothing was ever broken here, so there was nothing to resolve.
        return {
            "score": 1.0,
            "resolution": 1.0,
            "efficiency": 1.0,
            "errors": errors,
            "cleared": 1.0,
            "regressed": 0.0,
            "actions": 0,
        }

    # Anchor on the first dirty reading, never the peak: anchoring there would
    # pay a run for making things worse first.
    baseline, final = errors[first_dirty], errors[-1]

    settled = len(errors) - 1
    while settled > first_dirty and errors[settled - 1] == final:
        settled -= 1
    window = series[first_dirty : settled + 1]

    actions = max(window[-1].action - window[0].action, 1)
    cleared = (baseline - final) / baseline
    regressed = (
        sum(
            max(after.errors - before.errors, 0)
            for before, after in zip(window, window[1:])
        )
        / baseline
    )

    resolution = 1.0 if final == 0 else 0.0
    efficiency = max((cleared - regressed) / math.sqrt(actions), 0.0)
    score = resolution * efficiency

    return {
        "score": score,
        "resolution": resolution,
        "efficiency": efficiency,
        "errors": errors,
        "cleared": cleared,
        "regressed": regressed,
        "actions": actions,
    }


def _closed_by_wider(
    series: dict[tuple[str, str], list[Reading]],
) -> set[tuple[str, str]]:
    """Series left dirty that a later, wider run of the same project shows clean.

    Only readings *after* the series' last one count. That direction is the whole
    guard against the mirror case -- a narrow run failing after a wide one passed
    is new breakage, and an earlier clean run cannot answer for it.
    """
    closed = set()
    for (project, scope), group in series.items():
        if group[-1].errors == 0:
            continue
        coverage = SCOPE_COVERAGE.get(scope, 0)
        settled = group[-1].action
        if any(
            other_project == project
            and SCOPE_COVERAGE.get(other_scope, 0) > coverage
            and any(r.action > settled and r.errors == 0 for r in other)
            for (other_project, other_scope), other in series.items()
        ):
            closed.add((project, scope))
    return closed


def _score_family(readings: list[Reading]) -> dict[str, Any]:
    """Score every scope the family measured and average them, per project.

    Each scope is scored separately. A later, wider clean run can restore a
    narrow scope's resolution, but not its efficiency. The project coefficient
    then limits narrow-only evidence without penalizing extra narrow checks when
    full evidence is also present. Projects average separately.
    """
    series: dict[tuple[str, str], list[Reading]] = {}
    for reading in readings:
        series.setdefault((reading.project, reading.scope), []).append(reading)

    reports = {key: _score_series(group) for key, group in series.items()}

    # A wider clean run restores resolution only. It does not add efficiency that
    # the narrow series did not demonstrate.
    for key in _closed_by_wider(series):
        report = reports[key]
        report["resolution"] = 1.0
        report["closed_by_wider"] = True
        report["score"] = report["efficiency"]

    projects: dict[str, dict[str, Any]] = {}
    for project in sorted({project for project, scope in reports}):
        scopes = sorted({scope for owner, scope in reports if owner == project})
        coefficient = 1.0 if FULL_SCOPES & set(scopes) else NON_FULL_SCOPE_COEFFICIENT
        scored = [
            report for (owner, scope), report in reports.items() if owner == project
        ]
        quality = sum(report["score"] for report in scored) / len(scored)
        projects[project] = {
            "scopes": scopes,
            "scope_coefficient": coefficient,
            "quality_score": quality,
            "score": quality * coefficient,
        }

    return {
        "scopes": sorted({scope for _project, scope in reports}),
        "projects": projects,
        "quality_score": sum(p["quality_score"] for p in projects.values())
        / len(projects),
        "score": sum(p["score"] for p in projects.values()) / len(projects),
        "series": {
            f"{project}|{scope}": report for (project, scope), report in reports.items()
        },
    }


def _tool_success_rate(stream: list[ToolResult]) -> tuple[float, int, int] | None:
    """Fraction of the agent's non-validation calls that ran, with the counts.

    Return ``None`` when no call reported a verdict. The rate remains useful as
    metadata when its reward weight is zero.
    """
    verdicts = [
        call.result["success"]
        for call in stream
        if call.name not in VALIDATION_TOOL_NAMES
        and type(call.result.get("success")) is bool
    ]
    if not verdicts:
        return None
    succeeded = sum(verdicts)
    return succeeded / len(verdicts), succeeded, len(verdicts) - succeeded


def _score_validation_rubric(
    trajectory: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Score the validation and debugging behavior in one agent trajectory.

    Family scores are averaged over ``SCORED_FAMILIES``. Unrun families count as
    zero. A trajectory with no readable result also receives zero.
    """
    stream = _iter_tool_results(trajectory)
    families = {
        family: _score_family(readings)
        for family, readings in _readings(stream).items()
    }

    metadata: dict[str, Any] = {"families": families}
    if not families:
        # The verdict distinguishes unvalidated source work from a trajectory
        # that did not change source. Both score zero.
        metadata["verdict"] = (
            "never_validated"
            if any(_wrote_source(call) for call in stream)
            else "not_applicable"
        )
        metadata["never_run"] = list(SCORED_FAMILIES)
        return 0.0, metadata

    metadata["never_run"] = [
        family for family in SCORED_FAMILIES if family not in families
    ]
    # The fixed denominator makes each skipped family contribute zero.
    score = sum(
        families[family]["score"] for family in SCORED_FAMILIES if family in families
    ) / len(SCORED_FAMILIES)

    # Source written after the last readable result: the evidence is stale.
    last_check = max(
        (i for i, call in enumerate(stream) if _read_validation(call) is not None),
        default=-1,
    )
    stale = any(_wrote_source(call) for call in stream[last_check + 1 :])
    score *= STALE_MULTIPLIER if stale else 1.0

    tool_success = _tool_success_rate(stream)
    if tool_success is not None:
        rate, succeeded, failed = tool_success
        metadata["tool_success"] = {
            "rate": rate,
            "succeeded": succeeded,
            "failed": failed,
        }
        score = (1 - TOOL_SUCCESS_WEIGHT) * score + TOOL_SUCCESS_WEIGHT * rate

    metadata["verdict"] = "scored"
    metadata["stale"] = stale
    return score, metadata


def get_trajectory_score(
    trajectory: list[dict[str, Any]] | None,
    trajectory_score_type: str = DEFAULT_TRAJECTORY_SCORE_TYPE,
    reward_cap: int = 1000,
) -> dict[str, Any]:
    if trajectory is None or len(trajectory) == 0:
        return {"score_type": trajectory_score_type, "metadata": {}, "score": 0.0}

    if trajectory_score_type == "validation_rubric":
        score, metadata = _score_validation_rubric(trajectory)
        return {
            "score_type": trajectory_score_type,
            "metadata": metadata,
            "score": score,
        }
    elif trajectory_score_type == "tool_call_success_rate":
        metadata = {"tool_success": 0, "tool_fail": 0}
        for turn in trajectory:
            if turn.get("role") != "tool":
                continue
            content = turn.get("content")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    continue
            if not isinstance(content, dict) or "success" not in content:
                continue
            if content["success"]:
                metadata["tool_success"] += 1
            else:
                metadata["tool_fail"] += 1
        num_tool_calls = metadata["tool_success"] + metadata["tool_fail"]
        return {
            "score_type": trajectory_score_type,
            "metadata": metadata,
            "score": metadata["tool_success"] / num_tool_calls
            if num_tool_calls > 0
            else 0,  # percentage scores
        }
    elif trajectory_score_type == "validation":
        static_validation_tool_ids = []
        runtime_validation_tool_ids = []
        build_validation_tool_ids = []
        test_validation_tool_ids = []
        for turn in trajectory:
            tool_calls = turn.get("tool_calls")
            if turn.get("role") != "assistant" or tool_calls is None:
                continue
            for tool_call in tool_calls:
                tool = tool_call["function"]["name"]
                if tool == "static_validation":
                    static_validation_tool_ids.append(tool_call["id"])
                elif tool == "runtime_validation":
                    runtime_validation_tool_ids.append(tool_call["id"])
                elif tool == "build_validation":
                    build_validation_tool_ids.append(tool_call["id"])
                elif tool == "test_validation":
                    test_validation_tool_ids.append(tool_call["id"])
        metadata = {
            "static_validation": {
                "total_errors": 0,
            },
            "runtime_validation": {
                "crashes": 0,
                "console_errors": 0,
                "total_network_failures": 0,
                "smoke_pass": 0,
            },
            "build_validation": {"build_failed": 0, "type_check_failed": 0},
            "test_validation": {"failed": 0},
        }

        for turn in trajectory:
            if turn.get("role") != "tool":
                continue
            if turn["tool_call_id"] in static_validation_tool_ids:
                try:
                    validation_log = json.loads(turn["content"])
                    log_metadata = validation_log["metadata"]
                    metadata["static_validation"]["total_errors"] += log_metadata.get(
                        "total_errors", 0
                    )
                except Exception as e:
                    print(f"Static validation scoring error: {e}")
            elif turn["tool_call_id"] in runtime_validation_tool_ids:
                try:
                    validation_log = json.loads(turn["content"])
                    log_metadata = validation_log["metadata"]
                    metadata["runtime_validation"]["crashes"] += log_metadata.get(
                        "crashes", 0
                    )
                    metadata["runtime_validation"]["console_errors"] += (
                        log_metadata.get("console_errors", 0)
                    )
                    metadata["runtime_validation"]["total_network_failures"] += (
                        log_metadata.get("total_network_failures", 0)
                    )
                    if not log_metadata.get("smoke_pass", True):
                        metadata["runtime_validation"]["smoke_pass"] += 1
                except Exception as e:
                    print(f"Runtime validation scoring error: {e}")
            elif turn["tool_call_id"] in build_validation_tool_ids:
                try:
                    validation_log = json.loads(turn["content"])
                    log_data = validation_log["data"]
                    if "**Build FAILED**" in log_data:
                        metadata["build_validation"]["build_failed"] += 1
                    elif "**Type check FAILED — build skipped**" in log_data:
                        metadata["build_validation"]["type_check_failed"] += 1
                except Exception as e:
                    print(f"Build validation scoring error: {e}")
            elif turn["tool_call_id"] in test_validation_tool_ids:
                try:
                    validation_log = json.loads(turn["content"])
                    log_data = validation_log["data"]
                    if "❌ **Tests Failed**" in log_data:
                        metadata["test_validation"]["failed"] += 1
                    log_metadata = validation_log["metadata"]
                    metadata["test_validation"]["failed"] += log_metadata.get(
                        "failed", 0
                    )
                except Exception as e:
                    print(f"Test validation scoring error: {e}")

        num_errors = (
            sum(metadata["static_validation"].values())
            + sum(metadata["runtime_validation"].values())
            + sum(metadata["build_validation"].values())
            + sum(metadata["test_validation"].values())
        )
        return {
            "score_type": trajectory_score_type,
            "metadata": metadata,
            "score": max(reward_cap - num_errors, 0),  # absolute scores
        }
    else:
        raise Exception(
            "Unexpected DDB trajectory_score_type: " + trajectory_score_type
        )


TRAJECTORY_PATH = Path("/tmp/harbor_trajectory.json")
REWARD_PATH = Path(
    os.environ.get("TRAJECTORY_REWARD_PATH", "/logs/verifier/reward.txt")
)
REPORT_PATH = Path("/logs/verifier/trajectory-score.json")
TARGET_AGENT = os.environ.get("DDB_TRAJECTORY_AGENT", "ddt-agent")
TRAJECTORY_SCORE_TYPE = (
    os.environ.get("TRAJECTORY_SCORE_TYPE", DEFAULT_TRAJECTORY_SCORE_TYPE)
    .strip()
    .lower()
)
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
    flag = os.environ.get("TRAJECTORY_REWARD_ENABLED", "1")
    if flag not in {"0", "1"}:
        raise ValueError("TRAJECTORY_REWARD_ENABLED must be 0 or 1")
    if flag == "0":
        return
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
