"""Frozen-policy rollout diagnostics, with no verifier or reward filtering.

Full conversations/token streams remain in Harbor and proxy session artifacts.
Returned Samples are explicitly marked diagnostic records, never training data.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from copy import copy
from pathlib import Path

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.async_utils import run
from slime.utils.types import Sample

from harbor.slime_bridge.rollout import (
    _dedupe_by_task,
    _return_to_data_source,
    _run_harbor_distributed,
    _sync_sampling_overrides,
)

logger = logging.getLogger(__name__)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if (
        evaluation
        or not args.debug_rollout_only
        or not args.harbor_disable_verification
    ):
        raise RuntimeError(
            "Stability rollouts require debug-rollout-only and disabled verification."
        )
    if not args.harbor_stability_dir:
        raise RuntimeError("--harbor-stability-dir is required.")
    return run(_rollout(args, rollout_id, data_source))


async def _rollout(args, rollout_id, data_source):
    started = time.monotonic()
    await _sync_sampling_overrides(args, rollout_id)
    seen, groups, task_ids, deferred = set(), [], [], []
    while len(groups) < args.rollout_batch_size:
        fresh, ids, duplicates = _dedupe_by_task(
            data_source.get_samples(args.rollout_batch_size - len(groups)), seen
        )
        groups.extend(fresh)
        task_ids.extend(ids)
        deferred.extend(duplicates)
        if not fresh:
            break
    _return_to_data_source(data_source, deferred)
    if len(groups) != args.rollout_batch_size:
        raise RuntimeError(
            "Not enough distinct prompts for the requested stability batch."
        )
    trials_by_task = await _run_harbor_distributed(
        args, f"rollout_{rollout_id}", task_ids, args.n_samples_per_prompt
    )
    records, samples = [], []
    for group, task_id in zip(groups, task_ids, strict=True):
        trials = trials_by_task.get(task_id, [])
        if len(trials) > args.n_samples_per_prompt:
            raise RuntimeError(f"Unexpected extra trials for {task_id}: {len(trials)}")
        for attempt in range(args.n_samples_per_prompt):
            trial = trials[attempt] if attempt < len(trials) else None
            state = trial.output_state if trial else {}
            record = {
                "task_id": task_id,
                "attempt": attempt + 1,
                "trial_name": trial.trial_name if trial else None,
                "session_id": trial.session_id if trial else None,
                "ddb_task_id": trial.ddb_task_id if trial else None,
                "status": "missing"
                if trial is None
                else "error"
                if trial.error is not None
                else "completed",
                "error": trial.error if trial else "Harbor returned no trial record",
                "has_tokens": bool(trial and trial.has_tokens),
                "agent_streams": len(trial.token_ids) if trial else 0,
                "recorded_tokens": sum(map(len, trial.token_ids.values()))
                if trial
                else 0,
                "output_state": state or {"preview_status": "not_observed"},
            }
            records.append(record)
            sample = copy(group[attempt])
            sample.rollout_id = rollout_id * 100000 + len(records)
            sample.reward = None
            sample.status = (
                Sample.Status.COMPLETED
                if record["status"] == "completed"
                else Sample.Status.ABORTED
            )
            sample.metadata = {
                **(sample.metadata or {}),
                "diagnostic_only": True,
                **record,
            }
            samples.append(sample)
    counts = Counter(r["status"] for r in records)
    previews = Counter(r["output_state"]["preview_status"] for r in records)
    metrics = {
        "stability/requested": len(records),
        "stability/completed": counts["completed"],
        "stability/errors": counts["error"],
        "stability/missing": counts["missing"],
        "stability/with_tokens": sum(r["has_tokens"] for r in records),
        "stability/recorded_tokens": sum(r["recorded_tokens"] for r in records),
        "stability/elapsed_seconds": time.monotonic() - started,
        **{f"stability/preview_{k}": v for k, v in previews.items()},
    }
    report = {
        "step": rollout_id,
        "verification_disabled": True,
        "optimizer_updates": False,
        "metrics": metrics,
        "trials": records,
    }
    directory = Path(args.harbor_stability_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step-{rollout_id:04d}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)
    logger.info(
        "Stability step %s: %s", rollout_id, json.dumps(metrics, sort_keys=True)
    )
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)


def log_rollout(rollout_id, args, samples, metrics, rollout_time):
    """Avoid Slime's reward statistics: these records intentionally have no reward."""
    from slime.observability import logging_utils

    values = {**metrics, "rollout/step": rollout_id}
    logging_utils.log(args, values, step_key="rollout/step")
    return True
