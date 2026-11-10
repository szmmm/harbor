"""slime ``--rollout-function-path`` entry point backed by a Harbor job.

    --rollout-function-path harbor.slime_bridge.rollout.generate_rollout

Runs inside slime's ``RolloutManager`` Ray actor.  slime's training loop makes
this a hard barrier -- ``ray.get(rollout_manager.generate.remote(rollout_id))``
returns before ``train()`` and ``update_weights()``.

The trials themselves run on the harbor service (``harbor.slime_bridge.cli``) over HTTP,
set by ``--harbor-service-url`` or ``--harbor-service-urls``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from argparse import Namespace
from collections import Counter
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.rm_hub import batched_async_rm
from slime.utils.async_utils import run
from slime.utils.types import Sample

from harbor.slime_bridge.convert import (
    DROP_EMPTY,
    DROP_NO_TRAINABLE_TOKENS,
    DROP_OVER_CAP,
    DROP_UNSUPPORTED_AGENT,
    OnDrop,
    agent_name_of,
    is_trained_agent,
    trial_to_samples,
)
from harbor.slime_bridge.reward import TRAJECTORY_SCORE_KEY, required_score_key
from harbor.slime_bridge.trial import TrialRollout

logger = logging.getLogger(__name__)

#: Trials per group are bounded well below this; it only has to keep
#: ``group_index * _UID_STRIDE + trial_index`` collision-free within a step.
_UID_STRIDE = 4096

#: Why a received trial was not kept, reported through ``_samples_for_task(on_skip=...)``.
#: Each becomes a ``harbor/trials_skipped_<reason>`` wandb metric; a trial lands in
#: exactly one bucket, at the first gate that rejects it, so on every successful step
#: ``trials_received == trials_kept + sum(skips)``.
SKIP_SURPLUS = "surplus"
SKIP_MIXED_WEIGHTS = "mixed_weights"
SKIP_NO_TOKENS = "no_tokens"
SKIP_MISSING_SCORE = "missing_score"
SKIP_UNTRAINABLE = "untrainable"

_SKIP_REASONS = (
    SKIP_SURPLUS,
    SKIP_MIXED_WEIGHTS,
    SKIP_NO_TOKENS,
    SKIP_MISSING_SCORE,
    SKIP_UNTRAINABLE,
)
_DROP_REASONS = (
    DROP_UNSUPPORTED_AGENT,
    DROP_EMPTY,
    DROP_OVER_CAP,
    DROP_NO_TRAINABLE_TOKENS,
)


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Signature per slime's ``--rollout-function-path`` contract.

    The training output is ``list[task][trial][agent] -> Sample`` (depth 3).
    slime flattens it, but the depth-2 leaf list is what triggers
    ``slime/ray/rollout.py::_validate_rollout_id_annotated``, which requires a
    non-None, identical ``rollout_id`` on every sibling -- exactly what we want,
    so the loss reducer counts one trial once rather than once per agent.
    """
    if evaluation:
        return run(_eval(args, rollout_id))
    return run(_train(args, rollout_id, data_source))


# ---------------------------------------------------------------------------
# Training rollout
# ---------------------------------------------------------------------------


@dataclass
class _OpenGroup:
    """One task's slot in the batch, carried across refill rounds.

    ``group_index`` is fixed when the task is drawn so ``rollout_uid`` stays
    collision-free, and ``attempts`` is how many trials harbor has already run
    for this task this step -- it offsets the next round's ``trial_index`` so a
    refilled trial cannot reuse a uid the group already emitted.
    """

    group: list[Sample]
    group_index: int
    service_url: str
    samples: list[list[Sample]] = field(default_factory=list)
    attempts: int = 0


def _target_trials(min_trial_rate: float, n_trials: int) -> int:
    """Usable trials a group needs before it counts as complete."""
    if not (0 < min_trial_rate <= 1):
        raise ValueError(
            f"min_trial_rate should be in (0, 1], but we got {min_trial_rate}"
        )
    if min_trial_rate == 1:
        return n_trials
    return min(n_trials, max(2, math.ceil(min_trial_rate * n_trials)))


async def _train(
    args: Namespace, rollout_id: int, data_source: Any
) -> RolloutFnTrainOutput:
    batch_size = args.rollout_batch_size
    n_trials = args.n_samples_per_prompt
    trial_over_sample = getattr(args, "harbor_trial_over_sampling", 0)
    cancel_surplus = getattr(args, "harbor_cancel_surplus_trials", False)
    refill_rounds = getattr(args, "harbor_batch_refill_rounds", 3)
    target = _target_trials(getattr(args, "harbor_min_trial_rate", 1.0), n_trials)
    max_len = getattr(args, "harbor_max_sample_len", None) or getattr(
        args, "rollout_max_context_len", None
    )
    allow_mixed_weights = getattr(args, "harbor_allow_mixed_weight_versions", False)
    service_urls = _harbor_service_urls(args)
    if len(service_urls) > 1:
        await _preflight_harbor_services(args)

    data: list[list[list[Sample]]] = []
    deferred: list[list[Sample]] = []
    open_groups: dict[str, _OpenGroup] = {}
    seen_tasks: set[str] = set()
    next_group_index = 0
    if not getattr(args, "custom_rm_path", None):
        raise RuntimeError("--custom-rm-path is required.")
    required_score = required_score_key(args.custom_rm_path)
    await _sync_sampling_overrides(args, rollout_id)

    drops: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    trials_requested = 0
    trials_received = 0
    rounds = 0

    def on_drop(_agent_id: str, reason: str) -> None:
        drops[reason] += 1

    def on_skip(reason: str) -> None:
        skips[reason] += 1

    # Draw the whole batch up front.  Each group is n_samples_per_prompt
    # deep-copies of one prompt sample, already carrying index / group_index
    # (slime/rollout/data_source.py).  An epoch boundary can hand back a task
    # already drawn (see _dedupe_by_task); the duplicate is deferred and its
    # slot re-drawn.
    while len(open_groups) < batch_size:
        fresh, task_ids, duplicates = _dedupe_by_task(
            data_source.get_samples(batch_size - len(open_groups)), seen_tasks
        )
        deferred.extend(duplicates)
        for group, task_id in zip(fresh, task_ids):
            if getattr(args, "harbor_task_route_file", None):
                service_url = _task_routes_for_ids(args, [task_id])[task_id]
            else:
                service_url = service_urls[next_group_index % len(service_urls)]
            open_groups[task_id] = _OpenGroup(
                group=group,
                group_index=next_group_index,
                service_url=service_url,
            )
            next_group_index += 1
        if not fresh:
            break

    for round_index in range(refill_rounds + 1):
        if not open_groups:
            break
        rounds = round_index + 1

        kept = dict(
            sorted(Counter(len(g.samples) for g in open_groups.values()).items())
        )
        logger.info(
            "rollout %d round %d/%d: %d group(s) complete of %d, %d pending with kept trials %s, "
            "asking %d trial(s)/task, cap %s",
            rollout_id,
            round_index,
            refill_rounds,
            len(data),
            batch_size,
            len(open_groups),
            kept,
            n_trials + trial_over_sample,
            max_len,
        )

        # Every pending group is asked for a full group -- n_trials plus the
        # over-sampling spares -- not its shortfall: past round 0 a group is
        # only still open because trials failed, and a refill that asks for the
        # bare shortfall gambles the step on exactly the trials that just
        # failed.  What a group *takes* is still capped at what it lacks
        # (``limit`` below), so the surplus is insurance rather than a bigger
        # GRPO group, absorbed by harbor's n_concurrent_trials.
        # --harbor-cancel-surplus-trials tells the service that same cap as a
        # per-task target, so insurance a task turns out not to need is
        # cancelled mid-flight instead of running to be discarded here.
        label = f"rollout_{rollout_id}" + (
            f"_refill{round_index}" if round_index else ""
        )
        trials_requested += len(open_groups) * (n_trials + trial_over_sample)
        trials_by_task = await _run_harbor_distributed(
            args,
            label,
            list(open_groups),
            n_trials + trial_over_sample,
            task_routes={
                task_id: group.service_url for task_id, group in open_groups.items()
            },
            n_targets=(
                {
                    task_id: n_trials - len(g.samples)
                    for task_id, g in open_groups.items()
                }
                if cancel_surplus
                else None
            ),
        )
        for task_id, open_group in list(open_groups.items()):
            trials = trials_by_task.get(task_id, [])
            trials_received += len(trials)
            open_group.samples.extend(
                _samples_for_task(
                    group=open_group.group,
                    task_id=task_id,
                    trials=trials,
                    group_index=open_group.group_index,
                    first_trial_index=open_group.attempts,
                    first_accepted_index=len(open_group.samples),
                    limit=n_trials - len(open_group.samples),
                    max_len=max_len,
                    allow_mixed_weights=allow_mixed_weights,
                    required_score=required_score,
                    on_drop=on_drop,
                    on_skip=on_skip,
                )
            )
            open_group.attempts += len(trials)
            if len(open_group.samples) >= target:
                del open_groups[task_id]
                data.append(open_group.samples)

    # Anything still open failed its first job and every refill round,
    # zero-trial groups included -- a dead task keeps its slot and re-runs
    # rather than losing it to a fresh draw.
    short: dict[str, int] = {
        task_id: len(open_group.samples) for task_id, open_group in open_groups.items()
    }

    _return_to_data_source(data_source, deferred)

    batch_trials = sum(len(group) for group in data)

    # Before the empty-batch raise below, not after: a step whose groups all came
    # back short satisfies both, and this is the one that names the knob.
    if short:
        raise RuntimeError(
            f"rollout {rollout_id}: {len(short)} group(s) below the {target}-trial target "
            f"({dict(sorted(short.items()))}) with the {refill_rounds} refill round(s) spent."
        )
    if not data:
        raise RuntimeError(f"rollout {rollout_id}: no usable task group.")
    global_batch_size = getattr(args, "global_batch_size", None)
    if global_batch_size and batch_trials < global_batch_size:
        raise RuntimeError(
            f"rollout {rollout_id}: batch holds {batch_trials} trial(s) in "
            f"{len(data)} group(s), below --global-batch-size {global_batch_size}. "
            f"slime counts rollouts, not samples, so this is the number its dp_schedule "
            f"asserts on (num_steps = trials // global_batch_size >= 1) -- it would fail "
            f"there instead, after the trials have already run. Lower --global-batch-size, "
            f"or raise --rollout-batch-size."
        )
    if len(data) < batch_size:
        logger.warning(
            "rollout %d: short batch, %d/%d groups -- the data source had no more unique tasks",
            rollout_id,
            len(data),
            batch_size,
        )
    await _apply_reward_model(args, data)
    logger.info(
        "rollout %d: %d group(s), %d trial(s) in %d round(s), streams dropped %s",
        rollout_id,
        len(data),
        batch_trials,
        rounds,
        dict(sorted(drops.items())),
    )
    # Step counters plus per-leg verifier means over the kept batch, merged into
    # slime's per-step wandb log next to rollout/* (_log_rollout_data).
    # Monitoring only.  The counters make the silent subtraction visible: a step
    # that ran 96 trials and trained on 64 is otherwise byte-identical
    # downstream to one that ran 64.  Counter keys are emitted every step, zero
    # included -- a flat 0 line is the healthy signal, a missing key is
    # ambiguous -- while the *_mean keys stay conditional (absence != 0.0).
    metrics = _train_counter_metrics(
        groups_kept=len(data),
        trials_kept=batch_trials,
        trials_requested=trials_requested,
        trials_received=trials_received,
        rounds=rounds,
        skips=skips,
        drops=drops,
    )
    metrics.update(
        {
            f"harbor/{key}_mean": _mean(values)
            for key, values in sorted(_batch_verifier_components(data).items())
        }
    )
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


async def _apply_reward_model(args: Namespace, data: list[list[list[Sample]]]) -> None:
    """Route every sample's reward through ``--custom-rm-path``.

    slime invokes a custom RM from inside its *own* ``generate_rollout``
    (``slime/rollout/sglang_rollout.py``), which ``--rollout-function-path``
    replaces -- so without this call the flag parses and then does nothing.

    Going through ``batched_async_rm`` rather than resolving the dotted path
    ourselves keeps slime's dispatch intact.  Note it short-circuits on
    ``args.custom_rm_path`` -- which ``_train`` requires -- so a per-sample
    ``Sample.custom_rm_path`` is never consulted on this path.
    """
    samples = [sample for group in data for trial in group for sample in trial]
    rewards = await batched_async_rm(args, samples)
    for sample, reward in zip(samples, rewards, strict=True):
        sample.reward = float(reward)


def _dedupe_by_task(
    groups: list[list[Sample]], seen: set[str]
) -> tuple[list[list[Sample]], list[str], list[list[Sample]]]:
    """Keep one group per task id; return the rest to be deferred.

    ``get_samples`` concatenates the tail of one shuffled epoch with the head of
    the next (``slime/rollout/data_source.py``), so one task can appear twice in
    a single draw.  Left alone that is silent duplicate training: harbor would
    run 2xG trials of it, *both* groups would read the same
    ``trials_by_task[task_id]`` list, and every trial would be converted twice
    under different rollout_uids -- training on the same tokens twice in one
    batch, with the surplus trials all sharing ``group[0]``'s sample index.

    ``seen`` accumulates over the whole step and is updated in place, so a
    top-up draw cannot redraw a task that already holds a slot -- which would
    be the same double-conversion across draws instead of within one.
    """
    unique: list[list[Sample]] = []
    task_ids: list[str] = []
    duplicates: list[list[Sample]] = []

    for group in groups:
        task_id = task_id_of(group[0])
        if task_id in seen:
            duplicates.append(group)
            continue
        seen.add(task_id)
        unique.append(group)
        task_ids.append(task_id)

    if duplicates:
        logger.warning(
            "%d duplicate task group(s) in this draw (an epoch boundary splits "
            "get_samples across two shuffles); deferring them so their trials "
            "are not converted twice",
            len(duplicates),
        )
    return unique, task_ids, duplicates


def _return_to_data_source(data_source: Any, groups: list[list[Sample]]) -> None:
    """Hand unused prompt groups back so they are not skipped for the epoch.

    ``get_samples`` advances the dataset offset, so a group drawn and not
    trained on is lost until the next pass.  ``RolloutDataSourceWithBuffer``
    exists for exactly this, and returns them from the buffer first on the next
    step.  A read-only data source raises instead, which is not worth failing an
    otherwise-complete rollout over.
    """
    if not groups:
        return
    add_samples = getattr(data_source, "add_samples", None)
    if add_samples is None:
        return
    try:
        add_samples(groups)
    except Exception as exc:  # read-only source, or a group-size mismatch
        logger.warning(
            "could not return %d unused prompt group(s) to the data source: %s",
            len(groups),
            exc,
        )


def _check_weight_versions(
    trial: TrialRollout,
    task_id: str,
    trial_index: int,
    *,
    allow_mixed: bool,
) -> bool:
    """False when this trial straddled an ``update_weights()`` boundary.

    slime's training loop is a hard barrier -- ``generate`` returns before
    ``train()`` and ``update_weights()`` -- so under a healthy run every turn of
    every trial samples from one policy version.  More than one version means
    the barrier leaked.
    """
    trained = [agent_id for agent_id in trial.token_ids if is_trained_agent(agent_id)]
    versions = trial.distinct_weight_versions(trained)
    if len(versions) <= 1:
        return True

    detail = ", ".join(
        f"{agent_name_of(agent_id)}{'' if agent_id in trained else ' (not trained)'}"
        f"={agent_versions}"
        for agent_id, agent_versions in sorted(trial.weight_versions.items())
        if agent_versions
    )
    message = (
        f"task {task_id} trial {trial_index} ({trial.trial_name}) spans weight versions "
        f"{sorted(versions)} -- a request crossed an update_weights() boundary, so this "
        f"trial mixes two policies. Per agent: {detail}. "
        f"Proxy session {trial.session_id}."
    )
    if not allow_mixed:
        raise RuntimeError(
            message
            + " Refusing to train on it. slime's loop is a hard barrier, so this means a request outlived generate_rollout."
        )
    logger.error("%s Dropping it (--harbor-allow-mixed-weight-versions).", message)
    return False


def _trial_has_score(trial: TrialRollout, key: str) -> bool:
    """Whether every stream this trial could train carries the RM's score."""
    scores = (
        trial.trajectory_scores
        if key == TRAJECTORY_SCORE_KEY
        else trial.verifier_scores
    )
    return all(
        scores.get(agent_id) is not None
        for agent_id in trial.token_ids
        if is_trained_agent(agent_id)
    )


def _samples_for_task(
    *,
    group: list[Sample],
    task_id: str,
    trials: list[TrialRollout],
    group_index: int,
    first_trial_index: int,
    first_accepted_index: int,
    limit: int,
    max_len: int | None,
    allow_mixed_weights: bool,
    required_score: str | None = None,
    on_drop: OnDrop | None = None,
    on_skip: Callable[[str], None] | None = None,
) -> list[list[Sample]]:
    """Convert one task's trials into per-trial sibling lists.

    ``on_skip(reason)`` is called once per received trial that is *not* kept,
    with one of the ``SKIP_*`` constants -- the trial-level counterpart of
    convert's per-stream ``on_drop``.  Every rejected trial reports exactly one
    reason, at the first gate that rejects it.

    Two independent indices, because they count different things:

    * ``first_trial_index`` offsets the rollout_uid by everything harbor has
      already *run* for this task -- refills and over-sampled surplus included --
      so no two trials of a group can share a uid.
    * ``first_accepted_index`` offsets into ``group`` by what this task has
      already *kept*, since ``group`` holds exactly --n-samples-per-prompt prompt
      copies and only accepted trials need one.

    ``limit`` is the number still wanted, so neither --harbor-trial-over-sampling
    nor a service returning more trials than it was asked for can push a group
    past --n-samples-per-prompt.  Trials beyond it are never converted.
    """
    per_task: list[list[Sample]] = []

    for offset, trial in enumerate(trials):
        if len(per_task) >= limit:
            # Insurance this task did not need: the group is already back to --n-samples-per-prompt.
            if on_skip is not None:
                on_skip(SKIP_SURPLUS)
            continue
        trial_index = first_trial_index + offset

        if not _check_weight_versions(
            trial, task_id, trial_index, allow_mixed=allow_mixed_weights
        ):
            if on_skip is not None:
                on_skip(SKIP_MIXED_WEIGHTS)
            continue

        if not trial.has_tokens:
            logger.warning(
                "task %s trial %d: no token data (%s)",
                task_id,
                trial_index,
                trial.error,
            )
            if on_skip is not None:
                on_skip(SKIP_NO_TOKENS)
            continue

        if required_score is not None and not _trial_has_score(trial, required_score):
            logger.warning(
                "task %s trial %d (%s): recorded token streams but no %s -- "
                "--custom-rm-path would fail the step on it, so it is not kept (error=%s)",
                task_id,
                trial_index,
                trial.trial_name,
                required_score,
                trial.error,
            )
            if on_skip is not None:
                on_skip(SKIP_MISSING_SCORE)
            continue

        # Indexed by acceptance position, not by trial index: ``group`` is
        # --n-samples-per-prompt deep-copies of one prompt, while trial_index runs
        # over every trial harbor ran for this task.  Keying on trial_index would
        # collide two *accepted* trials onto group[0] as soon as a refill round or
        # --harbor-trial-over-sampling pushed it past the group's depth, giving two
        # kept samples the same Sample.index.
        accepted_index = first_accepted_index + len(per_task)
        proto = group[accepted_index] if accepted_index < len(group) else group[0]
        samples = trial_to_samples(
            proto,
            trial,
            rollout_uid=rollout_uid(group_index, trial_index),
            max_len=max_len,
            on_drop=on_drop,
        )
        if not samples:
            logger.warning(
                "task %s trial %d: %d recorded agent stream(s) %s but none trainable "
                "(no supported role, all prompt-only, or all over the %s-token cap)",
                task_id,
                trial_index,
                len(trial.token_ids),
                sorted({agent_name_of(a) for a in trial.token_ids}),
                max_len,
            )
            if on_skip is not None:
                on_skip(SKIP_UNTRAINABLE)
            continue

        per_task.append(samples)

    return per_task


#: ``args`` attribute -> proxy sampling-param key, for the --harbor-sampling-*
#: flags (args.py).  Distribution keys only; max_tokens/stop are DDB's.
_SAMPLING_FLAG_KEYS = (
    ("harbor_sampling_temperature", "temperature"),
    ("harbor_sampling_top_p", "top_p"),
    ("harbor_sampling_top_k", "top_k"),
    ("harbor_sampling_min_p", "min_p"),
)


def _requested_sampling_overrides(args: Namespace) -> dict[str, Any]:
    """The sampling policy the --harbor-sampling-* flags pin (unset = absent)."""
    overrides: dict[str, Any] = {}
    for attr, key in _SAMPLING_FLAG_KEYS:
        value = getattr(args, attr, None)
        if value is not None:
            overrides[key] = value
    return overrides


async def _sync_sampling_overrides(args: Namespace, rollout_id: int) -> None:
    """Push the run's sampling overrides to the rollout proxy and verify them.

    Runs at every step start (train and eval): the proxy's runtime override
    layer is in-memory, so a proxy restart mid-run would otherwise fall back
    to lower layers silently.  One localhost call per multi-hour step, same
    philosophy as the weight-version assertion -- drift becomes a loud
    failure before any trial runs, not a quietly off-policy batch after.

    No-op when no --harbor-sampling-* flag is set.  Raises RuntimeError,
    naming the knob, when a flag is set but --harbor-proxy-url or
    $PROXY_API_KEY is missing, the proxy is unreachable, or the echoed
    effective policy does not adopt a requested value.
    """
    overrides = _requested_sampling_overrides(args)
    if not overrides:
        return
    proxy_url = getattr(args, "harbor_proxy_url", None)
    if not proxy_url:
        raise RuntimeError(
            "--harbor-sampling-* flags are set but --harbor-proxy-url is not: "
            "the sampling overrides have nowhere to be pushed."
        )
    api_key = os.getenv("PROXY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "--harbor-sampling-* flags are set but $PROXY_API_KEY is not: "
            "PUT /sampling_overrides requires the bare proxy key."
        )
    if overrides.get("top_p") not in (None, 1.0):
        logger.warning(
            "rollout %d: --harbor-sampling-top-p %s pins truncated sampling the trainer "
            "cannot correct for -- the proxy records no top-p replay spans, and --use-tis "
            "fixes logprob numerics only.",
            rollout_id,
            overrides["top_p"],
        )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(
                f"{str(proxy_url).rstrip('/')}/sampling_overrides",
                json=overrides,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            effective = response.json().get("effective") or {}
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"rollout {rollout_id}: cannot push sampling overrides to the proxy at {proxy_url}: {e}"
        ) from e
    mismatched = {
        key: (value, effective.get(key))
        for key, value in overrides.items()
        if effective.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            f"rollout {rollout_id}: the proxy at {proxy_url} did not adopt the pushed sampling "
            f"overrides: {mismatched} (requested, effective) -- is it an older proxy build, or "
            f"is a different policy pinned in its config?"
        )
    logger.info(
        "rollout %d: sampling overrides in effect at the proxy: %s",
        rollout_id,
        effective,
    )


async def _run_harbor(
    args: Namespace,
    run_label: str,
    task_ids: list[str],
    n_trials: int,
    n_targets: dict[str, int] | None = None,
) -> dict[str, list[TrialRollout]]:
    """Run every trial on the harbor service and block until all terminate.

    ``n_targets`` (task id -> usable trials wanted) lets the service cancel a
    task's surplus attempts early; cancelled trials never appear in the reply.
    """
    service_url = getattr(args, "harbor_service_url", None)
    if not service_url:
        raise RuntimeError(
            "--harbor-service-url is required: harbor.slime_bridge drives harbor over HTTP."
        )

    deadline = getattr(args, "harbor_service_deadline_secs", 86400.0)
    if deadline <= 0:
        raise RuntimeError("--harbor-service-deadline-secs must be greater than zero.")

    body: dict[str, Any] = {
        "task_ids": task_ids,
        "n_attempts": n_trials,
        "run_label": run_label,
    }
    if n_targets is not None and len(n_targets) > 0:
        body["n_targets"] = n_targets
    token = os.getenv("HARBOR_RUNNER_TOKEN")
    request_kwargs: dict[str, Any] = {"json": body}
    if token is not None and token != "":
        request_kwargs["headers"] = {"Authorization": f"Bearer {token}"}
    async with asyncio.timeout(deadline):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=30.0)
        ) as client:
            response = await client.post(
                f"{str(service_url).rstrip('/')}/rollout",
                **request_kwargs,
            )
            response.raise_for_status()
            payload = response.json()
    return {
        task_id: [TrialRollout.from_dict(t) for t in trials]
        for task_id, trials in payload.items()
    }


# ---------------------------------------------------------------------------
# Evaluation rollout
# ---------------------------------------------------------------------------


def _harbor_service_urls(args: Namespace) -> list[str]:
    """Return the configured runner URLs, preserving singular compatibility."""
    singular = getattr(args, "harbor_service_url", None)
    plural = getattr(args, "harbor_service_urls", None)
    if singular and plural:
        raise RuntimeError(
            "--harbor-service-url and --harbor-service-urls cannot be used together."
        )
    urls = list(plural or ([singular] if singular else []))
    if not urls:
        raise RuntimeError(
            "--harbor-service-url or --harbor-service-urls is required: "
            "harbor.slime_bridge drives harbor over HTTP."
        )
    normalized = [str(url).rstrip("/") for url in urls]
    if any(not url for url in normalized):
        raise RuntimeError("Harbor service URLs must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("--harbor-service-urls must not contain duplicates.")
    return normalized


def _task_routes_for_ids(args: Namespace, task_ids: list[str]) -> dict[str, str]:
    """Resolve task ids to configured services, using a fixed manifest when set."""
    service_urls = _harbor_service_urls(args)
    route_file = getattr(args, "harbor_task_route_file", None)
    if not route_file:
        return {
            task_id: service_urls[index % len(service_urls)]
            for index, task_id in enumerate(task_ids)
        }

    try:
        manifest = json.loads(Path(route_file).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read Harbor task route manifest {route_file!r}: {exc}"
        ) from exc
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Harbor task route manifest schema_version must be 1.")
    runners = manifest.get("runners")
    tasks = manifest.get("tasks")
    if not isinstance(runners, dict) or not isinstance(tasks, dict):
        raise RuntimeError(  # noqa: TRY004 -- malformed configuration, consistent routing error
            "Harbor task route manifest requires object fields runners and tasks."
        )

    missing = sorted(set(task_ids) - set(tasks))
    if missing:
        raise RuntimeError(
            f"Harbor task route manifest is missing requested task ids: {missing}."
        )

    routes: dict[str, str] = {}
    for task_id in task_ids:
        task = tasks[task_id]
        runner_id = task.get("runner") if isinstance(task, dict) else None
        runner = runners.get(runner_id) if runner_id else None
        service_url = runner.get("service_url") if isinstance(runner, dict) else None
        if not service_url:
            raise RuntimeError(
                f"Harbor task {task_id!r} references unknown runner {runner_id!r}."
            )
        service_url = str(service_url).rstrip("/")
        if service_url not in service_urls:
            raise RuntimeError(
                f"Harbor task {task_id!r} is routed to unconfigured service {service_url!r}."
            )
        routes[task_id] = service_url
    return routes


async def _run_harbor_distributed(
    args: Namespace,
    run_label: str,
    task_ids: list[str],
    n_trials: int,
    *,
    task_routes: dict[str, str] | None = None,
    n_targets: dict[str, int] | None = None,
) -> dict[str, list[TrialRollout]]:
    """Partition one rollout across runners and merge only complete responses."""
    service_urls = _harbor_service_urls(args)
    duplicate_task_ids = sorted(
        task_id for task_id, count in Counter(task_ids).items() if count > 1
    )
    if duplicate_task_ids:
        raise RuntimeError(
            f"Harbor rollout request contains duplicate task ids: {duplicate_task_ids}."
        )
    if task_routes is None:
        task_routes = _task_routes_for_ids(args, task_ids)
    if set(task_routes) != set(task_ids):
        missing = sorted(set(task_ids) - set(task_routes))
        unexpected = sorted(set(task_routes) - set(task_ids))
        raise RuntimeError(
            f"Harbor task routes do not match request: missing={missing}, unexpected={unexpected}."
        )

    partitions: dict[str, list[str]] = {url: [] for url in service_urls}
    for task_id in task_ids:
        service_url = str(task_routes[task_id]).rstrip("/")
        if service_url not in partitions:
            raise RuntimeError(
                f"Harbor task {task_id!r} is routed to unconfigured service {service_url!r}."
            )
        partitions[service_url].append(task_id)

    async def run_partition(
        service_url: str, assigned: list[str]
    ) -> dict[str, list[TrialRollout]]:
        service_args = copy(args)
        service_args.harbor_service_url = service_url
        service_args.harbor_service_urls = None
        targets = (
            {
                task_id: n_targets[task_id]
                for task_id in assigned
                if task_id in n_targets
            }
            if n_targets is not None
            else None
        )
        if targets is not None and len(targets) > 0:
            return await _run_harbor(
                service_args,
                run_label,
                assigned,
                n_trials,
                n_targets=targets,
            )
        return await _run_harbor(service_args, run_label, assigned, n_trials)

    tasks: dict[str, asyncio.Task[dict[str, list[TrialRollout]]]] = {}
    async with asyncio.TaskGroup() as task_group:
        for service_url, assigned in partitions.items():
            if assigned:
                tasks[service_url] = task_group.create_task(
                    run_partition(service_url, assigned)
                )

    merged: dict[str, list[TrialRollout]] = {}
    for service_url, task in tasks.items():
        response = task.result()
        assigned = set(partitions[service_url])
        returned = set(response)
        if assigned != returned:
            missing = sorted(assigned - returned)
            unexpected = sorted(returned - assigned)
            raise RuntimeError(
                f"Harbor service {service_url} returned incomplete task ids: "
                f"missing task ids {missing}, unexpected task ids {unexpected}."
            )
        merged.update(response)
    return {task_id: merged[task_id] for task_id in task_ids}


async def _preflight_harbor_services(args: Namespace) -> None:
    """Fail before rollout when a multi-runner pool is inconsistent."""
    service_urls = _harbor_service_urls(args)
    if len(service_urls) < 2:
        return

    async with (
        httpx.AsyncClient(timeout=30.0) as client,
        asyncio.TaskGroup() as task_group,
    ):
        tasks = {
            service_url: task_group.create_task(client.get(f"{service_url}/health"))
            for service_url in service_urls
        }

    identities: dict[str, dict[str, Any]] = {}
    for service_url, task in tasks.items():
        response = task.result()
        response.raise_for_status()
        identities[service_url] = response.json()

    required = {"runner_id", "profile_id", "model_name"}
    missing_fields = {
        url: sorted(required - set(identity))
        for url, identity in identities.items()
        if required - set(identity)
    }
    runner_ids = [identity.get("runner_id") for identity in identities.values()]
    profiles = {identity.get("profile_id") for identity in identities.values()}
    models = {identity.get("model_name") for identity in identities.values()}
    if (
        missing_fields
        or len(set(runner_ids)) != len(runner_ids)
        or len(profiles) != 1
        or len(models) != 1
    ):
        raise RuntimeError(
            "Harbor runner pool mismatch: runners must have unique runner_id values "
            "and matching profile_id/model_name values; "
            f"health={identities}, missing_fields={missing_fields}."
        )


async def _eval(args: Namespace, rollout_id: int) -> RolloutFnEvalOutput:
    """Score the eval task list.

    slime consumes ``{dataset_name: {"rewards": [...], "truncated": [...],
    "samples": [...]}}`` (see ``sglang_rollout.eval_rollout_single_dataset``).
    We report rewards only: eval does not need trainable samples, so the trials
    run through the same service but nothing here converts them.

    **The eval reward is always the verifier score**, whichever RM
    ``--custom-rm-path`` names for *training*.
    """
    task_ids = getattr(args, "harbor_eval_task_ids", None)
    if not task_ids:
        logger.warning(
            "no --harbor-eval-task-ids configured; skipping eval %d", rollout_id
        )
        return RolloutFnEvalOutput(data={})

    n_trials = getattr(args, "harbor_eval_n_samples", 1)
    if len(_harbor_service_urls(args)) > 1:
        await _preflight_harbor_services(args)
    await _sync_sampling_overrides(args, rollout_id)
    trials_by_task = await _run_harbor_distributed(
        args, f"eval_{rollout_id}", list(task_ids), n_trials
    )

    rewards: list[float] = []
    trajectory_scores: list[float] = []
    component_scores: dict[str, list[float]] = {}
    eval_trials = 0
    eval_unscored = 0
    tasks_no_trials = 0
    for task_id in task_ids:
        trials = trials_by_task.get(task_id, [])
        eval_trials += len(trials)
        if not trials:
            tasks_no_trials += 1
        for trial in trials:
            verifier_score = trial_verifier_score(trial)
            if verifier_score is not None:
                rewards.append(verifier_score)
            else:
                eval_unscored += 1
            for key, value in trial.verifier_components.items():
                component_scores.setdefault(key, []).append(float(value))
            trajectory_scores.extend(
                float(score)
                for agent_id, score in trial.trajectory_scores.items()
                if is_trained_agent(agent_id)
            )

    # Eval-side step counters, emitted whenever eval ran -- the all-failed path
    # included, where they are the only record that distinguishes "ran and got
    # nothing" (counters at 0) from eval never being configured (metrics=None,
    # the early return above).
    metrics: dict[str, float] = {
        "harbor/eval_trials_requested": len(task_ids) * n_trials,
        "harbor/eval_trials": eval_trials,
        "harbor/eval_trials_unscored": eval_unscored,
        "harbor/eval_tasks_no_trials": tasks_no_trials,
    }

    if not rewards:
        logger.warning(
            "eval %d: %d task(s) produced no scored trial -- reporting no eval metrics for this step.",
            rollout_id,
            len(task_ids),
        )
        return RolloutFnEvalOutput(data={}, metrics=metrics)

    data = {
        getattr(args, "harbor_eval_dataset_name", None) or "harbor": {
            "rewards": rewards,
            "truncated": [0] * len(rewards),
        }
    }
    # Reported through metrics rather than in `data`: slime only reads "rewards",
    # "truncated" and "samples" out of a dataset entry, and passes `metrics`
    # straight into the eval log dict (_log_eval_rollout_data).
    if trajectory_scores:
        metrics["harbor/eval_trajectory_score_mean"] = _mean(trajectory_scores)
    # Per-leg verifier means over the same trials the rewards came from. Monitoring only.
    metrics.update(
        {
            f"harbor/eval_{key}_mean": _mean(values)
            for key, values in sorted(component_scores.items())
        }
    )
    logger.info(
        "eval %d: %d trial(s), mean verifier score %.4f, mean trajectory score %s",
        rollout_id,
        len(rewards),
        sum(rewards) / len(rewards),
        f"{_mean(trajectory_scores):.4f}" if trajectory_scores else "n/a",
    )
    return RolloutFnEvalOutput(data=data, metrics=metrics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rollout_uid(group_index: int, trial_index: int) -> int:
    """Unique per trial *within one step*, which is all ``rollout_id`` needs to be:
    slime fills unset values with fresh ints per step and only uses them to
    group siblings (``slime/ray/rollout.py::_convert_samples_to_train_data``)."""
    return group_index * _UID_STRIDE + trial_index


def task_id_of(sample: Sample) -> str:
    metadata = sample.metadata or {}
    task_id = metadata.get("task_id")
    if not task_id:
        raise ValueError(
            "prompt dataset row has no metadata.task_id; harbor.slime_bridge needs it to locate the task directory in the service's dataset."
        )
    return str(task_id)


def _mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def trial_verifier_score(trial: TrialRollout) -> float | None:
    scores = trial.verifier_scores.values()
    return (sum(float(s) for s in scores) / len(scores)) if scores else None


def _train_counter_metrics(
    *,
    groups_kept: int,
    trials_kept: int,
    trials_requested: int,
    trials_received: int,
    rounds: int,
    skips: Counter[str],
    drops: Counter[str],
) -> dict[str, float]:
    """The step's counter metrics, every key present even at zero.

    Fixed reason tuples are union'd with the observed counter keys, so a reason
    added upstream still surfaces (under its own key) instead of being lost.
    ``requested - received`` is cancelled surplus plus trials that never
    materialized; ``received == kept + sum(skips)`` on every successful step.
    """
    metrics: dict[str, float] = {
        "harbor/groups_kept": groups_kept,
        "harbor/trials_kept": trials_kept,
        "harbor/trials_requested": trials_requested,
        "harbor/trials_received": trials_received,
        "harbor/refill_rounds_used": max(rounds - 1, 0),
    }
    for reason in sorted(set(_SKIP_REASONS) | set(skips)):
        metrics[f"harbor/trials_skipped_{reason}"] = skips[reason]
    for reason in sorted(set(_DROP_REASONS) | set(drops)):
        metrics[f"harbor/streams_dropped_{reason}"] = drops[reason]
    return metrics


def _batch_verifier_components(
    data: list[list[list[Sample]]],
) -> dict[str, list[float]]:
    """Per-leg verifier scores of the batch's kept trials, keyed by component.

    One value per trial: the components are trial-level (reward.json), so a
    trial's first stream speaks for all of them.
    """
    per_key: dict[str, list[float]] = {}
    for group in data:
        for trial_samples in group:
            components = (trial_samples[0].metadata or {}).get(
                "verifier_components"
            ) or {}
            for key, value in components.items():
                per_key.setdefault(key, []).append(float(value))
    return per_key
