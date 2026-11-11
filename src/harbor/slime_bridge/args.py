"""``--harbor-*`` CLI flags.

slime's ``parse_args`` takes a hook that receives the parser and must return it
(``slime/utils/arguments.py``: ``parser = add_custom_arguments(parser)``)::

    from slime.utils.arguments import parse_args
    from harbor.slime_bridge.args import add_harbor_arguments

    args = parse_args(add_custom_arguments=add_harbor_arguments)
"""

from __future__ import annotations

import argparse


def add_harbor_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("harbor.slime_bridge")
    group.add_argument(
        "--harbor-disable-verification", action="store_true", default=False
    )
    group.add_argument("--harbor-stability-dir", type=str, default=None)

    # --- the harbor service -----------------------------------------------
    group.add_argument(
        "--harbor-service-url",
        type=str,
        default=None,
        help="Harbor rollout service, e.g. http://127.0.0.1:9500. REQUIRED for a "
        "training or eval rollout -- _run_harbor raises without it.",
    )
    group.add_argument(
        "--harbor-service-urls",
        type=str,
        nargs="+",
        default=None,
        help="Harbor rollout services for multi-node rollout. Tasks are assigned "
        "round-robin and keep the same service across refill rounds. Mutually "
        "exclusive with --harbor-service-url.",
    )
    group.add_argument(
        "--harbor-service-deadline-secs",
        type=float,
        default=86400.0,
        help="Overall deadline for one Harbor service request (default: 24 hours).",
    )
    group.add_argument(
        "--harbor-task-route-file",
        type=str,
        default=None,
        help="JSON manifest that pins each task id to a Harbor runner. Required "
        "when runner-local image shards are used.",
    )

    # --- sampling policy ----------------------------------------------------
    # The run's rollout sampling distribution, pushed to the rollout proxy
    # (PUT /sampling_overrides) at every step start and verified against the
    # echoed effective policy -- a proxy restart or a divergent value fails
    # the step loudly instead of silently sampling off-policy.  These are
    # deliberately NOT slime's --rollout-* flags: those are inert in this
    # topology, and --rollout-top-p must stay 1.0 (slime asserts on it).
    group.add_argument(
        "--harbor-proxy-url",
        type=str,
        default=None,
        help="Rollout proxy base URL, e.g. http://127.0.0.1:9400. REQUIRED when any "
        "--harbor-sampling-* flag is set; the push also needs $PROXY_API_KEY (the "
        "bare proxy key) in the trainer's environment.",
    )
    group.add_argument(
        "--harbor-sampling-temperature",
        type=float,
        default=1.0,
        help="Pin rollout temperature at the proxy, over whatever DDB requests.",
    )
    group.add_argument(
        "--harbor-sampling-top-p",
        type=float,
        default=1.0,
        help="Pin rollout top-p at the proxy, over whatever DDB requests.",
    )
    group.add_argument(
        "--harbor-sampling-top-k",
        type=int,
        default=None,
        help="Pin rollout top-k at the proxy (-1 disables top-k), over whatever DDB requests. "
        "Unset = not pinned.",
    )
    group.add_argument(
        "--harbor-sampling-min-p",
        type=float,
        default=None,
        help="Pin rollout min-p at the proxy, over whatever DDB requests. "
        "Unset = not pinned.",
    )

    # --- batch shaping ----------------------------------------------------
    group.add_argument(
        "--harbor-trial-over-sampling",
        type=int,
        default=0,
        help="Extra trials to run per task on top of every full-group request, so a "
        "group survives lost trials without a refill round.",
    )
    group.add_argument(
        "--harbor-cancel-surplus-trials",
        action="store_true",
        default=False,
        help="Cancel a task's still-running trials once it already has as many usable "
        "trials as it can keep (its shortfall to --n-samples-per-prompt), instead of "
        "letting --harbor-trial-over-sampling spares and full-group refill requests "
        "run to completion only to be discarded.",
    )
    group.add_argument(
        "--harbor-min-trial-rate",
        type=float,
        default=1.0,
        help="Fraction of --n-samples-per-prompt a task's group must reach before it "
        "counts as complete: target = min(n, max(2, ceil(rate * n))). Below target a "
        "group is refilled while --harbor-batch-refill-rounds allows, and the step "
        "aborts if it is still short.",
    )
    group.add_argument(
        "--harbor-batch-refill-rounds",
        type=int,
        default=3,
        help="Extra harbor jobs re-run for a group still below --harbor-min-trial-rate, "
        "zero usable trials included. Each refill keeps the group's usable trials and "
        "asks for a full group again; a group still short when the budget is spent "
        "fails the step.",
    )
    group.add_argument(
        "--harbor-max-sample-len",
        type=int,
        default=None,
        help="Drop stream tokens past this length, cutting at a turn boundary. "
        "Defaults to --rollout-max-context-len.",
    )

    # --- correctness gates ------------------------------------------------
    group.add_argument(
        "--harbor-allow-mixed-weight-versions",
        action="store_true",
        default=False,
        help="Drop trials that span two engine weight versions instead of failing the step.",
    )

    # --- eval --------------------------------------------------------------
    group.add_argument(
        "--harbor-eval-task-ids",
        type=str,
        nargs="+",
        default=None,
        help="Task ids for the eval rollout.",
    )
    group.add_argument(
        "--harbor-eval-n-samples",
        type=int,
        default=1,
        help="Trials per task during eval.",
    )
    group.add_argument(
        "--harbor-eval-dataset-name",
        type=str,
        default="harbor",
        help="Label eval rewards are reported to slime.",
    )

    return parser
