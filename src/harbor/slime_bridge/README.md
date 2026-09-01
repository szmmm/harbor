# harbor.slime_bridge

Drives the Harbor/DDB agentic rollout pipeline from [slime](https://github.com/THUDM/slime).

## Connecting Slime with Harbor

Two processes. Harbor runs as a service on the host that owns Docker and the DDB stack; the trainer reaches it over HTTP and holds no harbor code.

```bash
# on the CPU Docker host
python -m harbor.slime_bridge.cli --port 9500 \
       --dataset-dir /path/to/dataset \
       --model-name rl-exp-model \
       --n-concurrent-trials 8
```

```bash
# on the GPU trainer
python train_harbor.py \
       --rollout-function-path            harbor.slime_bridge.rollout.generate_rollout \
       --custom-rm-path                   harbor.slime_bridge.reward.harbor_verifier_reward \
       --custom-reward-post-process-path  harbor.slime_bridge.reward.harbor_reward_post_process \
       --rollout-batch-size 8 --n-samples-per-prompt 16 --num-steps-per-rollout 2 \
       --prompt-data tasks.jsonl --input-key instruction \
       --harbor-service-url http://<docker-host>:9500 \
       ...
```

* `--custom-rm-path` is **required** and picks the training signal: `harbor_verifier_reward` trains on the VLM judge's score for the built app, `harbor_trajectory_reward` on harbor's per-agent trajectory score (under the default `validation` type, `max(1000 - errors, 0)`). See §0.
* Dataset, jobs dir, model name, concurrency and timeouts are configured on the **service** — they are not `--harbor-*` flags. `--harbor-service-url` is required.
* `--harbor-*` flags need a parser hook that slime's stock `train.py` does not install. Use a four-line entry point beside it (`train_harbor.py` on the GPU node):

  ```python
  from slime.utils.arguments import parse_args
  from harbor.slime_bridge.args import add_harbor_arguments
  from train import train

  train(parse_args(add_custom_arguments=add_harbor_arguments))
  ```

* The package ships inside harbor (`src/harbor/slime_bridge`). On the GPU node install the checkout without its dependency tree — `pip install --no-deps -e /root/harbor` — so `harbor.slime_bridge` is importable inside slime's `RolloutManager` Ray actor; harbor's lazy top-level `__init__` imports nothing heavy, and the GPU-side modules touch none of harbor's internals.
* The bridge is dataset-agnostic: `--dataset-dir` (or `HARBOR_DATASET_DIR`) points at a directory of harbor task directories, one per task. Prompt dataset rows carry the task id in metadata, and the id is the basename of the task's directory under `--dataset-dir`:

  ```json
  {"instruction": "<instruction.md>", "metadata": {"task_id": "task-000001"}}
  ```

* Backing services (`source env.sh` first): MCP server `:6276`, DDB headless API `:8001`, and the rollout proxy `:9400` started with `--transport-mode slime --router-url http://$SGLANG_ROUTER_IP:$SGLANG_ROUTER_PORT`.

## Modules

| Module             | Host | Role                                                                                                                                                       |
|--------------------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `trial.py`         | both | `TrialRollout`, the wire type — no slime, no harbor.                                                                                                       |
| `harbor_client.py` | CPU  | Builds and runs the Harbor `Job`, returns `TrialRollout`s.                                                                                                 |
| `cli.py`           | CPU  | The rollout service; the only thing that calls harbor.                                                                                                     |
| `docker_host.py`   | CPU  | Asks the Docker daemon for its bridge gateway — the `DDB_PREVIEW_BASE_URL` default.                                                                        |
| `convert.py`       | GPU  | `TrialRollout` → `Sample`. Pure logic; imports only `slime.utils.types`.                                                                                   |
| `reward.py`        | GPU  | `harbor_verifier_reward` / `harbor_trajectory_reward` (the reward choice, `--custom-rm-path`) and `harbor_reward_post_process` (GRPO advantages per task). |
| `rollout.py`       | GPU  | The slime entry point: data source → service → samples → metrics.                                                                                          |
| `args.py`          | GPU  | `--harbor-*` flags the rollout function consumes.                                                                                                          |

## Batch shaping

The batch's trial count varies by design: tasks are GRPO groups, kept complete by trial over-sampling and refill rounds, with abort gates instead of silent shrinkage. That is GRPO/DAPO methodology rather than harbor wiring, so it lives in [`Rollout_Strategy.md`](Rollout_Strategy.md), together with the four `--harbor-*` flags that shape it — all implemented in `rollout.py`.

## Sampling policy: pinned at the proxy, owned by the trainer

slime never builds a sampling payload in this topology — DDB does, and the proxy resolves `slime push > sampling_overrides (config) > request > generation_config`. So `--rollout-temperature`, `--rollout-top-k`, `--rollout-max-response-len` and `--rollout-stop` are **inert**; the run's policy is the `--harbor-sampling-*` flags (`args.py` — distribution keys only, `max_tokens`/`stop` stay DDB's). `rollout._sync_sampling_overrides` PUTs them to `--harbor-proxy-url` (`/sampling_overrides`, bare `$PROXY_API_KEY` bearer) at every step start, train and eval, and raises unless the echoed effective policy adopts them — the proxy's runtime layer is in-memory, so re-pushing each step turns a proxy restart into a loud failure instead of a silent fallback. Two slime flags actively break this path:

* `--rollout-top-p != 1.0` → slime asserts `rollout_top_p_token_ids is not None` on every sample; the proxy records no top-p replay spans.
* `--use-rollout-routing-replay` → needs `return_routed_experts`, which the proxy neither requests nor records.

## Tests

```bash
venv/bin/python -m pytest -c /dev/null src/harbor/slime_bridge/tests/ -q   # -c /dev/null: the workspace venv lacks harbor's pytest plugins
```

They run without slime installed: `tests/conftest.py` stubs the slime modules, and `test_stub_matches_real_slime` — skipped unless slime is importable — pins the stub to the real `Sample` dataclass. Run the suite once inside the training environment to exercise it. `test_rollout.py` monkeypatches `_run_harbor`, covering the batch-shaping decisions (dedupe, refill rounds, re-run dead tasks, deferred duplicates, the raise gates, drop and skip metering, the step counter metrics, eval output) without a Docker host. `test_harbor_client.py` pins this package's half of the harbor contract.

What no test here executes is harbor's `_parse_record` itself — the suite hand-builds its `TrialResult`s, so the record tuple shape, mask alignment and per-agent split are exercised only inside harbor. A harbor refactor there breaks the contract silently; the first symptom is a tuple-unpacking `TypeError` hours into a rollout step.

* **`harbor-rollout/harbor_rollout.py`** — the standalone smoke-test driver. `harbor_client.py` owns the RL `JobConfig`; the two hold separate copies and `harbor_rollout.py` could become a shim over it.
