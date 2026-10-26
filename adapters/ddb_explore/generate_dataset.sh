#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
harbor_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$script_dir/src:$harbor_root/src${PYTHONPATH:+:$PYTHONPATH}"
# Hosts without uv use the Harbor virtual environment directly.
if command -v uv >/dev/null; then
    run=(uv run --no-sync --project "$harbor_root" python)
else
    run=("$harbor_root/.venv/bin/python")
fi
exec "${run[@]}" -m ddb_explore.main \
    --output-dir "$harbor_root/datasets/ddb-explore" \
    --prompt-data "$harbor_root/datasets/ddb-explore.prompts.jsonl" "$@"
