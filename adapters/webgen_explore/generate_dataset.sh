#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
harbor_root="$(cd -- "$script_dir/../.." && pwd)"
output_dir="$harbor_root/datasets/webgen-explore"

export PYTHONPATH="$script_dir/src:$harbor_root/adapters/ddb_explore/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-sync --project "$harbor_root" \
    python -m webgen_explore.main \
    --output-dir "$output_dir" \
    "$@"
