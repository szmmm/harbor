#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
harbor_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$script_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-sync --project "$harbor_root" \
    python -m ddb_explore.main \
    --output-dir "$harbor_root/datasets/ddb-explore" "$@"
