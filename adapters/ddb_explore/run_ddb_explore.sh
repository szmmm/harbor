#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
harbor_root="$(cd -- "$script_dir/../.." && pwd)"
export DDB_EXPLORE_DATASET="$harbor_root/datasets/ddb-explore"
export DDB_EXPLORE_CONFIG="$script_dir/run_ddb_explore_ddb.yaml"
exec bash "$script_dir/run_ddb.sh" "$@"
