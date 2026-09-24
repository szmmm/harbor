#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
harbor_root="$(cd -- "$script_dir/../.." && pwd)"
export DDB_EXPLORE_DATASET="$harbor_root/datasets/webgen-explore"
export DDB_EXPLORE_CONFIG="$script_dir/run_webgen_explore_ddb.yaml"
exec bash "$harbor_root/adapters/ddb_explore/run_ddb.sh" "$@"
