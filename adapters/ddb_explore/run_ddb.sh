#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
harbor_root="$(cd -- "$script_dir/../.." && pwd)"
env_file="${EXPLORE_ENV_FILE:-$harbor_root/utils/harbor-rollout/ddb_rubrics_env.sh}"
dataset_dir="${DDB_EXPLORE_DATASET:?Set DDB_EXPLORE_DATASET}"
config_file="${DDB_EXPLORE_CONFIG:?Set DDB_EXPLORE_CONFIG}"

if [[ ! -f "$env_file" ]]; then
    echo "ERROR: Environment file not found: $env_file" >&2
    exit 1
fi

# Load the existing credentials only in this process. The source file and any
# already-running ddb-rubrics jobs are not changed.
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

: "${VLM_EXPLORE_ENABLED:=1}"
: "${VLM_EXPLORE_MAX_STEPS:=24}"
: "${VLM_OMIT_TEMPERATURE:=1}"

proxy_url="${DDB_PROXY_URL:-${PROXY_URL:-}}"

required_names=(DDB_BASE_URL DDB_API_KEY)
for name in "${required_names[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: $name is not set in $env_file" >&2
        exit 1
    fi
done

if [[ -z "$proxy_url" ]]; then
    echo "ERROR: DDB_PROXY_URL or PROXY_URL is not set in $env_file" >&2
    exit 1
fi

case "${VLM_EXPLORE_ENABLED,,}" in
    1|true|yes|on)
        if [[ -z "${VLM_API_KEY:-}" ]]; then
            echo "ERROR: VLM_API_KEY is required when VLM exploration is enabled" >&2
            exit 1
        fi
        if [[ -z "${VLM_MODEL:-}" ]]; then
            echo "ERROR: VLM_MODEL must be set" >&2
            exit 1
        fi
        ;;
esac

if [[ ! -d "$dataset_dir" ]]; then
    echo "ERROR: Dataset directory not found: $dataset_dir" >&2
    echo "Run the adapter's generate_dataset.sh first." >&2
    exit 1
fi

export VLM_EXPLORE_ENABLED
export VLM_EXPLORE_MAX_STEPS
export VLM_OMIT_TEMPERATURE
export VLM_MODEL

cd -- "$harbor_root"
exec uv run harbor job start \
    -c "$config_file" \
    --agent-kwarg "proxy_url=$proxy_url" \
    "$@"
