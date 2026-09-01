#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
harbor_root="$(cd -- "$script_dir/../.." && pwd)"
output_dir="$harbor_root/datasets/ddb-rubrics"

exec uv run --no-project \
    --with-editable "$script_dir" \
    ddb-rubrics \
    --output-dir "$output_dir" \
    "$@"
