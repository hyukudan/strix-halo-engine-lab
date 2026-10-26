#!/usr/bin/env bash
set -eu
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${BENCH_PYTHON:-python3}" "$root/scripts/bench/lib/fetch_models.py" "$@"
