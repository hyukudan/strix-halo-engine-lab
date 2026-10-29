#!/usr/bin/env bash
set -euo pipefail
# Git Bash / Windows; GPU work only when explicitly invoked.
: "${CUDA_VISIBLE_DEVICES:?Set one GPU ordinal or UUID for the qualification window}"
export CUDA_VISIBLE_DEVICES
here="$(cd -- "$(dirname -- "$0")" && pwd)"
exec python "$here/rtx-session-gate.py" "$@"
