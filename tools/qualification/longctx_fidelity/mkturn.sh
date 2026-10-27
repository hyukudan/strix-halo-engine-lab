#!/usr/bin/env bash
# Run from the staged repository root; generates a script without executing it.
# Submit the result through your GPU-turn wrapper or scheduler.
set -euo pipefail
python3 -m tools.qualification.longctx_fidelity.turn "$@"
