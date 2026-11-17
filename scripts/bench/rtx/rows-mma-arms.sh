#!/usr/bin/env bash
# Child of gpu_guard.py only. No device-index override; both arms share its selected UUID and wall limit.
set -euo pipefail
binary=$1; out=$2
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || exit 96
STRIX_T_MAXROWS=16 STRIX_QRTX_ROWS_MMA=0 "$binary" --time > "$out/a.log" 2>&1
STRIX_T_MAXROWS=16 STRIX_QRTX_ROWS_MMA=1 "$binary" --time > "$out/b.log" 2>&1
"$binary" --compare "$out/a.log" "$out/b.log" > "$out/compare.log" 2>&1
# Keep raw TIME lines: their full site/row keys prevent pairing mismatched cases.
grep '^TIME ' "$out/a.log" > "$out/time-a.txt"
grep '^TIME ' "$out/b.log" > "$out/time-b.txt"
