#!/usr/bin/env bash
# Child of gpu_guard.py only. No device-index override; both arms share its selected UUID and wall limit.
set -euo pipefail
binary=$1; out=$2
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || exit 96
pairs=${ROWS_MMA_PAIRS:-1}
[[ $pairs =~ ^[1-3]$ ]] || exit 96
for ((pair=1; pair<=pairs; ++pair)); do
  STRIX_T_MAXROWS=16 STRIX_QRTX_ROWS_MMA=0 "$binary" --time > "$out/a${pair}.log" 2>&1
  STRIX_T_MAXROWS=16 STRIX_QRTX_ROWS_MMA=1 "$binary" --time > "$out/b${pair}.log" 2>&1
  "$binary" --compare "$out/a${pair}.log" "$out/b${pair}.log" > "$out/compare${pair}.log" 2>&1
  grep '^TIME ' "$out/a${pair}.log" > "$out/time-a${pair}.txt"
  grep '^TIME ' "$out/b${pair}.log" > "$out/time-b${pair}.txt"
done
