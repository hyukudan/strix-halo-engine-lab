#!/usr/bin/env bash
# No-model ticket: per-tile vs per-m16-block cost of the CUDA tensor-core expert route (qwen_moe_tile_bench).
# Loads no model; synthetic weights for 512 experts (~1.2 GB on the card). Two forms per load (64-row tile, wide 128-row)
# with a device bitwise comparison of their outputs. Expected wall ~35 s, inner timeout 100 s.
#   usage: tile_bench_ticket.sh <qwen_moe_tile_bench executable> [reps (default 9)]
# Device: CUDA_DEVICE_ORDER=PCI_BUS_ID and BENCH_CVD (default 0) select the card; the run refuses any device whose
# name does not contain BENCH_DEVICE_MATCH (default "PRO"). Abort rules: rc != 0, a BAD or DIFFER line, or wall > 100 s.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <qwen_moe_tile_bench executable> [reps]" >&2
    exit 2
fi
BIN=$1
REPS=${2:-9}
MATCH=${BENCH_DEVICE_MATCH:-PRO}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${BENCH_CVD:-0}
name=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader)
case "$name" in
    *"$MATCH"*) ;;
    *) echo "TILEBENCH refuse: device $CUDA_VISIBLE_DEVICES is '$name', not *$MATCH*" >&2; exit 4 ;;
esac
q() { nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader; }
echo "TILEBENCH before $(q)"
start=$(date +%s)
rc=0
timeout -k 5 100 "$BIN" "$REPS" || rc=$?
echo "TILEBENCH after $(q) wall_s=$(( $(date +%s) - start )) rc=$rc"
exit "$rc"
