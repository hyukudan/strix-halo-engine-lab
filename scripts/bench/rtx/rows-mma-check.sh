#!/usr/bin/env bash
# Usage: rows-mma-check.sh COORD_ROOT TEST_BINARY OUTPUT_DIR
# Run only after the owner announces the test brief. One FIFO ticket, PRO0 at534W, <=180s guarded workload including both arms; <5min including guard setup/post-check.
set -euo pipefail
coord=$(cd "$1" && pwd); binary=$2; out=$3
repo=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
lock="$coord/tools/pc-pro.lock"; stop="$coord/tools/pc-pro.STOP"
wait="$coord/tools/pc-pro.wait.sol-rows-mma.$$"
held=0
cleanup() { rm -f -- "$wait"; if ((held)); then rm -f -- "$lock/holder"; rmdir -- "$lock"; fi; }
trap cleanup EXIT
trap 'exit 130' INT TERM
[[ -x "$binary" && ! -e "$stop" ]] || exit 96
# Retain the timestamp. Never expire or remove another waiter's ticket/lock.
: > "$wait"
oldest() { find "$coord/tools" -maxdepth 1 -type f -name 'pc-pro.wait.*' -printf '%T@ %p\n' | sort -n | sed -n '1p' | cut -d' ' -f2-; }
deadline=$((SECONDS+1800))
while :; do
  [[ ! -e "$stop" ]] || exit 96
  ((SECONDS < deadline)) || exit 96
  if [[ $(oldest) == "$wait" ]] && mkdir "$lock" 2>/dev/null; then
    held=1
    # Recheck after atomic lock acquisition; a concurrently visible older ticket wins.
    if [[ $(oldest) == "$wait" ]]; then break; fi
    rmdir "$lock"; held=0
  fi
  sleep 5
done
printf 'sol-rows-mma %s\n' "$$" > "$lock/holder"
identity=$(nvidia-smi -i 0 --query-gpu=name,power.limit --format=csv,noheader,nounits | tr -d '\r')
[[ $identity == *'RTX PRO 6000'* ]] || exit 96
power=${identity##*,}; power=${power// /}
awk -v p="$power" 'BEGIN {exit !(p >= 533.5 && p <= 534.5)}' || exit 96
# Shared guard resolves the PRO by name/UUID; child inherits that UUID (never overwrites it with an index).
python "$repo/scripts/bench/rtx/gpu_guard.py" --gpu-name 'RTX PRO 6000' --timeout-s 180 \
  --max-idle-mib 2048 --latch "$stop" --out "$out/guard" -- \
  bash "$repo/scripts/bench/rtx/rows-mma-arms.sh" "$binary" "$out"
[[ -s "$out/guard/guard.json" ]] || exit 97
