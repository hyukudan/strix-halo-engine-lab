#!/bin/bash
# Link cost of ONE rank-to-rank exchange on a two-node pair, no model: strix-dist-gpu-probe (device producer, Communicator
# host path: D2H, wire, H2D, stream sync) per payload size and idle gap, with and without the user-space busy-poll of the
# USB4 host transport (STRIX_DIST_USB4_SPIN=1). Nothing in the system is changed.
#   Run on rank 0 inside a pair GPU turn:  KIT_PEER=<peer host> PROBE_BIN=<dir with strix-dist-gpu-probe on both nodes>
#       OUT=<output dir> bash pair-link-probe.sh
# Optional: SIZES (default 4096 12288 102400 1048576 16777216), LD_LIBRARY_PATH (passed to the peer), ARMS (subset of
# b2b g250 g30k s250 s30k). Writes $OUT/summary.txt (<= 40 lines) and $OUT/done.
# Arms: b2b = back to back; g250 = 250 us idle before each exchange (one exchange per intra-layer split step);
# g30k = 30 ms idle (the pipeline decode hop); s250 / s30k = the same with STRIX_DIST_USB4_SPIN=1.
# Wall: ~10 s per arm (load + rendezvous ~4 s); each arm is capped at 25 s and no arm starts after 105 s.
set -u
PEER=${KIT_PEER:?KIT_PEER}; B=${PROBE_BIN:?PROBE_BIN}; O=${OUT:?OUT}
SIZES=${SIZES:-"4096 12288 102400 1048576 16777216"}
ARMS=${ARMS:-"b2b g250 g30k s250 s30k"}
S=$O/summary.txt; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"; ssh -n -o BatchMode=yes "$PEER" "mkdir -p $O" || exit 2
echo "start $(date -Is) probe r0 $(sha256sum "$B/strix-dist-gpu-probe" | cut -c1-16) r1 $(ssh -n -o BatchMode=yes "$PEER" sha256sum "$B/strix-dist-gpu-probe" | cut -c1-16)" > "$S"
t0=$(date +%s)
arm() {  # $1 label, $2 pause us, $3 iterations, $4 spin 0|1
  local env="STRIX_PROBE_PATH=host STRIX_PROBE_PAUSE_US=$2 STRIX_PROBE_ITERATIONS=$3 STRIX_PROBE_LABEL=$1 STRIX_DIST_USB4_SPIN=$4"
  ssh -n -o BatchMode=yes "$PEER" "cd $O && env $env LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} timeout 25 $B/strix-dist-gpu-probe 1 $SIZES > $1-r1.out 2>&1; echo \$? > $1-r1.rc" &
  local pp=$!
  (cd "$O" && env $env timeout 25 "$B/strix-dist-gpu-probe" 0 $SIZES > "$1-r0.out" 2>&1)
  local rc=$?; wait $pp
  echo "== $1 pause ${2}us spin $4 iters $3 r0 rc=$rc r1 rc=$(ssh -n -o BatchMode=yes "$PEER" cat "$O/$1-r1.rc") t=$(( $(date +%s) - t0 ))s" >> "$S"
  python3 "$HERE/pair_link_sum.py" < "$O/$1-r0.out" >> "$S"
}
for a in $ARMS; do
  [ $(( $(date +%s) - t0 )) -gt 105 ] && { echo "stop: wall budget before arm $a" >> "$S"; break; }
  case $a in
    b2b) arm b2b 0 200 0 ;;  g250) arm g250 250 200 0 ;;  g30k) arm g30k 30000 30 0 ;;
    s250) arm s250 250 200 1 ;;  s30k) arm s30k 30000 30 1 ;;
  esac
done
echo "end $(date -Is) wall $(( $(date +%s) - t0 ))s" >> "$S"; touch "$O/done"
