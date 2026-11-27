#!/bin/bash
# Pair transport soak, no model: strix-dist-transport-soak on both ranks, tagged 4 KiB blocks, sizes 4 KiB .. 64 MiB in
# both directions (exchange and one-way rounds). Per arm it reports what reached the caller (SOAK-DAMAGE lines: lost /
# repeated / foreign / damaged block, offset, message, direction) and what the transport repaired (the close line
# "usb4 integrity: ..." with per-stream resends, CRC failures, losses, gaps, duplicates). Nothing in the system is changed.
#   Run on rank 0 inside a pair GPU turn:  KIT_PEER=<peer host> SOAK_BIN=<dir with strix-dist-transport-soak on both nodes>
#       OUT=<output dir> bash pair-soak.sh
# Optional: ARMS (subset of raw raw-load int int-load; default all four), SECONDS_PER_ARM (default 25), SIZES (comma list),
# LD_LIBRARY_PATH (passed to the peer). Writes $OUT/summary.txt (<= 60 lines) and $OUT/done.
# Arms: raw = STRIX_DIST_USB4_INTEGRITY=0 (the stream as the driver gives it: the first damage ends the arm);
#       int = the integrity layer (default); -load = a device copy loop on each rank during the transfers.
# Memory per rank: the largest size twice (in + out) plus 1 GiB device memory with -load.
set -u
PEER=${KIT_PEER:?KIT_PEER}; B=${SOAK_BIN:?SOAK_BIN}; O=${OUT:?OUT}
ARMS=${ARMS:-"raw raw-load int int-load"}
SEC=${SECONDS_PER_ARM:-25}
SIZES=${SIZES:-4096,65536,1048576,8388608,16777216,67108864}
S=$O/summary.txt
mkdir -p "$O"; ssh -n -o BatchMode=yes "$PEER" "mkdir -p $O" || exit 2
echo "start $(date -Is) soak r0 $(sha256sum "$B/strix-dist-transport-soak" | cut -c1-16) r1 $(ssh -n -o BatchMode=yes "$PEER" sha256sum "$B/strix-dist-transport-soak" | cut -c1-16)" > "$S"
t0=$(date +%s)
arm() {  # $1 label, $2 integrity 0|1, $3 extra args
  local env="STRIX_DIST_USB4_INTEGRITY=$2 STRIX_DIST_USB4_TIMEOUT_MS=15000"
  local cap=$(( SEC + 40 ))
  ssh -n -o BatchMode=yes "$PEER" "cd $O && env $env LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} timeout $cap $B/strix-dist-transport-soak --rank 1 --seconds $SEC --sizes $SIZES $3 > $1-r1.out 2>&1; echo \$? > $1-r1.rc" &
  local pp=$!
  (cd "$O" && env $env timeout $cap "$B/strix-dist-transport-soak" --rank 0 --seconds "$SEC" --sizes "$SIZES" $3 > "$1-r0.out" 2>&1)
  local rc=$?; wait $pp
  ssh -n -o BatchMode=yes "$PEER" "cat $O/$1-r1.out" > "$O/$1-r1.copy" 2>/dev/null
  echo "== $1 integrity $2 $3 r0 rc=$rc r1 rc=$(ssh -n -o BatchMode=yes "$PEER" cat "$O/$1-r1.rc") t=$(( $(date +%s) - t0 ))s" >> "$S"
  for r in r0 r1; do
    local f="$O/$1-$r.out"; [ "$r" = r1 ] && f="$O/$1-r1.copy"
    grep -a -m1 '^rank .* transport' "$f" | cut -c1-200 >> "$S"
    grep -a -m1 '^soak:' "$f" >> "$S"
    grep -a -m3 -E '^SOAK-(DAMAGE|ERROR)' "$f" | cut -c1-300 >> "$S"
    grep -a -m1 'usb4 integrity:' "$f" | cut -c1-400 >> "$S"
  done
}
for a in $ARMS; do
  [ $(( $(date +%s) - t0 )) -gt $(( 150 - SEC )) ] && { echo "stop: wall budget before arm $a" >> "$S"; break; }
  case $a in
    raw) arm raw 0 "" ;;  raw-load) arm raw-load 0 --gpu-load ;;
    int) arm int 1 "" ;;  int-load) arm int-load 1 --gpu-load ;;
  esac
done
echo "end $(date -Is) wall $(( $(date +%s) - t0 ))s" >> "$S"; touch "$O/done"
