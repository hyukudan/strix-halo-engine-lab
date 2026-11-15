#!/usr/bin/env bash
# DS41 two-node prefill, one A/B arm. Run on the rank-0 node INSIDE a pair GPU ticket; rank 1 runs on the peer
# over ssh with the same binary (sha checked) and the same environment.
# Usage: prefill_ab.sh --bin BIN --model GGUF --prompt CSV --out DIR --tag TAG [--peer HOST] [--env "K=V ..."]
#                      [--repeat N] [--trace] [--rocm DIR]
#   --env    the arm's switches plus the deployment's prefill environment (weight overlays etc.)
#   --trace  STRIX_DIST_TRACE per rank (H/G lines: produced / received per exchange) into DIR/TAG/tr.r0|r1
# Prints one "SUM" line: cold (first) and warm (last) wall ms per rank-0 run, logits fnv of both ranks, GPU busy %
# and clock per timed run on both nodes, the kfd evicted_ms increase and the minimum MemAvailable during the arm.
set -u
bin= model= prompt= out= tag= peer=peer xenv= rep=2 trace=0 rocm=/opt/rocm
while [ $# -gt 0 ]; do case $1 in
  --bin) bin=$2; shift;; --model) model=$2; shift;; --prompt) prompt=$2; shift;; --out) out=$2; shift;;
  --tag) tag=$2; shift;; --peer) peer=$2; shift;; --env) xenv=$2; shift;; --repeat) rep=$2; shift;;
  --trace) trace=1;; --rocm) rocm=$2; shift;; *) echo "prefill_ab: unknown $1" >&2; exit 64;; esac; shift; done
[ -x "$bin" ] && [ -f "$model" ] && [ -f "$prompt" ] && [ -n "$out" ] && [ -n "$tag" ] || { echo "prefill_ab: missing args" >&2; exit 64; }
export PATH=$rocm/bin:/usr/bin:/bin LD_LIBRARY_PATH=$rocm/lib
W=$out/$tag; rm -rf "$W"; mkdir -p "$W"
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
$SSH "$peer" "rm -rf $W; mkdir -p $W" || exit 3
for v in $(env | grep -oE "^STRIX_[A-Z0-9_]*"); do unset "$v"; done
ENVS="STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 $xenv"
[ $trace = 1 ] && ENVS="$ENVS STRIX_DIST_TRACE=$W/tr"
h0=$(sha256sum "$bin" | cut -c1-16); h1=$($SSH "$peer" "sha256sum $bin" | cut -c1-16)
[ "$h0" = "$h1" ] || { echo "ABORT $tag binary differs: $h0 $h1"; exit 4; }
A="--transport usb4-auto --prompt $prompt --ttft-timing --ttft-warmup 512 --repeat $rep"
$SSH "$peer" "test -f $prompt && test -f $model" || { echo "ABORT $tag prompt or model missing on the peer"; exit 5; }
# sampler: time, gpu busy %, sclk Hz, summed kfd evicted_ms of live processes, MemAvailable kB
SAMP='HW=$(dirname /sys/class/hwmon/hwmon*/freq1_input | head -1); while :; do echo "$(date +%s.%N) $(cat /sys/class/drm/card0/device/gpu_busy_percent) $(cat $HW/freq1_input) $(cat /sys/class/kfd/kfd/proc/*/stats_*/evicted_ms 2>/dev/null | awk "{s+=\$1} END{print s+0}") $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 0.04; done'
$SSH "$peer" "cd $W && (timeout 90 bash -c '$SAMP' > $W/samp1.txt 2>/dev/null &); date +%s.%N > $W/t0.r1; env $ENVS timeout -k 10 80 $bin $model --rank 1 $A" > "$W/r1.log" 2>&1 & R1=$!
(timeout 90 bash -c "$SAMP" > "$W/samp0.txt" 2>/dev/null &)
date +%s.%N > "$W/t0.r0"
(cd "$W" && env $ENVS timeout -k 10 85 "$bin" "$model" --rank 0 $A < /dev/null) > "$W/r0.log" 2>&1; rc0=$?
wait $R1; rc1=$?
pkill -u "$USER" -f "gpu_busy_percent" 2>/dev/null; $SSH "$peer" "pkill -u \$USER -f gpu_busy_percent" 2>/dev/null
scp -q -o BatchMode=yes "$peer:$W/samp1.txt" "$peer:$W/t0.r1" "$W/" 2>/dev/null
[ $trace = 1 ] && scp -q -o BatchMode=yes "$peer:$W/tr.r1" "$W/" 2>/dev/null
grep -ahE "rror|refus|FAIL" "$W/r0.log" "$W/r1.log" | head -3 | cut -c1-200
run() { grep -a "^ttft run $2: [0-9]* rows" "$W/r$1.log" | sed -E 's/.*wall ([0-9.]+) ms.*logits_fnv ([0-9a-f]+).*/\1 \2/'; }
# per timed run: mean busy %, mean sclk MHz; per arm: evicted_ms increase and minimum MemAvailable (MiB)
busy() { grep -aE "^ttft run [0-9]+: [0-9.]+ ms \(at" "$W/r$1.log" | sed -E 's/ttft run ([0-9]+): ([0-9.]+) ms \(at ([0-9.]+) ms\)/\2 \3/' |
  while read -r d at; do awk -v t0="$(cat "$W/t0.r$1")" -v d="$d" -v at="$at" 'BEGIN{e=t0+at/1000; s=e-d/1000}
    $1>=s && $1<=e {n++; b+=$2; f+=$3/1e6} END{if(n) printf "%.0f%%@%.0f ", b/n, f/n}' "$W/samp$1.txt"; done; }
mem() { awk 'NF>=5 {if(!n++){e0=$4}; if($4>em)em=$4; if(!m||$5<m)m=$5} END{printf "evicted+%dms memavail_min=%dMiB", em-e0, m/1024}' "$W/samp$1.txt"; }
last=$((rep - 1))
echo "SUM $tag rc=$rc0/$rc1 sha=$h0 | cold $(run 0 0) | warm $(run 0 $last) | r1 cold/warm fnv $(run 1 0 | cut -d' ' -f2)/$(run 1 $last | cut -d' ' -f2) | busy@MHz r0 $(busy 0)r1 $(busy 1)| r0 $(mem 0) r1 $(mem 1)"
