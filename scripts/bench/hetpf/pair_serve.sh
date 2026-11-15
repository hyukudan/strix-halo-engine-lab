#!/bin/bash
# pair_serve.sh OUT MODE HOLD: one DS41 pair serve process (rank 0 here, rank 1 on $PEER) for a remote client of the hetpf kit,
# meant to run INSIDE one pair GPU ticket on the rank-0 node. MODE exact | fast (fast = STRIX_DS41_FAST_PREFILL=1 on both
# ranks, the shared pair-defaults resolver). Writes OUT/ready once the server is READY, stops when OUT/stop exists or HOLD
# seconds after start, then OUT/done. GPU busy and sclk of both nodes are sampled every 0.25 s (OUT/mon0.txt, OUT/mon1.txt:
# "epoch_ns busy=% sclk=<dpm line>").
# Environment (same paths on both nodes): PEER (rank-1 ssh host), BIN (strix-deepseek41-serve), MODEL (Q2 gguf), TRELLIS (dir
# with ours-K4 and ours-K5), BIND (address the client reaches), PORT, KEYFILE (bearer key file), CTX (default 33280), SLOTS
# (default 2), ROCM (default /opt/rocm; its bin/ and lib/ are used), EXTRA (more serve arguments).
set -u
OUT=$1; MODE=$2; HOLD=$3
: "${PEER:?}" "${BIN:?}" "${MODEL:?}" "${TRELLIS:?}" "${BIND:?}" "${PORT:?}" "${KEYFILE:?}"
CTX=${CTX:-33280}; SLOTS=${SLOTS:-2}; ROCM=${ROCM:-/opt/rocm}; EXTRA=${EXTRA:-}
export PATH=$ROCM/bin:/usr/bin:/bin LD_LIBRARY_PATH=$ROCM/lib
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
T0=$(date +%s)
for v in $(env | grep -oE "^STRIX_[A-Z0-9_]*"); do unset "$v"; done
ENVS="STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000 HSA_DISABLE_FRAGMENT_ALLOCATOR=1"
ENVS="$ENVS STRIX_DS41_TRELLIS=$TRELLIS/ours-K4 STRIX_DS41_TRELLIS_MIX=head=$TRELLIS/ours-K5 STRIX_DS41_TRELLIS_SITES=all"
ENVS="$ENVS STRIX_DS41_TRELLIS_DENSE_FULL=1 STRIX_DS41_TRELLIS_HEADOUT=1"
case $MODE in fast) ENVS="$ENVS STRIX_DS41_FAST_PREFILL=1";; exact) ;; *) echo "bad MODE $MODE"; exit 64;; esac
A="--transport usb4auto --peer-timeout 120 --prefill-path device --slots $SLOTS --context $CTX --prefill-rows auto --no-dspark"
A="$A --http-port $PORT --http-bind $BIND --http-api-key-file $KEYFILE $EXTRA"
MON='d=/sys/class/drm/card1/device; [ -d $d ] || d=/sys/class/drm/card0/device; while :; do echo "$(date +%s%N) busy=$(cat $d/gpu_busy_percent) sclk=$(grep "\*" $d/pp_dpm_sclk | tr -d "\n")"; sleep 0.25; done'
mkdir -p "$OUT"; $SSH "$PEER" "mkdir -p $OUT" || { echo "peer unreachable"; touch "$OUT/done"; exit 3; }
H0=$(sha256sum "$BIN" | cut -c1-16); H1=$($SSH "$PEER" "sha256sum $BIN" | cut -c1-16)
echo "start $(date -Is) mode $MODE bin $H0/$H1 load $(cut -d' ' -f1-3 /proc/loadavg) memavail_gib $(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo)" > "$OUT/info.txt"
echo "envs $ENVS" >> "$OUT/info.txt"; echo "args $A" >> "$OUT/info.txt"
[ "$H0" = "$H1" ] || { echo "ABORT binary differs" >> "$OUT/info.txt"; touch "$OUT/done"; exit 4; }
bash -c "$MON" > "$OUT/mon0.txt" 2>/dev/null & M0=$!
$SSH "$PEER" "nohup bash -c '$MON' > $OUT/mon1.txt 2>/dev/null & echo \$!" > "$OUT/m1.pid"
TO=$((HOLD + 15))
$SSH "$PEER" "cd $OUT && export PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH; env $ENVS timeout -k 5 $TO $BIN $MODEL --rank 1 $A" > "$OUT/r1.log" 2>&1 & P1=$!
(cd "$OUT" && exec env $ENVS timeout -k 5 $TO "$BIN" "$MODEL" --rank 0 $A < /dev/null) > "$OUT/r0.log" 2>&1 & P0=$!
for _ in $(seq 1 240); do grep -q '"READY"' "$OUT/r0.log" && break; kill -0 $P0 2>/dev/null || break; sleep 0.5; done
if grep -q '"READY"' "$OUT/r0.log"; then echo "ready t+$(( $(date +%s) - T0 )) s" | tee -a "$OUT/info.txt" > "$OUT/ready"; fi
while [ ! -e "$OUT/stop" ] && kill -0 $P0 2>/dev/null && [ $(( $(date +%s) - T0 )) -lt "$HOLD" ]; do sleep 0.5; done
echo "stop t+$(( $(date +%s) - T0 )) s ($( [ -e "$OUT/stop" ] && echo client || echo hold/exit))" >> "$OUT/info.txt"
SP=$(pgrep -f "^$BIN $MODEL --rank 0 " | head -1); [ -n "$SP" ] && kill -TERM "$SP"
wait $P0; r0=$?; wait $P1; r1=$?
kill $M0 2>/dev/null; $SSH "$PEER" "kill \$(cat $OUT/m1.pid) 2>/dev/null"; scp -q "$PEER:$OUT/mon1.txt" "$OUT/" 2>/dev/null
echo "rc $r0/$r1 t+$(( $(date +%s) - T0 )) s" >> "$OUT/info.txt"
grep -aE "POISON|rror|STOPPED|declin|refus" "$OUT/r0.log" "$OUT/r1.log" | cut -c1-220 | head -6 >> "$OUT/info.txt"
touch "$OUT/done"
