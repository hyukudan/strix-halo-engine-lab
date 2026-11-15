#!/bin/bash
# run-pc.sh OUT: the hetpf DS41 campaign from the CUDA host (Git Bash or Linux), one detached run that ends with OUT/summary.txt
# and OUT/.done. Steps: (1) link probe host<->rank-0 node (node side via ssh, no GPU); (2) the CUDA prefill service on the
# selected device, R (RTX alone) x2 while it is the only GPU user; (3) for each TICKETS entry "MODE:ARMS" one pair serve
# process inside a pair GPU ticket (pair_serve.sh), the arms run from here as soon as the pair is READY, then OUT/stop;
# (4) service stopped, node monitors fetched, summarize.py. Never chains a build: binaries must exist.
# Environment (no defaults for hosts or paths):
#   NODE0 / NODE0_ADDR  ssh host of rank 0 / the address this host reaches its HTTP port on;  NODE1  ssh host of rank 1
#   NODE_KIT            dir on both nodes holding pair_serve.sh, link_probe.py and the key file (KEYNAME, default key.txt)
#   NODE_ENV            "BIN=... MODEL=... TRELLIS=... ROCM=..." for pair_serve.sh (paths valid on both nodes)
#   TURN                the node's pair-ticket command prefix, e.g. "GPU_TURN_MAX=120 <lab>/gpu-turn-pair.sh"
#   SVC MODEL_RTX KEY IDS TOKENS   service exe, its gguf (native path), bearer key file, prompt ids file, prompt length
#   SVC_ARGS            extra service arguments (default: --exact --host-gib 60 --host-commit-floor-gib 20 --host-pin-wait-s 90)
#   TICKETS             default "fast:A,B,A,B fast:B,A,B,A exact:B,A,B";  HOLD per ticket (s, default 115); PORT (18790)
#   COORD               optional coordination file: "PC-BIG start/end" lines are appended to it;  WAIT_READY (s, 1800)
set -u
OUT=$1; mkdir -p "$OUT"; K=$(cd "$(dirname "$0")" && pwd)
: "${NODE0:?}" "${NODE0_ADDR:?}" "${NODE1:?}" "${NODE_KIT:?}" "${NODE_ENV:?}" "${TURN:?}" "${SVC:?}" "${MODEL_RTX:?}" "${KEY:?}" "${IDS:?}" "${TOKENS:?}"
PORT=${PORT:-18790}; SPORT=${SPORT:-8097}; HOLD=${HOLD:-115}; WAIT_READY=${WAIT_READY:-1800}; KEYNAME=${KEYNAME:-key.txt}
TICKETS=${TICKETS:-"fast:A,B,A,B fast:B,A,B,A exact:B,A,B"}
SVC_ARGS=${SVC_ARGS:-"--exact --host-gib 60 --host-commit-floor-gib 20 --host-pin-wait-s 90"}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=${PY:-python}; SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"; S=$OUT/summary.txt; J=$OUT/runs.jsonl
P() { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) cygpath -m "$1";; *) echo "$1";; esac; }
log() { echo "$(date +%T) $*" | tee -a "$OUT/driver.log"; }
coord() { [ -n "${COORD:-}" ] && echo "- $(date +%H:%M) ${COORD_NAME:-hetpf}: $*" >> "$COORD"; return 0; }
trap 'touch "$OUT/.done"' EXIT
# (1) link
$SSH "$NODE0" "cd $NODE_KIT && (setsid python3 link_probe.py serve --port $((PORT + 1)) --idle-exit 60 > lp.out 2>&1 < /dev/null &)"; sleep 2
$PY -I "$(P "$K/link_probe.py")" probe --host "$NODE0_ADDR" --port $((PORT + 1)) --mib 64 --reps 3 --state-bytes 32.2e6 \
  --json "$(P "$OUT/link.json")" > "$OUT/link.txt" 2>&1; log "link $(tail -n 1 "$OUT/link.txt")"
# (2) service
u0=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$u0" -lt 2048 ] || { log "ABORT device busy ($u0 MiB)"; exit 4; }
coord "PC-BIG start (DS41 prefill service + pair hand-off tickets, device index $CUDA_VISIBLE_DEVICES only, <= 30 min)"
"$SVC" "$MODEL_RTX" --host 127.0.0.1 --port $SPORT --positions $((TOKENS + 512)) --api-key "$(cat "$KEY")" $SVC_ARGS > "$OUT/svc.out" 2> "$OUT/svc.err" &
SP=$!; WP=$(cat /proc/$SP/winpid 2>/dev/null || echo "")
( while kill -0 $SP 2>/dev/null; do echo "$(date +%s%N), $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader,nounits)" >> "$OUT/smi.samples"; sleep 0.5; done ) & MP=$!
stop_svc() { [ -n "$WP" ] && taskkill //F //T //PID "$WP" > /dev/null 2>&1; kill $SP $MP 2>/dev/null; coord "PC-BIG end"; }
for i in $(seq 1 150); do grep -q '"event":"READY"' "$OUT/svc.out" 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
grep -q '"event":"READY"' "$OUT/svc.out" || { log "ABORT service not ready"; tail -n 5 "$OUT/svc.err" >> "$S"; stop_svc; exit 5; }
log "service ready after $((i * 2)) s"; sleep 10
$PY -I "$(P "$K/het_ab.py")" --arms R,R --ids "$(P "$IDS")" --tokens "$TOKENS" --key "$(P "$KEY")" --label R0 --out "$(P "$J")" \
  --service http://127.0.0.1:$SPORT >> "$OUT/client.log" 2>&1
# (3) pair tickets
n=0; MONS=()
for t in $TICKETS; do
  n=$((n + 1)); mode=${t%%:*}; arms=${t#*:}; lab=T$n-$mode; D=$NODE_KIT/runs/$(basename "$OUT")-$lab
  $SSH "$NODE0" "mkdir -p $D && cd $NODE_KIT && (setsid bash -c \"$TURN hetpf-$lab env PEER=$NODE1 BIND=$NODE0_ADDR PORT=$PORT KEYFILE=$NODE_KIT/$KEYNAME $NODE_ENV bash $NODE_KIT/pair_serve.sh $D $mode $HOLD > $D/ticket.out 2>&1; touch $D/ticket.done\" < /dev/null > /dev/null 2>&1 &)"
  log "$lab submitted ($arms)"; t0=$(date +%s); ok=0
  while [ $(( $(date +%s) - t0 )) -lt "$WAIT_READY" ]; do
    st=$($SSH "$NODE0" "[ -e $D/ready ] && echo R; [ -e $D/ticket.done ] && echo D" | tr -d '\n')
    case $st in *R*) ok=1; break;; *D*) break;; esac; sleep 3
  done
  [ $ok = 1 ] || { log "$lab never ready (${st:-timeout})"; $SSH "$NODE0" "touch $D/stop"; [ "${st:-}" = D ] && continue; break; }
  log "$lab ready after $(( $(date +%s) - t0 )) s"
  $PY -I "$(P "$K/het_ab.py")" --arms "$arms" --ids "$(P "$IDS")" --tokens "$TOKENS" --key "$(P "$KEY")" --label "$lab" --mode "$mode" \
    --out "$(P "$J")" --service http://127.0.0.1:$SPORT --pair "http://$NODE0_ADDR:$PORT" >> "$OUT/client.log" 2>&1
  $SSH "$NODE0" "touch $D/stop"
  for _ in $(seq 1 60); do $SSH "$NODE0" "[ -e $D/done ]" && break; sleep 2; done
  mkdir -p "$OUT/$lab"; scp -q "$NODE0:$D/{info.txt,mon0.txt,mon1.txt,ticket.out}" "$OUT/$lab/" 2>/dev/null; MONS+=(--mon "$lab=$(P "$OUT/$lab")")
  log "$lab done: $(grep -E '^(ready|stop|rc)' "$OUT/$lab/info.txt" 2>/dev/null | tr '\n' ' ')"
done
$PY -I "$(P "$K/het_ab.py")" --arms R --ids "$(P "$IDS")" --tokens "$TOKENS" --key "$(P "$KEY")" --label R0 --out "$(P "$J")" \
  --service http://127.0.0.1:$SPORT >> "$OUT/client.log" 2>&1
stop_svc; sleep 3
{ grep -E 'rror|refus' "$OUT/svc.err" | head -n 3 | cut -c1-200
  $PY -I "$(P "$K/summarize.py")" "$(P "$J")" "${MONS[@]}" --smi "$(P "$OUT/smi.samples")" --link "$(P "$OUT/link.json")" 2>&1 | head -n 50
  awk -F', ' '$3+0>c{c=$3+0} END{print "RTX max SM MHz " c}' "$OUT/smi.samples"; } >> "$S"
log "summary written"
