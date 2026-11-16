#!/bin/bash
# Strict ABAB arms inside ONE server process (one model load): arms differ only by per-request fields (the shared serve
# API's measurement switches, e.g. "copy_rows", "draft_depth_max"), so a short GPU turn holds 3+ pairs. Same layout as
# qwen_serve_abab.sh (OUT/<k>-<arm>/bench.txt, summary.txt with per-arm busy SM MHz), so qwen_serve_summary.py reads it.
#   SERVE_EXE, SERVE_ARGS, SERVE_ENV (environment of the one server)
#   ARMS        'A|{"copy_rows":8} ;; B|{"copy_rows":16}'   (name|JSON object merged into every request of the arm)
#   ORDER       "A B A B A B"
#   CLIENT      client command taking --extra-json and the arm's output path (qwen_serve_classes.py ...)
#   OUT         output directory;  GPU_INDEX (default 0): the nvidia-smi index the server is pinned to
set -u
: "${SERVE_EXE:?}" "${SERVE_ARGS:?}" "${ARMS:?}" "${ORDER:?}" "${CLIENT:?}" "${OUT:?}"
GPU_INDEX=${GPU_INDEX:-0}
mkdir -p "$OUT"
smi() { nvidia-smi -i "$GPU_INDEX" --query-gpu=index,memory.used,utilization.gpu,clocks.sm --format=csv,noheader | tr '\n' ' '; }
declare -A JS
IFS=';' read -ra LIST <<< "${ARMS//;;/;}"
for spec in "${LIST[@]}"; do
  [ -n "${spec// /}" ] || continue
  n=$(echo "$spec" | cut -d'|' -f1 | tr -d ' '); JS[$n]=$(echo "$spec" | cut -d'|' -f2- | sed 's/^ *//;s/ *$//')
done
echo "start $(date +%T) smi $(smi) (one server, arms by request fields)" > "$OUT/summary.txt"
# shellcheck disable=SC2086
env CUDA_VISIBLE_DEVICES=$GPU_INDEX ${SERVE_ENV:-} "$SERVE_EXE" $SERVE_ARGS > "$OUT/serve.log" 2>&1 &
SPID=$!
k=0
for name in $ORDER; do
  k=$((k + 1)); A="$OUT/$k-$name"; mkdir -p "$A"
  ( while kill -0 $SPID 2>/dev/null; do echo "$(date +%T) $(smi)" >> "$A/smi.samples"; sleep 1; done ) & MPID=$!
  timeout "${LIMIT_S:-900}" $CLIENT --extra-json "${JS[$name]}" "$A/bench.txt" > "$A/client.log" 2>&1; rc=$?
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null
  { echo "== $k $name rc $rc fields ${JS[$name]}"
    grep -cE ' C[148] ' "$A/bench.txt" | sed 's/^/result lines: /'
    awk -v g="$GPU_INDEX," '$2==g {u+=$5; m++} $2==g && $5+0>=50 {s+=$7; n++}
      END{printf "busy-sample mean SM MHz %.0f (n=%d), mean util %.0f %%\n", n?s/n:0, n, m?u/m:0}' "$A/smi.samples"
  } >> "$OUT/summary.txt"
  [ $rc = 0 ] || break
done
WPID=$(cat /proc/$SPID/winpid 2>/dev/null || true)
if [ -n "$WPID" ]; then taskkill //F //PID "$WPID" > /dev/null 2>&1; else kill $SPID 2>/dev/null; fi
wait $SPID 2>/dev/null
echo "end $(date +%T) smi $(smi)" >> "$OUT/summary.txt"
