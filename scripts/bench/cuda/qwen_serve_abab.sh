#!/bin/bash
# Strict ABAB server arms for the Qwen CUDA server: one server process per arm (load, client, kill this arm's server only),
# GPU samples every second (index, MiB, util, SM MHz) beside every arm. Ends with OUT/summary.txt.
#   SERVE_EXE   server executable (strix-cuda-qwen-serve)
#   SERVE_ARGS  model / tables / sidecar / port / slot flags (the deploy declaration's values)
#   ARMS        "A|ENV=v ENV2=v ;; B|ENV=w"   (name|environment of the arm)
#   ORDER       arm names in run order, e.g. "A B A B A B"
#   CLIENT      client command; the arm's output path is appended (e.g. "python qwen_serve_classes.py http://127.0.0.1:8095 ...")
#   OUT         output directory;  GPU_INDEX (default 0): the nvidia-smi index the server is pinned to
set -u
: "${SERVE_EXE:?}" "${SERVE_ARGS:?}" "${ARMS:?}" "${ORDER:?}" "${CLIENT:?}" "${OUT:?}"
GPU_INDEX=${GPU_INDEX:-0}
mkdir -p "$OUT"
smi() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm --format=csv,noheader | tr '\n' ' '; }
declare -A ENVS
IFS=';' read -ra LIST <<< "${ARMS//;;/;}"
for spec in "${LIST[@]}"; do
  [ -n "${spec// /}" ] || continue
  ENVS[$(echo "$spec" | cut -d'|' -f1 | tr -d ' ')]=$(echo "$spec" | cut -d'|' -f2)
done
echo "start $(date +%T) smi $(smi)" > "$OUT/summary.txt"
k=0
for name in $ORDER; do
  k=$((k + 1)); A="$OUT/$k-$name"; mkdir -p "$A"
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES=$GPU_INDEX ${ENVS[$name]} "$SERVE_EXE" $SERVE_ARGS > "$A/serve.log" 2>&1 &
  SPID=$!
  ( while kill -0 $SPID 2>/dev/null; do echo "$(date +%T) $(smi)" >> "$A/smi.samples"; sleep 1; done ) & MPID=$!
  timeout "${LIMIT_S:-900}" $CLIENT "$A/bench.txt" > "$A/client.log" 2>&1; rc=$?
  WPID=$(cat /proc/$SPID/winpid 2>/dev/null || true)
  if [ -n "$WPID" ]; then taskkill //F //PID "$WPID" > /dev/null 2>&1; else kill $SPID 2>/dev/null; fi
  wait $SPID 2>/dev/null; sleep 3; kill $MPID 2>/dev/null
  { echo "== $k $name rc $rc env [${ENVS[$name]}]"
    grep -cE ' C[148] ' "$A/bench.txt" | sed 's/^/result lines: /'
    # sample fields: time, index, MiB, MiB, util, %, SM MHz, MHz (per GPU)
    awk -v g="$GPU_INDEX," '$2==g {u+=$5; m++} $2==g && $5+0>=50 {s+=$7; n++}
      END{printf "busy-sample mean SM MHz %.0f (n=%d), mean util %.0f %%\n", n?s/n:0, n, m?u/m:0}' "$A/smi.samples"
  } >> "$OUT/summary.txt"
done
echo "end $(date +%T) smi $(smi)" >> "$OUT/summary.txt"
