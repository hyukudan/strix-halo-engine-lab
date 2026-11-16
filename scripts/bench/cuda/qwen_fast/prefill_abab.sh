#!/usr/bin/env bash
# Strict ABAB prefill arms of the CUDA Qwen server on one GPU: each arm = a fresh server process, the prefill client
# (cold + warm per length), GPU clock / utilisation samples, then the server is stopped. Outputs only under $OUT.
# Run it inside the GPU-ownership wrapper of your lab (this script takes no lock and posts nothing).
#
# Required env: SERVE (strix-cuda-qwen-serve binary), MODEL (merged GGUF), TABLES, ROPE, IDS (token ids of a long text),
#               OUT (output dir). Optional: MTP, VOCAB, PORT (8095), GPU_INDEX (0), LIMIT_S (900 per arm),
#               CLIENT_ARGS ("--lens 8192,32768 --runs 2"), COMMON ("--max-positions 34816 --http-slots 1").
# ARMS="name|ENV=v ENV2=v|--flags ;; name2||--profile fast ;; ..."  (order = run order, e.g. exactA fastA exactB fastB ...)
# Summary: $OUT/summary.txt (per arm: pf lines, busy-sample mean SM MHz, GPU busy fraction, peak MiB); $OUT/.done at exit.
set -u
: "${SERVE:?}" "${MODEL:?}" "${TABLES:?}" "${ROPE:?}" "${IDS:?}" "${OUT:?}" "${ARMS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8095}; GPU=${GPU_INDEX:-0}; PY=${PYTHON:-python}
mkdir -p "$OUT"; trap 'touch "$OUT/.done"' EXIT
smi() { nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits | tr -d ' \r'; }
# Stop the server of one arm without hanging: a native Windows exe ignores the shell's signal, so take its Windows pid at
# stop time (after the exec, from /proc or the pid listening on PORT), force it, then wait with a bound.
stop_server() {
  local spid=$1 wpid="" i
  if command -v taskkill > /dev/null 2>&1; then
    wpid=$(cat "/proc/$spid/winpid" 2>/dev/null || true)
    [ -n "$wpid" ] && taskkill //F //PID "$wpid" > /dev/null 2>&1
    wpid=$(netstat -ano 2>/dev/null | tr -d '\r' | awk -v p=":$PORT" '$1=="TCP" && $2 ~ p"$" && $4=="LISTENING" {print $5; exit}')
    [ -n "$wpid" ] && [ "$wpid" != 0 ] && taskkill //F //PID "$wpid" > /dev/null 2>&1
  fi
  kill "$spid" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$spid" 2>/dev/null || return 0; sleep 1; done
  echo "warning: server pid $spid still alive after 60 s" >&2
}
EXTRA=()
[ -n "${MTP:-}" ] && EXTRA+=(--mtp-sidecar "$MTP")
[ -n "${VOCAB:-}" ] && EXTRA+=(--draft-vocab "$VOCAB")
echo "start $(date +%T) gpu $GPU $(smi)" > "$OUT/summary.txt"
IFS=';;' read -ra LIST <<< "${ARMS// ;; /;;}"
for spec in "${LIST[@]}"; do
  [ -n "$spec" ] || continue
  name=$(echo "$spec" | cut -d'|' -f1 | tr -d ' '); envs=$(echo "$spec" | cut -d'|' -f2); flags=$(echo "$spec" | cut -d'|' -f3)
  A="$OUT/$name"; mkdir -p "$A"
  # shellcheck disable=SC2086
  env $envs "$SERVE" --model "$MODEL" --tables "$TABLES" --rope-table "$ROPE" "${EXTRA[@]}" --port "$PORT" \
    ${COMMON:---max-positions 34816 --http-slots 1} $flags > "$A/serve.log" 2>&1 &
  SPID=$!
  ( while kill -0 $SPID 2>/dev/null; do echo "$(date +%s.%N | cut -c1-14) $(smi)" >> "$A/smi.samples"; sleep 0.5; done ) & MPID=$!
  # shellcheck disable=SC2086
  timeout "${LIMIT_S:-900}" "$PY" -I "$HERE/prefill_client.py" "http://127.0.0.1:$PORT" "$IDS" "$A/client.txt" \
    ${CLIENT_ARGS:---lens 8192,32768 --runs 2} > "$A/client.log" 2>&1; rc=$?
  stop_server "$SPID"; sleep 2; kill $MPID 2>/dev/null
  { echo "== arm $name rc $rc env [$envs] flags [$flags]"
    grep -E 'profile|identity|VRAM after load|rror' "$A/serve.log" | head -4 | cut -c1-240
    grep -E '^pf ' "$A/client.txt" | cut -c1-200
    awk -F, '$2+0>=50{s+=$3; n++} {t++} $1+0>m{m=$1+0} END{printf "busy-sample mean SM MHz %.0f, busy fraction %.2f (n=%d), peak MiB %d\n", n?s/n:0, t?n/t:0, t, m}' \
      <(cut -d' ' -f2 "$A/smi.samples")
  } >> "$OUT/summary.txt"
done
echo "end $(date +%T) $(smi)" >> "$OUT/summary.txt"
