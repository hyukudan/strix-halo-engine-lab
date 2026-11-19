#!/bin/bash
# Chart decode kit (qwen35 dense, strix-cuda-qwen35-spec): ONE engine load, short-input prompts from chart/ (English prose,
# Japanese prose, code, thinking on), greedy, N_GEN tokens each. Per prompt: optional serial reference (--check 1, ids
# equality), then the speculative arms of the shared controller (--arms F1,F2 = ArbiterConfig::frontier_every values,
# interleaved --pairs P times). Outputs run.log (serial[], spec[], step[] = verify/draft/host ms per step + model depth
# histogram, EXACT[]), ids/<prompt>.serial.ids and <prompt>.spec-f<F>-p<P>.ids, clk.csv (1 s GPU clock/power samples).
# Usage: chart_decode.sh EXE PACK OUT_DIR PROMPTS(comma: en-prose,ja-prose,code,think) [spec args, e.g. --arms 0,8 --pairs 3 --check 0]
# env: N_GEN (default 1200), GPU_ID (nvidia-smi --id for the clock log; unset = no log). Tokenize first:
#   python tokenize_prompts.py MODEL_DIR chart
set -u
[ $# -ge 4 ] || { echo "usage: chart_decode.sh EXE PACK OUT_DIR PROMPTS [spec args]"; exit 2; }
EXE=$1; PACK=$2; OUT=$3; NAMES=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
winpath() { command -v cygpath > /dev/null && cygpath -m "$1" || echo "$1"; }
files=""
for n in ${NAMES//,/ }; do
  f="$here/chart/$n.ids"
  [ -f "$f" ] || { echo "missing $f (run tokenize_prompts.py MODEL_DIR chart)"; exit 2; }
  files="$files${files:+,}$(winpath "$f")"
done
mkdir -p "$OUT/ids"
if [ -n "${GPU_ID:-}" ]; then
  ( while [ ! -e "$OUT/.end" ]; do
      nvidia-smi --id="$GPU_ID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu \
        --format=csv,noheader,nounits >> "$OUT/clk.csv" 2>/dev/null
      sleep 1
    done ) &
fi
"$EXE" --model "$PACK" --turbo-weights converted --prompt-file "$files" --n-gen "${N_GEN:-1200}" --ids-out "$(winpath "$OUT/ids")" "$@" \
  > "$OUT/run.log" 2> "$OUT/err.log"
rc=$?
touch "$OUT/.end"
wait
exit $rc
