#!/bin/bash
# Qwen HIP FAST gate v2 (tools/strix-gate/README.md "FAST gate v2"): ONE option_qual invocation = one model load,
# sized for one GPU ticket (inner timeout 110 s). Re-run (one ticket each) until every planned arm of every case is in
# OUT/runs.jsonl; the runner resumes. Then, on any CPU:
#   python3 REPO/tools/strix-gate/gate_v2.py qual OUT --ref REF --cand FAST --null NULL --cand-free FASTF --null-free NULLF
# and per lever (localisation): --cand Q --cand-free QF (G/GF, R/RF).
#
# usage: fast-gate-v2.sh REPO BIN MODEL MODEL_RECEIPT MANIFEST BASE_PLAN OUT CASES LENGTH OPTIONS [PRED] [ARMS]
#   OPTIONS  comma list of non-REF arms in run order, e.g. NULL,FAST,FASTF,NULLF (phase A) or Q,G,R,QF,GF,RF (phase B)
#   PRED     option_qual --predict (default LENGTH=60); ARMS defaults to fast-gate-v2-arms.json next to this script
# The binary must contain the route tap (strix-route-file/v1 line on stderr) and the arms must run eagerly (no graph).
# Route files go to OUT/routes (about rows x top-k x 8 bytes per MoE layer per recording arm).
set -u
REPO=$1; BIN=$2; MODEL=$3; RECEIPT=$4; MANIFEST=$5; BASE=$6; OUT=$7; CASES=$8; LEN=$9; OPTS=${10}
PRED=${11:-$LEN=60}; ARMS=${12:-$(cd "$(dirname "$0")" && pwd)/fast-gate-v2-arms.json}
mkdir -p "$OUT"
python3 - "$BASE" "$OUT/plan.json" "$LEN" "$OPTS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
p["lengths"] = [int(sys.argv[3])]
p["ref_repeat_cases"] = []
p["options"] = sys.argv[4].split(",")
p["family"] = p["options"]
json.dump(p, open(sys.argv[2], "w"))
PY
for v in $(env | grep -o "^STRIX_[A-Z0-9_]*"); do unset "$v"; done
ulimit -s 65536
cd "$REPO" || exit 1
timeout 110 python3 -m tools.qualification.option_qual.run --manifest "$MANIFEST" --plan "$OUT/plan.json" --arms "$ARMS" \
  --binary "$BIN" --model "$MODEL" --model-receipt "$RECEIPT" --out "$OUT" --soft-seconds 106 --reserve-seconds 4 \
  --cap-floor 30 --cap-factor 1.5 --predict "$PRED" --no-effective-config --keep-logits --cases $CASES
rc=$?
echo "fast-gate-v2 rc=$rc"
grep -ah "^strix-route-file/v1" "$OUT"/log/*.stderr 2>/dev/null | cut -c1-200 | tail -n 12
exit $rc
