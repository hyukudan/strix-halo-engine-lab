#!/bin/bash
# Gemma 4 decode FAST gate v2 (tools/strix-gate/README.md "FAST gate v2", --format steps). One engine process per arm and
# prompt-length band (one model load), teacher-forced on each case's recorded continuation through the shared capture
# (STRIX_BITCALC_CASES, apps/bitcalc_measurement.hpp): ROWS decode selections per case, FP32 logits of every row.
#
# usage: fast-gate-v2.sh BIN MODEL ROPE_DIR MANIFEST_DIR OUT ARM LENGTH ROWS CASE [CASE...]
#   MANIFEST_DIR  an option_qual manifest (manifest.json + cases/): every CASE has >= LENGTH prompt tokens; a longer
#                 prompt is cut to LENGTH and the cut tokens lead its continuation (case name <case>.cut<LENGTH>)
#   ARM REF   --profile exact (the reference)
#       NULL  REF + STRIX_DENSE_NULL=0:LENGTH-1:${NULL_ELEM:-7}: one FP16 ulp on the anchor row's embedding (the first decode
#             row; core/null_perturb.hpp), the evaluation-order null of every window
#       any other name: --profile ${CAND_PROFILE:-fast} --allow-pending with the environment in CAND_ENV ("K=V K=V"),
#             e.g. CAND_ENV=STRIX_G4_DECODE_ATTN=g4-dattn-f2
#   Writes OUT/logits/<case>.<ARM>.bin (cases already captured are skipped), OUT/log/<ARM>-L<LENGTH>.{out,err}, OUT/sum.txt.
# Then, on any CPU, per band directory:
#   python3 tools/strix-gate/gate_v2.py qual OUT --ref REF --cand CAND --null NULL [--tier turbo] --json OUT/gate.json
# Environment: TMO (process timeout, s; default 440), ENGINE_ARGS (extra engine options, e.g. "--threads 8").
set -u
[ $# -ge 9 ] || { sed -n '2,18p' "$0" >&2; exit 64; }
BIN=$1; MODEL=$2; ROPE=$3; MAN=$4; OUT=$5; ARM=$6; LEN=$7; ROWS=$8; shift 8
for v in $(env | grep -o "^STRIX_[A-Z0-9_]*"); do unset "$v"; done
mkdir -p "$OUT/logits" "$OUT/log" "$OUT/cases" || exit 1
SUM=$OUT/sum.txt; TAG=$ARM-L$LEN
LIST=$OUT/cases/$TAG.list
python3 -I - "$MAN" "$OUT" "$ARM" "$LEN" "$ROWS" "$LIST" "$@" <<'PY' || exit 2
import json, os, sys
man, out, arm, length, rows, lst = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
want = sys.argv[7:]
cases = {c["case_id"]: c for c in json.load(open(os.path.join(man, "manifest.json")))["cases"]}
lines = []
for cid in want:
    c = cases.get(cid)
    if c is None:
        sys.exit(f"{cid}: not in the manifest")
    ids = [t for t in open(os.path.join(man, c["prompt_file"])).read().replace(",", " ").split()]
    cont = [t for t in open(os.path.join(man, c["cont_file"])).read().replace(",", " ").split()]
    if len(ids) < length:
        sys.exit(f"{cid}: {len(ids)} prompt tokens, the band needs {length} (one null position per process)")
    prompt = os.path.join(man, c["prompt_file"])
    name = cid
    if len(ids) > length:
        # A longer prompt is cut to the band: the cut tokens lead the continuation (still the document's held-out text).
        ids, cont = ids[:length], ids[length:] + cont
        name = f"{cid}.cut{length}"
        prompt = os.path.join(out, "cases", f"{name}.prompt")
        with open(prompt, "w") as f:
            f.write(",".join(ids) + "\n")
    if len(cont) < rows:
        sys.exit(f"{cid}: continuation {len(cont)} < {rows} rows")
    logits = os.path.join(out, "logits", f"{name}.{arm}.bin")
    if os.path.exists(logits):
        continue
    force = os.path.join(out, "cases", f"{name}.r{rows}.force")
    with open(force, "w") as f:
        f.write(",".join(cont[:rows]) + "\n")
    lines.append(f"{force} {logits} {prompt}\n")
open(lst, "w").writelines(lines)
print(len(lines))
PY
N=$(grep -c . "$LIST")
[ "$N" -gt 0 ] || { echo "$TAG: every case already captured" >> "$SUM"; exit 0; }
E=(); P=(--profile exact)
case $ARM in
  REF) ;;
  NULL) E=(STRIX_DENSE_NULL=0:$((LEN - 1)):${NULL_ELEM:-7}) ;;
  *) P=(--profile "${CAND_PROFILE:-fast}" --allow-pending); read -r -a E <<< "${CAND_ENV:-}" ;;
esac
T0=$(date +%s)
# shellcheck disable=SC2086
env "${E[@]}" STRIX_BITCALC_CASES="$LIST" timeout -k 5 "${TMO:-440}" "$BIN" --model "$MODEL" --backend device \
  --rope-tables "$ROPE" "${P[@]}" --draft off --dflash off --ignore-eos --max-tokens "$ROWS" --output json ${ENGINE_ARGS:-} \
  > "$OUT/log/$TAG.out" 2> "$OUT/log/$TAG.err"
RC=$?
python3 -I - "$OUT/log/$TAG.out" "$OUT/log/$TAG.err" "$TAG" "$RC" "$(( $(date +%s) - T0 ))" "$N" "$ROWS" >> "$SUM" <<'PY'
import json, sys
out, err, tag, rc, wall, n, rows = sys.argv[1:8]
recs = []
for line in open(out, errors="replace"):
    line = line.strip()
    if line.startswith("{"):
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
forced = [r.get("bitcalc_forced_steps") for r in recs]
dec = [r["timings"]["decode_ms"] for r in recs if "timings" in r]
pre = [r["timings"]["prefill_ms"] for r in recs if "timings" in r]
nulls = sum(1 for l in open(err, errors="replace") if "gate null control applied" in l)
ok = len(recs) == int(n) and all(f == int(rows) for f in forced)
tok = (len(dec) * (int(rows) - 1)) / (sum(dec) / 1000) if dec and sum(dec) > 0 else 0.0
print(f"{tag} rc={rc} wall={wall}s cases={len(recs)}/{n} forced_ok={ok} null_flips={nulls} "
      f"prefill_ms_mean={sum(pre) / max(len(pre), 1):.0f} decode_tok_s={tok:.2f}")
bad = [l.strip()[:200] for l in open(err, errors="replace") if "error" in l.lower() or "what()" in l]
for l in bad[:3]:
    print(f"  {tag} stderr: {l}")
PY
exit $RC
