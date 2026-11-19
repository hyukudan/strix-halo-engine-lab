#!/usr/bin/env bash
# Hy4-flash G2 one-node model identity: 32 greedy ids + first-token logits digest per gate prompt vs the CPU oracle.
#
#   scripts/bench/common/ticket.sh OUT hy4/g2-identity --bin BIN --model PART1.gguf --prompts DIR --ref REF.txt
#       [--class exact|fast] [--args 'EXTRA ENGINE ARGS'] [--env 'K=V ...'] [--cap 140]
#
#   BIN     the Hy4 engine (one rank). Engine contract (gates-plan.md section 3): `BIN MODEL --identity DIR --greedy 32`
#           reads DIR/prompts.sha256 and DIR/<id>.ids, runs every prompt in ONE process (one load), and prints per prompt
#           "hy4 identity prompt=<id> n_prompt=<P> greedy=<32 ids> first_logits_fnv64=<16 hex> class=<exact|fast:NAME>".
#   DIR     check_fixtures.py export --out DIR (the committed tests/hy4_fixtures ids; same sha256 as the manifest).
#   REF.txt the oracle's lines for the same prompts (tests/hy4_fixtures/ref/oracle-identity.txt once produced).
#   --class exact: ids 32/32 AND first_logits_fnv64 equal (bitwise vs hy4-host-ref-v1);
#           fast: ids 32/32 and the engine declares a fast class (the logits distance is G4's job).
# Expected summary lines:  "G2 gate-en PASS ids 32/32 class=..."  "G2 gate-zh PASS ..."  "G2 summary pass=2 fail=0 missing=0"
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' model='' prompts='' ref='' cls=exact extra='' envs='' cap=140
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --prompts) prompts=$2 ;; --ref) ref=$2 ;; --class) cls=$2 ;;
        --args) extra=$2 ;; --env) envs=$2 ;; --cap) cap=$2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin && -e $model && -f $prompts/prompts.sha256 && -f $ref ]] || kit_die 'need --bin, --model, --prompts DIR (exported) and --ref'
[[ $cls == exact || $cls == fast ]] || kit_die '--class exact|fast'
out=${KIT_OUT:-bench-out}
read -ra A <<<"$extra"; read -ra E <<<"$envs"
kit_fits 60 || kit_die "budget $(kit_left) s < 60 s for one load + 2 prompts"
export KIT_TAG=g2 KIT_ARM=g2-$cls KIT_ROUND=0 KIT_TOKENS=0 KIT_EXPECTED=0
kit_clean_env
echo "#### g2 class=$cls bin=$(kit_sha "$bin") t+$(($(date +%s) - KIT_T0))s"
kit_run "$out/g2" "$(( cap < $(kit_left) - 3 ? cap : $(kit_left) - 3 ))" env "${E[@]}" "$bin" "$model" --identity "$prompts" --greedy 32 "${A[@]}"
# "== " lines reach OUT/summary.txt (kit_summary_done)
kit_py ../hy4/check_fixtures.py identity --rung G2 --class "$cls" --ref "$ref" --got "$out/g2/stdout" | tee "$out/g2/verdict.txt" |
    sed 's/^/== /'
