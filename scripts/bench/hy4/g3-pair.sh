#!/usr/bin/env bash
# Hy4-flash G3 pair identity: the TP2 pair must print the SAME identity lines as one node (TP decode partial-sum contract:
# partials added in rank order = one oracle, hy4_place-REPORT). Same prompts, same reference file as G2.
#
#   KIT_PEER=<ssh alias> scripts/bench/common/ticket.sh OUT hy4/g3-pair --bin BIN --model PART1.gguf --prompts DIR
#       --ref REF.txt [--class exact|fast] [--args 'EXTRA'] [--env 'K=V ...'] [--peer-bin B] [--peer-model M]
#
# Rank 1 runs on KIT_PEER (same binary sha required, inputs checked readable there before any GPU time), rank 0 here.
# Engine contract: `BIN MODEL --rank R --transport usb4-auto --identity DIR --greedy 32`; rank 1 prints
# "hy4 rank 1 ready" once it waits for rank 0; rank 0 prints the "hy4 identity ..." lines (gates-plan.md section 3).
# Expected summary lines: "G3 gate-en PASS ids 32/32 ..."  "G3 gate-zh PASS ..."  "G3 summary pass=2 fail=0 missing=0"
# and, when --ref is the G2 engine stdout instead of the oracle, the same verdict proves pair == one node.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' model='' prompts='' ref='' cls=exact extra='' envs='' pbin='' pmodel=''
pair_env=${HY4_PAIR_ENV:-STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000}
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --prompts) prompts=$2 ;; --ref) ref=$2 ;; --class) cls=$2 ;;
        --args) extra=$2 ;; --env) envs=$2 ;; --peer-bin) pbin=$2 ;; --peer-model) pmodel=$2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin && -e $model && -f $prompts/prompts.sha256 && -f $ref ]] || kit_die 'need --bin, --model, --prompts DIR (exported) and --ref'
[[ $cls == exact || $cls == fast ]] || kit_die '--class exact|fast'
[[ -n ${KIT_PEER:-} ]] || kit_die 'KIT_PEER (ssh alias of rank 1) is required'
pbin=${pbin:-$bin} pmodel=${pmodel:-$model}
out=${KIT_OUT:-bench-out}
kit_same_binary "$bin" "$pbin"
kit_peer_inputs "$pmodel" "$prompts/prompts.sha256"
read -ra PE <<<"$pair_env"; read -ra A <<<"$extra"; read -ra E <<<"$envs"
kit_fits 70 || kit_die "budget $(kit_left) s < 70 s for a pair load + 2 prompts"
export KIT_TAG=g3 KIT_ARM=g3-$cls KIT_ROUND=0 KIT_TOKENS=0 KIT_EXPECTED=0
kit_clean_env
p1=(env "${PE[@]}" "${E[@]}" "$pbin" "$pmodel" --rank 1 --transport usb4-auto --identity "$prompts" --greedy 32 "${A[@]}")
p0=(env "${PE[@]}" "${E[@]}" "$bin" "$model" --rank 0 --transport usb4-auto --identity "$prompts" --greedy 32 "${A[@]}")
echo "#### g3 class=$cls bin=$(kit_sha "$bin") t+$(($(date +%s) - KIT_T0))s"
kit_pair_arm "$out/g3" "$(($(kit_left) - 3))" '^hy4 rank 1 ready|what\(\)' p1 p0
# "== " lines reach OUT/summary.txt (kit_summary_done)
kit_py ../hy4/check_fixtures.py identity --rung G3 --class "$cls" --ref "$ref" --got "$out/g3-r0/stdout" |
    tee "$out/g3-verdict.txt" | sed 's/^/== /'
