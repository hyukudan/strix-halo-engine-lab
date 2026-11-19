#!/usr/bin/env bash
# Hy4-flash G5 prefill receipts (pair TP2 or one node): length x arm, cold and warm in ONE process (--repeat 2).
# Binding of the shared runner (common/runlib.sh kit_pair_arm / kit_run, receipt.py, abab.py); no new harness.
#
#   [KIT_PEER=<alias>] scripts/bench/common/ticket.sh OUT hy4/g5-prefill --bin BIN --model PART1.gguf --ids DIR
#       [--sizes 4K,16K] [--arm 'NAME|ARGS|ENV' ...] [--rounds 1] [--repeat 2] [--one-node] [--peer-bin B]
#
#   DIR  p<N>.ids from common/prompt_ids.py --tokenize-cmd 'strix-tokenize PART1.gguf encode' --format ids
#        --sizes 4096,16384,32768 (seed scripts/bench/prompts/seed.txt; same sha on both nodes).
# Engine contract (gates-plan.md section 3): `BIN MODEL [--rank R --transport usb4-auto] --prompt-ids FILE --repeat K
#   --max-new 1` prints per run on stderr "prefill-wall-ms=<ms>" (receipt.py default) and on stdout
#   "hy4 prefill run <k>: <N> tokens wall <ms> ms logits_fnv64 <16 hex>"; run 0 = cold, the last = warm.
# Budget (150 s ticket incl. load, PREDICTION at the 799/717/707 tok/s target): 4K+16K in one ticket (~6 + ~23 s per
# run, x2), 32K alone (~46 s per run, x2). ABAB: arms alternate inside a round; 3 rounds = 3 tickets for one arm.
# Expected: one "== s<size>-<arm>-<round> ... pf_ms warm=... cold=... tok/s=... sclk=... Gclk/tok=..." line per run
# (receipt.py) in OUT/summary.txt, rank 0 and rank 1 logits_fnv64 equal (pair).
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' model='' ids='' sizes=4K,16K arms=() rounds=1 rep=2 one=0 pbin=''
pair_env=${HY4_PAIR_ENV:-STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000}
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --ids) ids=$2 ;; --sizes) sizes=$2 ;; --arm) arms+=("$2") ;;
        --rounds) rounds=$2 ;; --repeat) rep=$2 ;; --peer-bin) pbin=$2 ;;
        --one-node) one=1; shift; continue ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin && -e $model && -d $ids ]] || kit_die 'need --bin, --model and --ids'
((${#arms[@]})) || arms=('exact||')
out=${KIT_OUT:-bench-out}
if ((one == 0)); then
    [[ -n ${KIT_PEER:-} ]] || kit_die 'pair run needs KIT_PEER (or --one-node)'
    pbin=${pbin:-$bin}
    kit_same_binary "$bin" "$pbin"
    kit_peer_inputs "$model" "$ids"
fi
read -ra PE <<<"$pair_env"
IFS=, read -ra S <<<"$sizes"
kit_clean_env
for ((r = 0; r < rounds; r++)); do
    for s in "${S[@]}"; do
        n=${s^^}; [[ $n == *K ]] && n=$((${n%K} * 1024))
        need=$((45 + rep * n / 700 / (2 - one)))   # load + rep prefills at >= 700 tok/s (PREDICTION) per rank pair
        for a in "${arms[@]}"; do
            IFS='|' read -r name args envs <<<"$a"; read -ra A <<<"$args"; read -ra E <<<"$envs"
            tag=s$s-$name-$r
            kit_fits "$need" || { echo "== $tag SKIP (left $(kit_left) s < $need)"; continue; }
            export KIT_TAG=$tag KIT_ARM=s$s-$name KIT_ROUND=$r KIT_TOKENS=$n KIT_EXPECTED=$rep
            export KIT_RECEIPT_ARGS="--need-prefill"
            echo "#### $tag args[$args] env[$envs] t+$(($(date +%s) - KIT_T0))s"
            if ((one)); then
                bash "$here/../common/run1.sh" "$out/$tag" "$(($(kit_left) - 3))" -- env "${E[@]}" "$bin" "$model" \
                    --prompt-ids "$ids/p$n.ids" --repeat "$rep" --max-new 1 "${A[@]}"
            else
                p1=(env "${PE[@]}" "${E[@]}" "$pbin" "$model" --rank 1 --transport usb4-auto --prompt-ids "$ids/p$n.ids"
                    --repeat "$rep" --max-new 1 "${A[@]}")
                p0=(env "${PE[@]}" "${E[@]}" "$bin" "$model" --rank 0 --transport usb4-auto --prompt-ids "$ids/p$n.ids"
                    --repeat "$rep" --max-new 1 "${A[@]}")
                kit_pair_arm "$out/$tag" "$(($(kit_left) - 3))" '^hy4 rank 1 ready|what\(\)' p1 p0
            fi
            sleep "${KIT_SETTLE_S:-2}"
        done
    done
done
