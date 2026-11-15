#!/usr/bin/env bash
# DeepSeek V4.1 pair prefill: length x rows x arm, cold and warm in ONE process pair (--ttft-timing --repeat 2).
#
#   scripts/bench/common/ticket.sh OUT ds41/prefill-pair --bin BIN --model MODEL --ids DIR [--sizes 2K,8K,32K]
#       [--rows 0] [--arm 'NAME|ARGS|ENV' ...] [--rounds 3] [--repeat 2]
#
#   BIN   strix-deepseek41-prefill   DIR  p<N>.csv from common/prompt_ids.py --format csv   --rows 0 = engine default (auto)
# Each run: rank 1 on KIT_PEER, rank 0 here; "ttft run 0" = cold, the last "ttft run k" = warm (ms); the logits
# fnv of rank 0 and rank 1 must agree (--require). ABAB order: round-major over arms.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' model='' ids='' sizes=8K rows=0 arms=() rounds=3 rep=2 pbin='' pmodel='' pids=''
pair_env=${DS41_PAIR_ENV:-STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000}
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --ids) ids=$2 ;; --sizes) sizes=$2 ;; --rows) rows=$2 ;;
        --arm) arms+=("$2") ;; --rounds) rounds=$2 ;; --repeat) rep=$2 ;;
        --peer-bin) pbin=$2 ;; --peer-model) pmodel=$2 ;; --peer-ids) pids=$2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin && -e $model && -d $ids ]] || kit_die 'need --bin, --model and --ids'
((${#arms[@]})) || arms=('exact||')
pbin=${pbin:-$bin} pmodel=${pmodel:-$model} pids=${pids:-$ids}
out=${KIT_OUT:-bench-out}
kit_same_binary "$bin" "$pbin"
read -ra PE <<<"$pair_env"
RW=(); ((rows > 0)) && RW=(--rows "$rows")
IFS=, read -ra S <<<"$sizes"
for ((r = 0; r < rounds; r++)); do
    for s in "${S[@]}"; do
        n=${s^^}; [[ $n == *K ]] && n=$((${n%K} * 1024))
        need=$((40 + (n > 8192 ? 15 : 0) + (n > 20480 ? 30 : 0)))
        for a in "${arms[@]}"; do
            IFS='|' read -r name args envs <<<"$a"; read -ra A <<<"$args"; read -ra E <<<"$envs"
            tag=s$s-$name-$r
            kit_fits "$need" || { echo "== $tag SKIP (left $(kit_left) s < $need)"; continue; }
            export KIT_TAG=$tag KIT_ARM=s$s-$name KIT_ROUND=$r KIT_TOKENS=$n
            export KIT_RECEIPT_ARGS="--need-prefill --prefill-regex ^ttft.run.[0-9]+:.([0-9.]+).ms"
            p1=(env "${PE[@]}" "${E[@]}" "$pbin" "$pmodel" --rank 1 --transport usb4-auto --prompt "$pids/p$n.csv" "${RW[@]}"
                --ttft-timing --ttft-warmup 512 --repeat "$rep" "${A[@]}")
            p0=(env "${PE[@]}" "${E[@]}" "$bin" "$model" --rank 0 --transport usb4-auto --prompt "$ids/p$n.csv" "${RW[@]}"
                --ttft-timing --ttft-warmup 512 --repeat "$rep" "${A[@]}")
            echo "#### $tag args[$args] env[$envs] t+$(($(date +%s) - KIT_T0))s"
            kit_pair_arm "$out/$tag" "$(($(kit_left) - 3))" '^ttft|rank 1|what\(\)' p1 p0
            f0=$(grep -aoE 'fnv [0-9a-f]{8}' "$out/$tag-r0/stderr" 2>/dev/null | tail -1)
            f1=$(kit_peer "grep -aoE 'fnv [0-9a-f]{8}' $(printf %q "${KIT_PEER_OUT:-$out}/$tag-r1/stderr") | tail -1" 2>/dev/null)
            [[ -n $f0 && $f0 == "$f1" ]] && echo "   ranks agree ($f0)" || echo "   ! ranks DIFFER or no fnv (r0 '$f0' r1 '$f1')"
            sleep "${KIT_SETTLE_S:-2}"
        done
    done
done
