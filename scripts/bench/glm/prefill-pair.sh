#!/usr/bin/env bash
# GLM pair prefill: exact vs FAST (or any arm), several prompts in ONE load (--pf-ttft), first = cold, last = warm.
#
#   scripts/bench/common/ticket.sh OUT glm/prefill-pair --bin BIN --model MODEL_DIR --prompts A.u32,B.u32[,...]
#       [--arm 'NAME|ARGS|ENV' ...] [--rounds 3] [--capacity 34816] [--max-rows 8192]
#
#   BIN   strix-glm5-rank          prompts: little-endian uint32 files (common/prompt_ids.py --format u32 with the
#         model's tokenizer.json); list the same file twice to get a cold and a warm figure of one length.
#   arms  default 'exact||' and 'fast|--profile fast|'. Rank 1 runs on KIT_PEER.
# Each run prints "pf <name>-ttft: N tokens X s ... logits fnv H, state fnv H"; the receipt takes the first as cold
# and the last as warm; the logits/state fnv of every prompt must match between rank 0 and rank 1.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' model='' prompts='' arms=() rounds=3 cap=34816 rows=8192 pbin='' pmodel='' pprompts=''
pair_env=${GLM_PAIR_ENV:-STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000 HSA_DISABLE_FRAGMENT_ALLOCATOR=1}
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --prompts) prompts=$2 ;; --arm) arms+=("$2") ;; --rounds) rounds=$2 ;;
        --capacity) cap=$2 ;; --max-rows) rows=$2 ;; --peer-bin) pbin=$2 ;; --peer-model) pmodel=$2 ;; --peer-prompts) pprompts=$2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin && -e $model && -n $prompts ]] || kit_die 'need --bin, --model and --prompts'
((${#arms[@]})) || arms=('exact||' 'fast|--profile fast|')
pbin=${pbin:-$bin} pmodel=${pmodel:-$model} pprompts=${pprompts:-$prompts}
out=${KIT_OUT:-bench-out}
kit_same_binary "$bin" "$pbin"
read -ra PE <<<"$pair_env"
tokens=$(($(stat -c %s "${prompts##*,}") / 4))
for ((r = 0; r < rounds; r++)); do
    for a in "${arms[@]}"; do
        IFS='|' read -r name args envs <<<"$a"; read -ra A <<<"$args"; read -ra E <<<"$envs"
        tag=$name-$r
        kit_fits 60 || { echo "== $tag SKIP (left $(kit_left) s)"; continue; }
        export KIT_TAG=$tag KIT_ARM=$name KIT_ROUND=$r KIT_TOKENS=$tokens
        export KIT_RECEIPT_ARGS='--need-prefill --prefill-regex ^pf.\S+-ttft:.\d+.tokens.([0-9.]+).s@s'
        base=(--world 2 --transport usb4-auto --mtp 0 --slots 1 --capacity "$cap" --max-rows "$rows" --pf-ttft-mode gemv)
        p1=(env "${PE[@]}" "${E[@]}" "$pbin" "$pmodel" --rank 1 "${base[@]}" --pf-ttft "$pprompts" "${A[@]}")
        p0=(env "${PE[@]}" "${E[@]}" "$bin" "$model" --rank 0 "${base[@]}" --pf-ttft "$prompts" "${A[@]}")
        echo "#### $tag args[$args] env[$envs] t+$(($(date +%s) - KIT_T0))s"
        kit_pair_arm "$out/$tag" "$(($(kit_left) - 3))" '^pf |rank 1|what\(\)' p1 p0
        f0=$(grep -aoE 'logits fnv \w+, state fnv \w+' "$out/$tag-r0/stderr" "$out/$tag-r0/stdout" 2>/dev/null | cut -d: -f2- | md5sum | cut -c1-8)
        f1=$(kit_peer "cat $(printf %q "${KIT_PEER_OUT:-$out}/$tag-r1/stderr") $(printf %q "${KIT_PEER_OUT:-$out}/$tag-r1/stdout") 2>/dev/null | grep -aoE 'logits fnv \w+, state fnv \w+' | md5sum | cut -c1-8")
        [[ $f0 == "$f1" ]] && echo "   ranks agree (fnv digest $f0)" || echo "   ! ranks DIFFER (r0 $f0 r1 $f1)"
        sleep "${KIT_SETTLE_S:-2}"
    done
done
