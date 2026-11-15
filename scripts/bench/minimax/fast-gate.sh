#!/usr/bin/env bash
# MiniMax FAST quality gate, floor-relative: teacher-forced logit dumps of a reference arm, >= 3 (5 recommended)
# null-control arms and the candidate, all on the same binary and prompts, then fast_logit_gate.py.
#
#   GPU (one arm per ticket):
#     ticket.sh OUT minimax/fast-gate run --bin BIN --model MODEL_DIR --arm 'NAME|ARGS|ENV' --sessions a=A.u32,b=B.u32
#         [--capacity 34816] [--rows 256]
#   CPU (after the tickets, no GPU):
#     fast-gate.sh seal    --out OUT --arm NAME --bin BIN --model MODEL_DIR --sessions ...   (both ranks)
#     fast-gate.sh compare --out OUT --ref NAME --cand NAME [--null NAME ...] [--dnll-rule floor] [gate args...]
#
# run: strix-minimax-rank on both ranks (rank 1 on KIT_PEER) with STRIX_SESSION_CACHE=0 STRIX_MM_STATE_FNV=1 and
# the quality dump env (STRIX_MM_QUALITY_OUT/ROWS/SELECT); sessions run in order in one process (first = cold,
# last = warm, "rank 0 <name>: prompt N tokens, TTFT X s"). A run directory is never reused.
# seal/compare: the rules (KL, top-1, dNLL; "floor" = mean + 2 sd of the null arms) are tools/strix-gate/
# fast_logit_gate.py's, not this script's; the engine/profile pairs allowed to use the floor rule are listed there.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
gate=$KIT_ROOT/tools/strix-gate/fast_logit_gate.py
mode=${1:-help}; shift || true
bin='' model='' arm='' sessions='' cap=34816 rows=256 out=${KIT_OUT:-bench-out} ref='' cand='' nulls=() rest=()
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --arm) arm=$2 ;; --sessions) sessions=$2 ;; --capacity) cap=$2 ;;
        --rows) rows=$2 ;; --out) out=$2 ;; --ref) ref=$2 ;; --cand) cand=$2 ;; --null) nulls+=("$2") ;;
        *) rest+=("$1"); shift; continue ;;
    esac
    shift 2
done
IFS='|' read -r name args envs <<<"${arm:-x||}"
read -ra A <<<"$args"; read -ra E <<<"$envs"
SA=() PR=()
IFS=, read -ra SS <<<"$sessions"
for s in "${SS[@]}"; do SA+=(--session "${s%%=*}:${s#*=}:1"); PR+=(--prompt "${s%%=*}=${s#*=}"); done
case $mode in
run)
    [[ -x $bin && -e $model && ${#SS[@]} -gt 0 && -n $name ]] || kit_die 'run needs --bin, --model, --sessions and --arm'
    kit_same_binary "$bin"
    d=$out/$name
    [[ -e $d-r0 ]] && kit_die "run dir $d-r0 exists (never reused)"
    q=(STRIX_SESSION_CACHE=0 STRIX_MM_STATE_FNV=1 STRIX_MM_QUALITY_ROWS=$rows STRIX_MM_QUALITY_SELECT=1)
    peer_out=${KIT_PEER_OUT:-$out}
    p1=(env "${q[@]}" STRIX_MM_QUALITY_OUT="$peer_out/$name-r1/q" "${E[@]}" "$bin" "$model" 1 usb4-auto "$peer_out/$name-r1/w"
        --slots 1 --capacity "$cap" --max-rows 4096 "${SA[@]}" "${A[@]}")
    p0=(env "${q[@]}" STRIX_MM_QUALITY_OUT="$d-r0/q" "${E[@]}" "$bin" "$model" 0 usb4-auto "$d-r0/w"
        --slots 1 --capacity "$cap" --max-rows 4096 "${SA[@]}" "${A[@]}")
    tokens=$(($(stat -c %s "${SS[-1]#*=}") / 4))
    export KIT_TAG=$name KIT_ARM=$name KIT_ROUND=0 KIT_TOKENS=$tokens
    export KIT_RECEIPT_ARGS='--need-prefill --prefill-regex ^rank.0.\S+:.prompt.\d+.tokens,.TTFT.([0-9.]+).s@s --require ranks.AGREE'
    printf '%s\n' "${E[@]}" >"$out/$name.env"
    kit_pair_arm "$d" "$(($(kit_left) - 3))" 'rank 1|what\(\)' p1 p0 ;;
seal)
    [[ -n $name && -n $bin ]] || kit_die 'seal needs --arm and --bin'
    EA=(); while read -r e; do [[ -n $e ]] && EA+=(--env "$e"); done <"$out/$name.env"
    for rk in 0 1; do
        dir=$out/$name-r$rk; ((rk == 1)) && dir=${KIT_PEER_OUT:-$out}/$name-r1
        cmd=("$KIT_PY" -I "$gate" seal --dir "$dir/q" --rank "$rk" --arm "$name" --run-id "$name" --binary "$bin" --rows "$rows" "${PR[@]}" "${EA[@]}" "${rest[@]}")
        if ((rk == 0)); then "${cmd[@]}"; else kit_peer "cd $(printf %q "$(kit_peer_root)") && $(printf '%q ' "${cmd[@]}")"; fi || kit_die "seal rank $rk failed"
    done
    echo "sealed $name" ;;
compare)
    [[ -n $ref && -n $cand ]] || kit_die 'compare needs --ref and --cand'
    NA=(); for n in "${nulls[@]}"; do NA+=(--null-arm "$out/$n-r0/q"); done
    "$KIT_PY" -I "$gate" compare "$out/$ref-r0/q" "$out/$cand-r0/q" "${NA[@]}" "${PR[@]/#--prompt/--tokens}" "${rest[@]}" ;;
*) sed -n '2,16p' "$0"; exit 64 ;;
esac
