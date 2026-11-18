#!/bin/bash
# Qwen dense layer split over ranks (strix-hip-qwen35-pair), one ticket per call:
#   selfcut   one node, one process: the uncut run, then the same prompt cut at each K (range hooks + hidden export /
#             import); logits bytes and ids must be equal; also writes the one-node reference logits (ref-logits.f32).
#   identity  two ranks (local = rank 0, KIT_PEER = rank 1, the head stage): logits of the last prompt row + every decode
#             step and the ids, compared with the selfcut reference found under --ref on KIT_PEER.
#   timing    two ranks: --repeat 2 prefills per process (cold, warm) + decode, no dump.
# Run it inside the node GPU queue (single node: gpu-turn.sh; two ranks: gpu-turn-pair.sh on the rank-0 node). No host,
# user or model path is built in: binaries, model, prompt and output come from arguments, the peer from KIT_PEER.
# Usage: dense_pair.sh selfcut|identity|timing --bin DIR --model GGUF --ids FILE --out DIR [--ref DIR] [--cuts 32,13]
#        [--gen 32] [--wide 512] [--placement auto|single|cut:A] [--prefill-rate ROWS_PER_S] [--transport usb4auto|tcp]
set -u
mode=${1:-}; shift || true
gen=32 wide=512 cuts=32,13 placement=auto rate=0 ref='' transport=usb4auto bin='' model='' ids='' out=''
while (($#)); do
    case $1 in
    --bin) bin=$2 ;; --model) model=$2 ;; --ids) ids=$2 ;; --out) out=$2 ;; --ref) ref=$2 ;; --cuts) cuts=$2 ;;
    --gen) gen=$2 ;; --wide) wide=$2 ;; --placement) placement=$2 ;; --prefill-rate) rate=$2 ;; --transport) transport=$2 ;;
    *) echo "dense_pair: unknown argument $1" >&2; exit 2 ;;
    esac
    shift 2
done
[[ -n $bin && -n $model && -n $ids && -n $out ]] || { echo "dense_pair: --bin --model --ids --out are required" >&2; exit 2; }
P=$bin/strix-hip-qwen35-pair
mkdir -p "$out"
S=$out/summary-$mode.txt
rm -f "$out/.done-$mode"
common=(--model "$model" --prompt-ids "@$ids" --wide "$wide" --gen "$gen")
q() { printf '%q ' "$@"; }
case $mode in
selfcut)
    timeout 140 "$P" "${common[@]}" --self-cut "$cuts" --dump-logits "$out/ref-logits.f32" > "$out/selfcut.out" 2> "$out/selfcut.err"
    rc=$?
    { echo "selfcut rc=$rc bin $(sha256sum "$P" | cut -c1-16)"; grep -aE '^(load time|self-cut|dumped)' "$out/selfcut.out" | cut -c1-600
      tail -n 3 "$out/selfcut.err" | cut -c1-300; } > "$S" ;;
identity | timing)
    : "${KIT_PEER:?KIT_PEER (ssh alias of the rank-1 node) is required}"
    if [[ $mode == identity ]]; then extra=(--repeat 1 --dump-logits "$out/pair-logits.f32"); else extra=(--repeat 2); fi
    args=("${common[@]}" --placement "$placement" --prefill-rate "$rate" --transport "$transport" "${extra[@]}")
    ssh -n -o BatchMode=yes "$KIT_PEER" "mkdir -p $(q "$out") && cd $(q "$out") || exit 9
        timeout 170 $(q "$P" --rank 1 "${args[@]}") > r1-$mode.out 2> r1-$mode.err; echo rc=\$? > r1-$mode.rc" &
    peer=$!
    timeout 170 "$P" --rank 0 "${args[@]}" > "$out/r0-$mode.out" 2> "$out/r0-$mode.err"
    rc=$?
    wait "$peer"
    {
        echo "$mode r0 rc=$rc r1 $(ssh -n -o BatchMode=yes "$KIT_PEER" "cat $(q "$out/r1-$mode.rc")") bin r0 $(sha256sum "$P" | cut -c1-16) r1 $(ssh -n -o BatchMode=yes "$KIT_PEER" "sha256sum $(q "$P")" | cut -c1-16)"
        grep -avE '^(loaded|mtp)' "$out/r0-$mode.out" | head -n 14 | sed 's/^/r0 /' | cut -c1-400
        ssh -n -o BatchMode=yes "$KIT_PEER" "grep -avE '^(loaded|mtp)' $(q "$out/r1-$mode.out") | head -n 14" | sed 's/^/r1 /' | cut -c1-400
        tail -n 2 "$out/r0-$mode.err" | sed 's/^/r0 err /' | cut -c1-300
        ssh -n -o BatchMode=yes "$KIT_PEER" "tail -n 2 $(q "$out/r1-$mode.err")" | sed 's/^/r1 err /' | cut -c1-300
        if [[ $mode == identity && -n $ref ]]; then
            ssh -n -o BatchMode=yes "$KIT_PEER" "cd $(q "$out") || exit 9
                if cmp -s $(q "$ref/ref-logits.f32") pair-logits.f32; then echo 'logits vs one node: IDENTICAL ('\$(stat -c %s pair-logits.f32)' bytes)'; else echo 'logits vs one node: DIFFER '\$(cmp $(q "$ref/ref-logits.f32") pair-logits.f32 2>&1 | head -n 1); fi
                a=\$(grep -a '^self-cut uncut' $(q "$ref/selfcut.out") | sed 's/.* ids //'); b=\$(grep -a '^generated:' r1-identity.out | sed 's/^generated: //')
                if [ -n \"\$a\" ] && [ \"\$a\" = \"\$b\" ]; then echo 'ids vs one node: IDENTICAL'; else echo 'ids vs one node: DIFFER'; fi"
        fi
    } > "$S" 2>&1 ;;
*) echo "dense_pair: mode must be selfcut, identity or timing" >&2; exit 2 ;;
esac
touch "$out/.done-$mode"
