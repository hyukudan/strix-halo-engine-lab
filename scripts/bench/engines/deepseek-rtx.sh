#!/usr/bin/env bash
# Profile globals cross sourced module boundaries.
# shellcheck disable=SC2034,SC2154
set -eu
# Sourced modules exchange profile globals and command arrays with bench.sh.
status=available
pair=0
identity=q2-q80-tierstream-keep64-warm-controller
binaries=(strix-tokenize strix-cuda-ds41-replica strix-cuda-ds41-prefill)
assets=(DeepSeek-V4.1-Flash-1017-Q2.gguf DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf MODEL.receipt.json STATE.bin ds41-draft-vocab)
# Pin provenance: include/strix/deepseek41_model.hpp; state/receipt are deployment inputs.
asset_sha256s=(5365629079214ff28becbbab7c0900ef9865b79c0f0c64d951924bb362ae5b67 d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9 null null null)
asset_bytes=(365713686528 7966294016 null null null)
profile_environment() { rank_env=(); }
rtx_command() {
    local p=$1 n=$2
    command=("$(binary strix-cuda-ds41-replica)" "$model_dir/${assets[0]}" --receipt "$model_dir/MODEL.receipt.json"
        --import-state "$model_dir/STATE.bin" --draft "$model_dir/${assets[1]}"
        --phases "chats0,schats0,chats0" --max-positions 65536
        --seqs-file "$out/$p.tsv" --chats 0 --chat-prompt "$n" --chat-steps "$gen_tokens" --qual)
    [[ $profile != baseline ]] || command+=(--baseline)
}
rtx_prefill_command() {
    local n=$1
    command=("$(binary strix-cuda-ds41-prefill)" "$model_dir/${assets[0]}"
        --prompt-ids "$out/prefill-$n.ids" --tokens "$n" --positions "$n"
        --pass-rows 1024 --repeat "$((repeats + warmup))" --auto-budget --hashes)
}
rtx_execute() {
    local n rep
    if decode_enabled; then
        rtx_command decode 96
        if [[ $dry == 1 ]]; then show_command "${command[@]}"; else measure decode rtx "$gen_tokens" 1; fi
    fi
    if includes prefill; then
        for n in "${context_values[@]}"; do
            rtx_prefill_command "$n"
            if [[ $dry == 1 ]]; then show_command "${command[@]}"
            else measure "prefill-$n" ttft 1 1; fi
        done
    fi
    if [[ $serve == 1 ]]; then
        printf 'SKIP serve: standalone RTX exposes a file queue, no HTTP endpoint.\n' >&2
        if [[ $dry != 1 ]]; then py skipped "$out" 'HTTP: standalone replica serves a file queue; no public HTTP endpoint.'; fi
    fi
}
profile_dry_run() {
    local out=DETERMINISTIC_PROMPTS
    if decode_enabled; then
        rtx_command decode 96
        effective_config decode --print-config "${command[@]}"
    fi
    rtx_execute
}
profile_run() {
    if decode_enabled; then
        rtx_command decode 96
        effective_config decode --print-config "${command[@]}"
        py initialize "$out"
    fi
    rtx_execute
}
