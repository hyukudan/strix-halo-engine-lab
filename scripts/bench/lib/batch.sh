#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2154
batch_measure() {
    local label=$1 manifest=$2 log="$out/$1.log"
    if [[ ${config_only:-0} == 1 ]]; then effective_config "$label" --print-effective-config "${command[@]}"; return; fi
    if [[ $dry == 1 ]]; then pair_preview "${command[@]}"; return; fi
    local planned_remote_dir=${effective_remote_dirs[$label]}
    pair_run "$log" "${command[@]}"
    py ingest-batch "$out" "$manifest" "$log"
}
batch_serve() {
    if [[ ${config_only:-0} == 1 ]]; then effective_config serve --print-effective-config "${command[@]}"; return; fi
    if [[ $dry == 1 ]]; then
        pair_preview "${command[@]}"
        printf 'HTTP concurrency=%s; output=%s; repeat=%s; warmup=%s\n' "$concurrency" "$gen_tokens" "$repeats" "$warmup"
    else
        local planned_remote_dir=${effective_remote_dirs[serve]}
        serve_measure
    fi
}
profile_dry_run() {
    local out=DETERMINISTIC_PROMPTS plan
    plan=$(py plan "$out" 1)
    local config_only=1
    eval "$plan"
    config_only=0
    eval "$plan"
}
profile_run() {
    local plan
    mkdir -p "$out/native"
    plan=$(py plan "$out" 0)
    local config_only=1
    eval "$plan"
    py initialize "$out"
    config_only=0
    eval "$plan"
}

# The decode app has no resolver. Probe the published rank app and transfer only env.
deepseek_native_decode() {
    local label=$1 manifest=$2
    local -a rank_env=("${rank_env[@]}")
    if [[ ${config_only:-0} == 1 ]]; then
        effective_config "$label" --print-effective-config "$(binary strix-deepseek41-rank)" "$model_dir/DeepSeek-V4.1-Flash-1017-Q2.gguf" --rank 0 --transport usb4auto
        return
    fi
    if [[ $dry == 1 ]]; then
        printf 'Decode environment: resolved rank ENVIRONMENT above (arguments are not copied).\n'
        rank_env=('RESOLVED_RANK_ENV=from-config-print')
    else
        eval "${effective_decode_env[$label]}"
    fi
    # The README DSpark rows were measured with STRIX_DS41_DSPARK_WINDOW=controller; the decode app defaults to
    # fixed:5, so the kit sets it and records it in the reference key.
    rank_env+=(STRIX_DS41_DSPARK_WINDOW=controller)
    batch_measure "$label" "$manifest"
}
