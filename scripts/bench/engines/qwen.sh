#!/usr/bin/env bash
# Profile globals cross sourced module boundaries.
# shellcheck disable=SC2034,SC2154
set -eu
# Sourced modules exchange profile globals and command arrays with bench.sh.
status=available
pair=0
identity=hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1
binaries=(strix-tokenize strix-hip-decode-token)
assets=(Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf draft-vocab)
# Pin provenance: include/strix/mtp_draft_profiles.hpp. Unknown published pins remain null.
asset_sha256s=(8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2 5516d3ce407783a424315b6a2a0d8b06341665bb3541b98e5db0fe944add4c3c null)
asset_bytes=(null 2786568256 null)
profile_environment() {
    rank_env=()
    if [[ $profile == gsq ]]; then
        assets[0]=hybrid-q2.gguf
        asset_sha256s[0]=106603af0bc020222b90ed73ebfa04fd812280f53ec69135202a6cc71fc98b1e
        rank_env=(STRIX_GSQ_Q20_A8=1)
    fi
}
qwen_command() {
    local prompt=$1 generate=$2 context=8192 needed=0
    if [[ $prompt =~ (ctx|prefill-)([0-9]+)$ ]]; then needed=${BASH_REMATCH[2]}; fi
    while ((context < needed + generate)); do context=$((context * 2)); done
    if [[ $dry == 1 ]]; then
        ids=("TOKEN_IDS_$prompt")
    else
        local token_lines
        token_lines=$(py ids-lines "$out/$prompt.ids")
        mapfile -t ids <<<"$token_lines"
    fi
    command=(env "${rank_env[@]}" "$(binary strix-hip-decode-token)" "$model_dir/${assets[0]}" "${ids[@]}"
        --context "$context" --execution-mode eager --residency register --generate "$generate"
        --optimization-profile auto --numerical-profile "$identity")
    # No --gpu-greedy-token: the pinned numerical profile, the known-prefix window and MTP each refuse
    # that alternate selection controller (validate_numerical_profile_cli, known-prefix and MTP checks).
    [[ $context == 8192 ]] || command+=(--unqualified-context)
    [[ $profile != baseline ]] || command+=(--baseline)
}
# The public resolver owns production prefill, including context sizing and tiling.
qwen_prefill_production() {
    command=("$(binary strix-hip-decode-token)" "$model_dir/${assets[0]}" "${ids[@]}" --generate 1)
    # Baseline keeps historical context sizing and is a separate serial population.
    if [[ $profile == baseline ]]; then
        command+=(--baseline)
        local n=${p#prefill-}
        if ((n + 1 > 8192)); then command+=(--context "$((n + 512))" --unqualified-context); fi
    fi
}
# Under --numerical-profile the binary admits a tiled organization only with the full v5 organization
# spelled out (no implicit flag changes).
QWEN_V5_ORGANIZATION=(--parallel-hyper-mix --staged-hyper-projections --fused-staged-hyper-mix
    --typed-fused-hyper-projections --prefetch-typed-hyper-rows --prefetch-q8-hyper-output
    --lane-chain-hyper-output --prefetch-q8-qsa-input --parallel-moe-router --wave-moe-router-topk
    --virtual-q8-gdn-input --prefetch-q8-gdn-input --virtual-q8-gdn-output --deferred-moe-status
    --prefetch-q8-shared-expert --q4k-a8g32-gate-up --grouped-exact-q51-down --virtual-q8-lm-head
    --q51-slot-parallel-down --batched-moe-activation --wave8-q8-gdn-input --wave8-q8-gdn-output
    --prefetch-q8-lm-head --wave8-q8-lm-head --wave8-q8-qsa-output)
qwen_tiled() {
    local tile=${1:-8192}
    [[ $profile == baseline ]] || command+=("${QWEN_V5_ORGANIZATION[@]}")
    command+=(--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows "$tile"
        --known-prefix-workspace-budget-bytes 8589934592
        --known-prefix-host-metadata-budget-bytes 1073741824
        --known-prefix-uma-budget-bytes 137438953472 --known-prefix-host-headroom-bytes 4294967296)
}
qwen_mtp() {
    # MTP calibration rows require the tiled prompt; every MTP launch, including HTTP, is tiled.
    local content=$1 tile=${2:-8192} depth=7 window=8 windows=2,4,8
    [[ $profile != mtp-k* ]] || depth=${profile#mtp-k}
    if ((depth < 3)); then window=2; windows=2; elif ((depth < 7)); then window=4; windows=2,4; fi
    command+=(--mtp --mtp-draft-profile distill-g --mtp-sidecar "$model_dir/${assets[1]}"
        --mtp-target-sha256 "$target_sha" --mtp-target-content "$content"
        --mtp-conversion effective-norm-embedding-first-v1 --mtp-normalized-rounding bf16-rne
        --mtp-projection-rounding bf16-rne --mtp-sidecar-residency register
        --mtp-header-bytes 33554432 --mtp-sidecar-retained-bytes 3221225472
        --mtp-sidecar-setup-bytes 4294967296 --mtp-golden-max-bytes 67108864
        --mtp-replay-host-bytes 67108864 --mtp-runtime-bytes 134217728
        --mtp-target-capture-bytes 16777216 --mtp-policy-id 1 --mtp-candidate-depth "$depth"
        --mtp-calibration-rows 16 --mtp-target-window-rows "$window" --mtp-target-window-rows-set "$windows"
        --mtp-window-arena-bytes 2147483648 --mtp-window-capture-bytes 67108864
        --mtp-window-checkpoint-bytes 8589934592 --mtp-window-total-bytes 17179869184
        --mtp-window-history-intervals 8192 --mtp-window-expert-union --mtp-window-expert-union-bytes 65536
        --mtp-window-prefix --mtp-window-hyper-mix --mtp-window-qsa-rows --mtp-draft-chain
        --mtp-draft-q4 --mtp-draft-vocab "auto:$model_dir/draft-vocab" --mtp-draft-context-vocab 4096)
    qwen_tiled "$tile"
}
qwen_emit() {
    local p=$1 mode=$2 generation=$3 rep=$4
    # Prefill inspection, like execution, receives no kit environment.
    if [[ $mode == ttft ]]; then local rank_env=(); fi
    effective_config "$p-$mode" --print-effective-config "${command[@]}"
    if [[ $dry == 1 ]]; then show_command "${command[@]}"; return; fi
    if ((rep <= warmup)); then
        run_local "$out/$p-$mode-warm-$rep.log" "${command[@]}"
        if [[ $mode == ttft && $profile != baseline ]]; then py qwen-admission "$out/$p-$mode-warm-$rep.log"; fi
    else measure "$p" "$mode" "$generation" "$rep"; fi
}
qwen_execute() {
    local p name rep n content=MEASURED_FRAMED_CONTENT golden target_sha=MEASURED_FILE_SHA256 golden_sha=MEASURED_GOLDEN_SHA256
    if [[ $dry != 1 ]]; then target_sha=$(py model-hash "$model_dir/${assets[0]}"); fi
    if decode_enabled; then
        for n in "${context_values[@]}"; do
            for name in "${prompt_names[@]}"; do
                p=$name-ctx$n
                # Plain mode supplies content identity and a token reference for MTP calibration, on the
                # same tiled organization as the spec arm.
                for ((rep=1; rep<=warmup+repeats; rep++)); do
                    qwen_command "$p" "$gen_tokens"
                    [[ $profile == baseline ]] || qwen_tiled
                    qwen_emit "$p" plain "$gen_tokens" "$rep"
                done
                [[ $profile != plain && $profile != baseline ]] || continue
                qwen_calibrate "$p" "$out/$p-plain-$((warmup+1)).log"
                for ((rep=1; rep<=warmup+repeats; rep++)); do
                    qwen_command "$p" "$gen_tokens"; qwen_mtp "$content"
                    command+=(--mtp-golden "$golden" --mtp-golden-sha256 "$golden_sha")
                    qwen_emit "$p" spec "$gen_tokens" "$rep"
                done
            done
        done
    fi
    if includes prefill; then
        for n in "${context_values[@]}"; do
            p=prefill-$n
            # Production prefill must be admitted automatically; no draft or calibration is needed.
            for ((rep=1; rep<=warmup+repeats; rep++)); do
                qwen_command "$p" 1; qwen_prefill_production
                qwen_emit "$p" ttft 1 "$rep"
            done
        done
    fi
    if [[ $serve == 1 ]]; then qwen_serve; fi
}
# Sets the caller's content, golden and golden_sha from a plain receipt and one calibration launch.
qwen_calibrate() {
    local p=$1 plain_log=$2
    if [[ $dry != 1 ]]; then content=$(py receipt-field "$plain_log" model_content_identity); fi
    golden="$out/$p.golden"
    qwen_command "$p" "$gen_tokens"; qwen_mtp "$content"
    effective_config "$p-calibration" --print-effective-config "${command[@]}" --mtp-golden-dump "$golden"
    if [[ $dry == 1 ]]; then show_command "${command[@]}" --mtp-golden-dump "$golden"
    else
        run_local "$out/$p-calibration.log" "${command[@]}" --mtp-golden-dump "$golden"
        golden_sha=$(py hash "$golden")
    fi
}
# The server accepts only the tiled MTP route, so the HTTP population is the spec population's
# production MTP command (sidecar, pins, depth, windows) plus its startup calibration golden, on the
# same T8192 tiled organization; the server still reports T and cache policy, which must match the claim.
# Its responses report proposed/verified/accepted drafts, so every cell is a positive control.
qwen_serve() {
    if [[ $profile == plain || $profile == baseline ]]; then
        printf '%s\n' "SKIP serve: Qwen HTTP serving runs MTP; profile $profile has no HTTP population" >&2
        if [[ $dry != 1 ]]; then py skipped "$out" "HTTP: Qwen serving runs MTP; profile $profile has no HTTP population"; fi
        return 0
    fi
    local n=${context_values[0]} c p
    for c in "${context_values[@]}"; do ((c >= n)) || n=$c; done
    # The shortest context-expanded first prompt is the calibration prompt; decode already calibrated it.
    p=${prompt_names[0]}-ctx$n
    if decode_enabled; then
        golden="$out/$p.golden"
        if [[ $dry != 1 ]]; then
            content=$(py receipt-field "$out/$p-plain-$((warmup+1)).log" model_content_identity)
            golden_sha=$(py hash "$golden")
        fi
    else
        # Serve-only: one untimed plain launch supplies the content identity, then calibration.
        qwen_command "$p" "$gen_tokens"
        effective_config "$p-plain" --print-effective-config "${command[@]}"
        if [[ $dry == 1 ]]; then show_command "${command[@]}"; else run_local "$out/$p-serve-plain.log" "${command[@]}"; fi
        qwen_calibrate "$p" "$out/$p-serve-plain.log"
    fi
    export BENCH_DRAFT_ROLE=positive-control
    qwen_serve_cell serve 8192
}
# One server per cell; the pinned T is the cell's claim and must match the server's startup receipt.
qwen_serve_cell() {
    local label=$1 tile=$2
    qwen_command "$p" "$gen_tokens"; qwen_mtp "$content" "$tile"
    command+=(--mtp-golden "$golden" --mtp-golden-sha256 "$golden_sha" --http-serve --http-port "${BENCH_PORT:-18880}")
    effective_config "$label" --print-effective-config "${command[@]}"
    if [[ $dry == 1 ]]; then show_command "${command[@]}"; else serve_measure "$label" "$tile"; fi
}
profile_dry_run() { local out=DETERMINISTIC_PROMPTS; qwen_execute; }
profile_run() { qwen_execute; py initialize "$out"; }
