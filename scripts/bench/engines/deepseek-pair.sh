#!/usr/bin/env bash
# Profile globals cross sourced module boundaries.
# shellcheck disable=SC2034,SC2154
set -eu
# Sourced modules exchange profile globals and command arrays with bench.sh.
status=available
pair=1
identity=q2-u6-trellis-mixB-usb4gpu
binaries=(strix-tokenize strix-deepseek41-rank strix-deepseek41-dspark-rank strix-deepseek41-serve strix-deepseek41-prefill)
assets=(DeepSeek-V4.1-Flash-1017-Q2.gguf DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf c-u6.gguf ours-K4 ours-K5 vocab)
# Pin provenance: include/strix/deepseek41_model.hpp; shared enforcement in lib/kit.py.
asset_sha256s=(5365629079214ff28becbbab7c0900ef9865b79c0f0c64d951924bb362ae5b67 d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9 67e6e10ed97bff78c125e4e1433c3cb8dba4c0e3e2b2d42f9595e251b6fa2122 null null null)
asset_bytes=(365713686528 7966294016 null null null null)
profile_environment() {
    rank_env=("STRIX_DS41_PACKS=$model_dir")
    [[ $profile != baseline ]] || rank_env+=(STRIX_DS41_BASELINE=1)
}
source "$BENCH_ROOT/scripts/bench/lib/batch.sh"
