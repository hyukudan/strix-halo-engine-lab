#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2154
set -eu
status=available
pair=1
identity=exl3-4.05-usb4gpu
binaries=(strix-glm5-rank)
assets=(glm)
asset_sha256s=(null)
asset_bytes=(null)
profile_environment() { rank_env=(); [[ $profile != baseline ]] || rank_env+=(STRIX_GLM_BASELINE=1); }
source "$BENCH_ROOT/scripts/bench/lib/batch.sh"
