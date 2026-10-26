#!/usr/bin/env bash
set -eu
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export BENCH_ROOT
cd -- "$BENCH_ROOT"
source scripts/bench/lib/common.sh
source scripts/bench/lib/pair.sh
bin_dir='build with spaces'; peer_bin_dir='peer build'
model_dir='models with spaces'; peer_model_dir='peer models'
out=outbox; remote_dir=remote-run
pair_remote_command env "STRIX_DS41_OVERLAY=$model_dir/c-u6.gguf" "$bin_dir/rank" --rank 0 --prompt "$out/prompt.ids"
[[ ${remote_command[1]} == 'STRIX_DS41_OVERLAY=peer models/c-u6.gguf' ]]
[[ ${remote_command[2]} == 'peer build/rank' && ${remote_command[4]} == 1 && ${remote_command[6]} == remote-run/prompt.ids ]]
value="an asset's name with spaces"
roundtrip=$(bash -c "printf '%s' $(remote_quote "$value")")
[[ $roundtrip == "$value" ]]

pair_remote_command "$bin_dir/strix-minimax-rank" "$model_dir/minimax" 0 usb4gpu "$out/native" --session "d0:$out/chat.u32:127"
[[ ${remote_command[0]} == 'peer build/strix-minimax-rank' && ${remote_command[2]} == 1 ]]
[[ ${remote_command[5]} == --session && ${remote_command[6]} == 'd0:remote-run/chat.u32:127' ]]
pair_remote_command "$bin_dir/strix-glm5-rank" "$model_dir/glm" --rank 0 --world 2 --prompt "$out/chat.u32,$out/code.u32"
[[ ${remote_command[3]} == 1 && ${remote_command[7]} == 'remote-run/chat.u32,remote-run/code.u32' ]]
