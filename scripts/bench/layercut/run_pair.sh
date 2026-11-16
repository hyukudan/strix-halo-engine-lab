#!/usr/bin/env bash
# Run on rank 0. Paths are arguments and must exist at the same relative location on both nodes.
set -euo pipefail
if (( $# < 8 )); then
    echo "usage: $0 BINARY MODEL_DIR PROMPT.u32 PEER_SSH_HOST PEER_IP BASE_PORT RELAY_PORT K [C=2048] [STEPS=64] [COUNT=2] [OUT_DIR=layercut-out]" >&2
    exit 2
fi
binary=$1 model=$2 prompt=$3 peer=$4 peer_ip=$5 port=$6 relay_port=$7 cut=$8
block=${9:-2048} steps=${10:-64} count=${11:-2} out=${12:-layercut-out}
for value in "$port" "$relay_port" "$cut" "$block" "$steps" "$count"; do
    [[ $value =~ ^[0-9]+$ ]] || { echo 'Expected decimal integer' >&2; exit 2; }
done
(( port > 0 && port < 65535 && relay_port > 0 && relay_port <= 65535 && cut > 0 && cut < 60 && block > 0 && block % 16 == 0 && count > 0 )) || exit 2
bytes=$(wc -c < "$prompt")
(( bytes > 0 && bytes % 4 == 0 )) || { echo 'Prompt must contain complete u32 token IDs' >&2; exit 2; }
capacity=$((bytes / 4 + steps + 1))
mkdir -p -- "$out"
common=(--layercut-listen "cut:$prompt:$port:$steps:$count" --cut "$cut" --cut-block "$block" --cut-act relay --capacity "$capacity")
rank1=(env "STRIX_MM_PREFILL_IDENTITY=${STRIX_MM_PREFILL_IDENTITY:-}" "$binary" "$model" 1 usb4-auto "$out" "${common[@]}" --cut-relay-listen "$relay_port")
printf -v remote_command '%q ' "${rank1[@]}"
printf -v remote_out '%q' "$out"
ssh "$peer" "mkdir -p -- $remote_out && exec $remote_command" >"$out/rank1.log" 2>&1 &
peer_pid=$!
trap 'kill "$peer_pid" 2>/dev/null || true' EXIT INT TERM
"$binary" "$model" 0 usb4-auto "$out" "${common[@]}" --cut-relay "$peer_ip:$relay_port" 2>&1 | tee "$out/rank0.log"
wait "$peer_pid"
trap - EXIT INT TERM
