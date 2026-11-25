#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/gate-ticket.sh"
ticket=$(mktemp)
trap 'rm -f -- "$ticket"' EXIT
printf '%s\n' 'decode 64 abc tier=turbo policy=weights.head.q4_g64@1 storage=gguf.q4_g64@1 route=row' > "$ticket"
strix_gate_ticket_matches "$ticket" 'decode 64 abc tier=turbo policy=weights.head.q4_g64@1 storage=gguf.q4_g64@1 route=row'
if strix_gate_ticket_matches "$ticket" 'decode 64 abc tier=fast policy=weights.head.q4_g64@1 storage=gguf.q4_g64@1 route=decode-arm'; then
    echo 'ticket accepted a mismatching reducer tier and route' >&2
    exit 1
fi
echo 'PASS: ticket binds candidate tier, policy, storage and route'
