#!/bin/bash
# chunkinv-attn-state-v1 GPU fixtures: fixtures.sh BUILD_DIR OUT_DIR
# GDN FAST open-tail contract (rev 2/3) and the dense flash split identities (block share vs dense-flash-absseg-v1).
# Run inside the node's GPU turn wrapper; prints one line per fixture.
set -u
B=${1:?build dir}; O=${2:?out dir}; mkdir -p "$O"
for rev in 1 3; do STRIX_QPF2K_GDN_FP16_CHUNK=$rev timeout 60 "$B/strix-hip-gdn-fp16-chunk-tests" > "$O/gdn-rev$rev.txt" 2>&1
  echo "gdn rev $rev rc $? | $(grep 'open tail:' "$O/gdn-rev$rev.txt")"; done
for n in ${CHUNKINV_KEYS:-9000 20000}; do
  for mode in 0 1; do STRIX_G4_PREFILL_FLASH_ABSSEG=$mode timeout 60 "$B/strix-gemma4-prefill-attention-device-tests" --chunkinv "$n" > "$O/g4-$mode-$n.txt" 2>&1
    echo "g4 absseg=$mode n $n rc $?: $(grep 'aligned plans' "$O/g4-$mode-$n.txt")"; done
done
