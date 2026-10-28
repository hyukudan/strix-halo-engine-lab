#!/usr/bin/env bash
# Native gfx1151 gate. First argument is the freshly built benchmark; remaining
# arguments select rank/model assets (or synthetic weights when --model is absent).
# Each configuration starts a new process for startup-cached selectors.
set -euo pipefail
bench=${1:?usage: $0 BENCH [--rank R] [--model MODEL_DIR --layer L]}
shift
# Optional subsets for time-boxed runs: GATE_OVERLAPS="1 0", GATE_DEPTHS="4 8 12" (defaults: all).
for overlap in ${GATE_OVERLAPS:-1 0}; do
    for depth in ${GATE_DEPTHS:-4 8 12}; do
        for wide in 0 1 2; do
            for bits in 3 5; do
                for waves in 4 8; do
                    mode=--diverse-fallback
                    # WIDE=2 (production) widens only N=2..4, so eight-row steps stay narrow and qualify.
                    if [[ $depth == 4 && $wide != 1 && $bits == 3 && $waves == 4 ]]; then mode=--diverse; fi
                    # K5 fallback uses synthetic weights; loaded assets retain their own bits.
                    if [[ $bits == 5 && " $* " == *" --model "* ]]; then continue; fi
                    STRIX_MMX_MOE_V=4 STRIX_MMX_MOE_DEPTH3=$depth STRIX_MMX_MOE_DEPTH8R=0 \
                    STRIX_MMX_MOE_OVERLAP=$overlap STRIX_MMX_MOE_WIDE=$wide STRIX_MMX_MOE_W=$waves \
                        "$bench" "$mode" --test --syn-bits "$bits" "$@"
                done
            done
        done
    done
done
