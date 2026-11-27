# Trellis pack -> Strix pack transcoder (spec, first slice)

Several engines run third-party EXL3 trellis packs (GLM-5.3 Flash attention K6, MiniMax M3.1 Flash attention K6 and head
8 bpw, DeepSeek V4.1 Flash dense roles, Nemotron). Their FAST and TURBO tiers read the same weight bytes as EXACT in
decode. A narrower Strix pack form (SP4 / SP3 / IU4 g64) of a role needs the role's dense weight first: the trellis
codes live in a rotated domain and cannot be re-quantised code by code.

## Conversion

The device forward of one EXL3 linear (`tools/trellis_reference.py`: `input_stage`, `core_gemv`, `output_stage`) is

    y = svh (.) H_out( Q^T H_in( suh (.) x ) )

with `Q` the decoded trellis coefficients (inputs x outputs; `src/weight/formats/exl3.cpp` `decode_exl3`, transformed
domain), `suh` / `svh` the binary16 input / output scales and `H` the normalised Sylvester Hadamard over 128-element
blocks (`H H = I`). The dense weight is

    W[n][k] = svh[n] * sum_n' H[n][n'] * sum_q Q[q][n'] * H[q][k] * suh[k]

`strix::weight::exl3_dense::dense_rows` (`include/strix/weight/formats/exl3_dense.hpp`, host, FP32, output-major rows)
computes it in O(inputs x outputs x log 128). The device rounds its stages to FP16, so `W` is the operator those stages
approximate, not their bits: every transcoded role is a lossy lever with its own gate v2 per band, never EXACT.

The Strix pack side is the existing calibrated writer (`strix-lowbit-calib-pack`, `include/strix/weight/host_lowbit.hpp`
and `host_lowbit_gptq.hpp`: grouped SP4 / SP3 tables with activation moments), so the transcoded pack has the same
container, identity and conformance tests as a pack made from a BF16 checkpoint.

## Conformance (tests/weight/exl3_dense_test.cpp)

- the 128-block Hadamard keeps the norm, is its own inverse and maps e0 to the flat vector;
- `W x` equals the staged operator in FP32 on synthetic packs (2, 3, 4, 5, 6, 8 bits; 128 x 256 and 384 x 128);
- the reader's coefficients are the shared registry decoder's (same tile and ring position);
- malformed shapes, short ring streams, rates above 8 bits and nonfinite scales are refused.

## What remains (in order)

1. Source binding: an EXL3 entry in `include/strix/weight/pack_source.hpp` so `strix-lowbit-calib-pack` reads a trellis
   tensor (rings + `suh` + `svh` by the pack's tensor names, given as data in the model TOML) through `dense_rows`.
2. Calibration moments per role from text already on disk (no downloads), then the SP4 pack of the attention projections
   (and the MiniMax head) with its identity.
3. Consumer: the pair engines' attention projections read the shared grouped SP4 rows (`src/core/resident_q8_rows.hpp`)
   in decode, decode-only beside the trellis image (the Inkling 2 Mini arm is the pattern); ISA loop on the node.
4. Gate v2 per band on the pair with the held-out window set, then ABAB, then FAST / TURBO columns with decode ladder
   rows. The private dense decoder in `src/minimax/rtxpf/fp32_identity_ref.hpp` binds to this reader and is deleted.
