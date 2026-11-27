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

## Pack builder source (tests/weight_exl3_pack_tests.cpp)

`strix::weight::PackMatrixSource` has an `exl3` encoding: a tensor named `<base>.trellis` (I16 or U16, shape
`[in/16, out/16, 16*K]`, K in 1..8) binds together with `<base>.suh` (F16 `[in]`) and `<base>.svh` (F16 `[out]`) and
decodes to `dense_rows` (rows = outputs, columns = inputs; both whole 128-blocks). `strix-lowbit-calib-pack` selects it
like any other matrix (`--storage q_proj.trellis=sp4`) and the rest of the tool is unchanged: with `--moments` the
calibrated coders run on the dense rows, without them the existing uncalibrated rule (`--technique rtn`) applies. The
source digest covers the rings and both scale vectors. Pack tensor names are the `.trellis` names.

Conformance (2, 3, 4, 5, 6, 8 bits): bind and decode equal `dense_rows` bit for bit; the SP4 rows (bytes and header) of
the source equal those of the dense FP32 matrix packed directly; the shared SP4 row decoder returns the coder's values;
refusals cover widths that are not 16*K, K outside 1..8, payload and shape disagreement, shapes outside 128-blocks,
missing, mis-sized or mis-typed `suh` / `svh`.

`coefficients_kn` returns the transformed-domain matrix `Q[k * outputs + n]` (no scales, no Hadamard). The FP32 prefill
identity host reference (`src/minimax/rtxpf/fp32_identity_ref.hpp`) expands its dense copy through it; the test proves the
reference and the shared reader give the same floats on all 65536 codebook states and on random matrices at every rate.

## What remains (in order)

1. Calibration moments per role from text already on disk (no downloads), then the SP4 pack of the attention
   projections (and the MiniMax head) with its identity.
2. Consumer: the pair engines' attention projections read the shared grouped SP4 rows (`src/core/resident_q8_rows.hpp`)
   in decode, decode-only beside the trellis image (the Inkling 2 Mini arm is the pattern); ISA loop on the node.
3. Gate v2 per band on the pair with the held-out window set, then ABAB, then FAST / TURBO columns with decode ladder
   rows.
