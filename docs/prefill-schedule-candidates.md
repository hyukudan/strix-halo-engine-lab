# Prefill schedule candidates

These opt-in schedules retain their existing arithmetic contracts on gfx1151.
Native qualification covers byte comparisons of all schedules, a production-base
bridge, and full-vocabulary logits plus all seven physical state components.
Defaults (integrate/qwen-w5, ON-by-default rule 2026-10-23 22:03, exact and never slower in the
combined end-to-end ABA): `STRIX_INTPREFILL_DENSE_F16_TUNE=4` and `STRIX_INTPREFILL_MOE_TUNE=82`;
set either to 0 for the previous schedules.
They apply only to the existing many-row prefill projections. They retain the
selected numerical profile, weight codes, scale boundaries and accumulator order.

| Selector | Values | Schedule |
|---|---|---|
| `STRIX_INTPREFILL_MOE_TUNE` | 0..31, 64..95 | Bit 0: one register-prefetch stage instead of two; bit 1: batch-major, 128-assignment CSR batches (16 waves) when the prompt averages at least 96 assignments per expert, so one dequantized weight tile serves up to 128 rows of its expert; bit 2: compact 128-column tile with all the batch rows; bit 3: unpadded XOR-permuted LDS rows; bit 4: the prologue writes each prompt row's scaled FP16 image once and gate/up copies it; bit 6: short exact dequantization (FP16-subnormal codes through `v_fma_mix`, admitted per block when every value is an exact FP32 value, base conversion otherwise; Q8_0 always base). FP16 routed prefill only. Values >= 256 are timing diagnostics, not exact. |
| `STRIX_INTPREFILL_DENSE_F16_TUNE` | 0..7 | Bit 0: 64-half LDS rows XOR-permuted by row % 8 instead of the 72-half pitch; bit 1: output tile varies fastest; bit 2: pipelined K loop (two register stages of global loads in flight, LDS-only barriers, no private arrays; tiles of up to 128 rows). Existing FP16 tile selection still applies. |
| `STRIX_INTPREFILL_DENSE_A8_SINGLE_LDS` | 0 or 1 | One shared activation image instead of two in the existing staged IU8 kernel. An extra CTA barrier protects readers before overwrite. |

The dense IU8 schedule uses the original signed Q8_0 codes and independent
32-feature weight scales. Its activation quantizer runs once per input projection,
before the grid of output tiles. It retains the ascending per-block FP32 FMA
epilogue. It does not requantize weights to a common scale. The MoE IU8 kernels
already expand the packed linear codes into unsigned byte operands, reuse the
prefill prologue's A8 rows across output tiles, and fuse SwiGLU with down-input
preparation. Scale and minimum corrections remain necessary for affine blocks.

To study integer arithmetic under `fast-prefill-f16wmma-v1`, use the existing
`--prefill-fast-components` identity: omitting `moe` selects its A8 expert path;
omitting `dense` and `dense8` selects the original Q8_0/A8 dense path. For example,
`hc,qsa,router` selects both integer families. This changes prefill numerics and
requires a separate model-quality gate. Do not treat its equality to an A8
component reference as equality to FP16 prefill. `dense8` is a distinct weight
requantization experiment and is not part of these schedules.

No A4 activation path is introduced. IU4 requires both matrix operands to have
valid four-bit codes; a four-bit weight format alone does not make A8 activations
eligible. The standalone integer-projection laboratory is not used by these
prefill schedules.

Qualification must compare complete written intermediates and logical model
state, every logit row and greedy continuation against the current prefill, with
decode using its existing profile. Integer quality arms additionally require
full-vocabulary next-token KL and top-1 agreement on the fixed prompt suite and
agent trace before adoption. Compiler resource estimates do not qualify speed
or correctness. Diagnostic fixtures are `strix-hip-moe-prefill-schedule-tests`
and `strix-hip-prefill-dense-schedule-tests`; execute HIP tests only under the
operator's GPU scheduling protocol.

## Native qualification

The native schedule fixture compares all seven nonzero MoE combinations to zero
across 140 cases (linear weight formats, nonfinite inputs, duplicate routes,
imbalanced routing, full and partial CSR batches). The dense fixture covers
32/64/128/256 rows, all eligible FP16 tiles, both input storage types, output
tails, and four/eight-wave single/double-LDS IU8 projections. These comparisons
retain every output byte rather than applying a numerical tolerance.

The model prefix diagnostic also retains the initial committed and full physical
state before its serial oracle runs. Its later receipts include every physical
window logit row, committed state and discarded state tails. It uses the existing
Qwen R1/R2/R4/R8 contracts; it does not introduce a new decode window size.
All five schedules keep the per-output arithmetic order. Choosing A8 instead of
FP16, or enabling the separate dense8 weight requantizer, changes numerics and
is outside this qualification. No new GEMM backend or reduction tree is adopted.

Resource improvements are hypotheses for a performance campaign, not speedups.
On the native target the compact MoE tile halves LDS from 46848 to 23424 bytes;
single-LDS dense A8 halves its staging allocation from 9216 to 4608 bytes.
Native ELF register-spill counters remain zero and the audited kernels use wave32.

Native model qualification used the published production Qwen source and its
existing replica environment. For code at 1024 tokens and prose at 8192 tokens,
production base, new selectors OFF, and MoE=7/dense-FP16=3 have identical complete
prefix receipts and terminal full-vocabulary logit digests. Single-LDS is
separately byte-identical to double-LDS on the existing original-Q8/A8 dense
prefill path at 1024 tokens. That comparison does not establish FP16/A8 quality
equivalence or promote a different numerical profile. Each prefix run uses the
serial oracle and 22 acceptance, rejection and owner-transition cases.

## Redo measurements (gfx1151)

VALU and WMMA instructions do not overlap on gfx1151 (8 WMMA + 256 FMA per iteration take the sum of their separate
times), and the f16 WMMA peak is 55 TFLOP/s with register or LDS operands. The FP16 prefill projections therefore pay
every staging instruction in WMMA time. Dense bit 2 removes the per-stage drain of the global loads that
`__syncthreads` imposed. MoE bits 1, 4 and 6 remove dequantization and conversion instructions per useful product.
Staging the next step during the multiply (a second LDS image, or a register-only convert-ahead) is exact but slower,
since VALU and WMMA serialize regardless and the WMMA operand reads wait on lgkmcnt behind the staging stores.
