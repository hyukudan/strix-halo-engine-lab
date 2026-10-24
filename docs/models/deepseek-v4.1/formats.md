# deepseek41 weight and cache formats (decode)

Status: format layer only. Decoders, the A8 activation contract and the decode products of
routed-expert rows for gfx1151, with host references that define every result bit. No model
execution uses them yet.

Files: `include/strix/ds41_formats.hpp` and `src/ds41_formats.cpp` (host), `include/strix/hip_ds41_formats.hpp`
and `src/hip_ds41_formats.hip` (device), `include/strix/ds41_iq2_xxs_grid.inc` (IQ2_XXS grid, shared by both),
`tests/ds41_formats_tests.cpp` (CPU) and `tests/hip_ds41_formats_tests.hip` (device). Both libraries are separate
targets (`strix_ds41_formats`, `strix_ds41_hip`) built with `-ffp-contract=off`, IEEE denormals and correctly
rounded division; no existing target changes.

## Formats

| Format | Use | Block | Integer form of 32 weights |
| --- | --- | --- | --- |
| IQ2_XXS (type 16) | routed gate/up | 66 B / 256: FP16 `d`, then 8 groups of two u32 (four 8-bit grid indices; four 7-bit sign indices and a 4-bit scale `s`) | codes in {±8, ±25, ±43} (grid magnitudes with signs), scale `d * (0.5 + s) * 0.25` |
| Q2_K (type 10) | routed down | 84 B / 256: 16 scale bytes (low nibble scale, high nibble min), 64 code bytes, FP16 `d`, `dmin` | 2-bit codes q, value `d * sc * q - dmin * m` per 16-weight half |
| MXFP4 (type 39) | draft-stage routed experts | 17 B / 32: E8M0 exponent, 16 bytes of E2M1 nibbles (low nibbles are elements 0-15) | doubled E2M1 codes {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}, scale 2^(e - 128) |
| E4M3 + E8M0 rows | Engram tables | 264 B: 256 E4M3 codes, 8 E8M0 scales (one per 32) | value = E4M3 × 2^(e - 127) |

IQ2_XXS sign bytes are the 7-bit index with its parity as bit 7 (an even number of negative elements); the
device computes them instead of reading a table, and applies them to the grid bytes as `(g ^ m) + (m & 0x01010101)`
(no carries: magnitudes are at most 43). MXFP4 codes come from two `v_perm_b32` table selections. E8M0 code 0 is
2^-127 (2^-128 for the halved form) and 255 is NaN; E4M3 is the finite variant (S.1111.111 is NaN, ±448 maximum).

## A8 activations

Groups of 32 values: `delta = amax / 127` (correctly rounded), `code = rint(x / delta)` clamped to ±127; a zero
group or one with amax below 2^-119 has delta 1 and zero codes; the group's code sum is kept. This is the A8
contract of the existing routed-expert A8 kernels.

## Declared products

One wave per output row. Lane l accumulates the 32-weight groups l, l + 32, ... in ascending order, starting from
zero, with the group term

- IQ2_XXS, MXFP4: `float(Σ code·x) * (scale * delta)`;
- Q2_K: `(d * float(A) - dmin * float(B)) * delta`, with `A = Σ_h (sc_h & 15)·Σ q·x` and `B = Σ_h (sc_h >> 4)·Σ x`
  over the group's two 16-weight halves;

and the lanes combine as `v[l] = v[l] + v[l ^ d]` for d = 16, 8, 4, 2, 1 (the result is lane 0). The integer
sums are exact (`v_dot4_i32_iu8`), so the only roundings are the listed FP32 operations. `product_iq2_xxs`,
`product_q2_k` and `product_mxfp4` evaluate exactly this on the host. Products take 1 to 8 activation rows and a
list of experts in one launch, or the window routes of ds41-moe.md.

## Qualification

- CPU: table invariants (grid magnitudes, sign parity), E4M3/E8M0/E2M1 values, E4M3 encoding (round to nearest
  even, saturation, NaN), hand-built blocks, products against a double-precision dot of the decoded weights, A8
  rounding and tiny groups.
- Device, bitwise against the host: all scalar table codes; synthetic blocks of the three formats (finite FP16
  scales including subnormals, the whole E8M0 range); the A8 quantizer (codes, deltas, sums); products for 1, 3
  and 8 rows over three experts in reversed order; and real blocks of the pinned files when `STRIX_DS41_Q2_GGUF`
  and `STRIX_DS41_DSPARK_GGUF` are set (whole experts 0/191/383 of the gate, up and down tensors of layers 0, 1, 20
  and 39; experts 0/64/127 of the three draft stages; 4096 random rows of each Engram table).
- `STRIX_DS41_BENCH=1` times whole routed-expert windows with cold caches; see `ds41-moe.md`, which also covers
  the window kernels that now implement these products (activation rows staged in LDS, several weight rows per
  wave, each weight group decoded once for every row). One-row products read at the scattered-read rate of the
  device (about 220 GB/s).

## Native FP8 dense weights (optional, default off)

The original checkpoint stores the dense projections (attention `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`; shared expert
`w1`, `w3`, `w2`) as `F8_E4M3` codes `[rows][width]` with one `F8_E8M0` exponent per 32 x 32 block (scale tensor
`[rows / 32][width / 32]`, `quantization_config.weight_block_size = [32, 32]`, `scale_fmt = ue8m0`); the output head is
BF16 there. The model file holds Q8_0 requantizations of these tensors (relative RMS error about 5.4e-3 against the
FP8 values, measured on layers 0-2 with `tools/ds41_native_fp8/compare_q8.py`). `ds41_native_fp8.hpp` reads the
original codes instead: exact weights at 8.008 bits per weight (Q8_0: 8.5).

Every E4M3 value is an integer multiple of 2^-9, `N = e4m3 * 2^9` with `|N| <= 229376`, so the dot of a 32-weight group
with its A8 codes is an exact int32 (`|D| < 2^30`; gfx1151 has no FP8 dot instruction, the codes are decoded by integer
operations and summed from 24-bit products). The group term is `float(D) * (2^(e - 136) * delta)` (exponents
restricted to [10, 254] so the scale is a normal FP32 power of two; the checkpoint uses 114-119) with the lane order
and butterfly of the declared products above. Identities: `e4m3-b32x32-a8-int-product-v1` (q_a, q_b, kv),
`a8+e4m3-b32x32-a8-int-product-grouped8-v1` (attn_output_a), `e4m3-b32x32-a8-int-ksplit2-h0+h1-v1` (attn_output_b) and
`e4m3-b32x32-a8-int-swiglu-clamp10-v1` (shared expert: gate and up products, `swiglu_clamped`, A8, down product).

`DeviceForwardOptions::native_fp8` / `STRIX_DS41_NATIVE_FP8=DIR` (the checkpoint, or the dense extract written by
`tools/ds41_native_fp8/extract.py`) and `native_fp8_sites` / `STRIX_DS41_NATIVE_FP8_SITES` (`q_a,q_b,kv,o_a,o_b,shexp`
or `all`) select it; `STRIX_DS41_FP8_W=1|2|4` sets the outputs per wave (same bits; default 1 at 1-2 rows, 2 from 3 rows). Decode only: BatchForward and
DevicePrefill keep the Q8_0 form, and the host forward has no FP8 form (no op-hash gate for these steps). Status:
draft; CPU reference and device tests (`strix-ds41-native-fp8-tests`, `strix-hip-ds41-native-fp8-tests`) only, no model
fidelity or speed gate yet.
