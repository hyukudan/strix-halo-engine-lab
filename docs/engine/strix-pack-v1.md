# Strix Pack v1 format and conformance

The terms MUST and MAY apply to the wire format and arithmetic identities below. SP2, SP3, SP4 and the fixed-table
SP4T are the revision 1 storages `strix.sp2`, `strix.sp3`, `strix.sp4` and `strix.sp4t`. They use the same group
geometry. The candidate layouts in the block-layout benchmark have their own addressing and codebooks.

## Matrix payload

A matrix has shape `[N,K]`, ordered by row then column, with `N >= 1`, `K >= 128` and `K % 128 == 0`. It consists
of `N * (K/128)` consecutive 128-weight superblocks, with a fresh first superblock for each row. Every superblock has four
32-weight groups. The smallest matrix is `1 x 128`. There is no format-level row-count limit beyond the dimension,
checked byte-count and file-range constraints. Offsets, dimensions and lengths are unsigned 64-bit integers in the
container. Compute product bounds before multiplication, and apply the backend's buffer and index limits when staging.
A partial K superblock MUST be refused. There is no in-row padding or payload row alignment. Odd N and odd numbers of
activation rows are valid, and any backend tile padding MUST stay outside the logical outputs.

| Storage | Code plane(s), half-open byte intervals | Scale bytes | Bytes / block | Bytes / weight |
|---|---|---|---:|---:|
| SP2 | Low plane `[0,32)` | `[32,36)` | 36 | 0.28125 |
| SP3 | Low plane `[0,32)`, sign plane `[32,48)` | `[48,52)` | 52 | 0.40625 |
| SP4 and SP4T | Nibbles `[0,64)` | `[64,68)` | 68 | 0.53125 |

Within a superblock, for column `j in [0,128)`, SP2 and the SP3 low plane store the unsigned two-bit field in byte
`j/4` at bit `2*(j%4)`. SP3 adds bit `j%8` of byte `32+j/8` as the high bit of its code. SP4 and SP4T store the
four-bit field in byte `j/2`, at bit `4*(j%2)`. Bit 0 of any byte is its least significant bit.

For B bits, let `L = 2^(B-1)` and `T` be the table in the tensor header. Code `c` in numeric order is
`+T[c]` for `0 <= c < L`, and `-T[c-L]` for `L <= c < 2L`, multiplied by the group scale. All codes are
assigned. The default coder tables produce the following complete code lists, with the first entry at code 0.

| Storage | Signed levels in code order, before scaling |
|---|---|
| SP2 | +1, +3, −1, −3 |
| SP3 | +3, +10, +18, +30, −3, −10, −18, −30 |
| SP4 and SP4T | +1, +3, +5, +8, +10, +14, +18, +24, −1, −3, −5, −8, −10, −14, −18, −24 |

Scale byte `b` is at `16*B + j/32`. Its exponent field is `b>>3`, its fraction is `b&7`, and its real value is
`(8 + (b&7)) * 2^((b>>3) + base - 3)`. At a fixed base all 256 scales are positive and strictly increasing, from
`2^base` to `(15/8) * 2^(31+base)`. With an admitted header, scales and reconstructed weights are exact, finite,
normal binary32 and bfloat16 values. Scales may be subnormal or out of range in binary16. The product of a table
level and the scale is exact in binary16 if the product falls in its normal range. Only the unscaled level always
fits int8. The wire format has no NaN, infinity, zero or signed-zero code.

## Per-tensor header

The 16 bytes below are stored as 32 lowercase hex characters in the container entry. They do not precede the
matrix payload in the file or in residence.

| Byte(s) | Requirement |
|---|---|
| 0 | Version, 1 |
| 1 | Width, 2, 3 or 4, matching the storage |
| 2 | log2 group size, 5 |
| 3 | Flags. Bit 0 permits only the linear table `T[i] = 2*i+1`. Bit 2 marks the coder's conservative assessment that FP16 folding may be unsafe. Bit 1 (reserved for FP16 scales) and bits 3 to 7 MUST be zero. Decoding always uses the table and scales in the file. |
| 4 | Signed two's-complement tensor base, −120 through +90 inclusive |
| 5–7 | Reserved, zero |
| 8–15 | First `L` unsigned magnitudes strictly increasing, all remaining slots zero |

An admissible magnitude is in `[1,31]` and its odd part is at most 15, which gives at most four significant bits.
The list is `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30`. SP4T MUST use the default SP4 table.
Bit 0 may be clear for a linear table. To compute bit 2, compare the smallest table magnitude times the smallest
payload scale and the largest magnitude times the largest scale with the binary16 normal range
`[2^-14, 65504]`. Set it if either bound lies outside that range. Readers MUST use the actual product range for
a folded policy that requires this guarantee. The flag is metadata and does not affect the decoded values.

## Reference encoder

The writer stores codes and scales, so a reader does not need the training weights, table fitting method, or calibration
statistics. These rules specify the reference encoder, including the bytes it writes for a supplied table and inputs.

1. Validate nonempty whole-block geometry and the table. Refuse any weight that is NaN or infinite. Optional FP32
   importance is per K column and MUST be finite and nonnegative. The input preflight occurs before touching outputs.
   A non-finite calibration intermediate invalidates the output of the entire coding operation.
2. For a nonzero matrix let `smax = max(abs(W)) / T[L-1]` in binary64. The base is
   `clamp(floor(log2(smax))-30, -120, 90)`; a zero matrix uses base 0. Binary32 input subnormals participate
   normally. A scale is rounded to the nearest byte value, with exact ties going to the smaller byte, saturation at
   both ends, and mantissa carry into the next binade.
3. For each group, let `s0 = max(abs(group)) / T[L-1]` in binary64. Test each byte in increasing order between
   the *rounded*, inclusive endpoints `s0/2` and `9*s0/8`. For a candidate, choose the closest magnitude to
   each absolute input on that scaled table. Exact midpoint ties use the lower index. Inputs beyond the largest
   magnitude saturate to the largest index, and values below the smallest magnitude choose index 0. The sign bit
   is set only for `w < 0`, so both input zeros choose the positive sign. Minimize the reconstruction error
   using separate binary64 multiply and add operations in ascending K order, rounding to nearest-even. The
   term is `(importance * difference) * difference` (unit importance for RTN). Exact error ties use the lower byte.
   An all-zero group uses byte 0 and code 0 throughout, so it reconstructs to `T[0] * 2^base`.
4. Calibrated importance uses the diagonal of the input second moments. With error feedback, copy the source row
   and zero columns flagged dead by the factorization. Traverse K in increasing order. Choose each group's single
   byte from its current values when visiting its first column; retain that byte for its remaining columns.
   An all-zero group at that point keeps code 0. In each column quantize the current value, decode it, compute
   `error = (current - decoded) / U[i,i]` and update all later columns `current[j] -= error * U[i,j]`.
   Each subtraction, division and multiplication is a separate nearest-even binary32 operation. `U` is the
   upper inverse-moment factor of the damped calibration problem (see the implementation's inverse factor).
   Diagonal-only moments can supply importance. Feedback requires the full factor.
5. Optional table fitting starts with the default table, alternates reference assignments and weighted
   least-squares magnitudes over sampled rows, snaps to the admissible integer grids with tops
   16,18,...,30, and retains the best measured table including the default. The per-tensor header stores the
   chosen table. A no-calibration call is the same operation as RTN for that table.

**Lossless re-pack.** Preserve the header and original scale bytes and re-encode each reconstructed value
against exactly its original scale. Positive scales and strictly increasing, nonzero magnitudes make the code
unique at a fixed scale. Thus all valid payloads reproduce bit for bit. As an example of scale ambiguity,
SP2 with base 0 can represent +3 using code 1 with scale byte 0 or code 0 with scale byte 12. Rerunning encoder
scale search or feedback from reconstructed floats is a new quantization and can produce different bytes.

## Exactness classes

**Ordered binary32 projection.** This is the CPU and grouped reference identity on admitted packed weights.
For each activation row and each weight row, start with binary32 +0, then for columns `j = 0,1,...,K-1` do
`acc = fma32(x[j], decode(w[j]), acc)`. Use one rounded fused multiply-add per column, round-to-nearest-even and
preserve subnormals. Consumers MUST keep this order, including at group and superblock boundaries. Vectorizing
over independent output rows preserves it. Pre-scaling activations, doing an unfused multiply/add, reducing
partials, or applying the group scale after accumulating defines another identity. Float bit comparison applies
to finite results (including both zeros) and signed infinities. IEEE NaN classification applies when an activation
or an arithmetic exception yields NaN; NaN payload bits have no cross-backend contract. The grouped projection
returns assignments in expert-major schedule order and can gather the input by token or read it already grouped.
Route weighting and the activation or combine steps are separate operations.

**SP4T fixed-table head.** The HIP small-M path converts input activations to binary16 (nearest-even), uses the
signed integer levels as exact binary16 operands and starts each 32-column group's FP32 partial at +0. It chains
two K16 WMMA operations in ascending K within the group. With `base=0` in the group scale, it applies one FP32
FMA per group to the running +0-initialized output, in ascending K. At the store it multiplies by exact
`2^base` in binary32. Softcap, when selected, runs after that projection using the shared head epilogue.
The WMMA inner order is the native arithmetic identity `hip.sp4t.f16-group32-scale-chain@1`. Matching its
packed operands or row-count invariance does not establish ordered-binary32 projection identity. The
fixture uses exactly representable WMMA partials to give a portable expected digest for the head grouping.
The head's independent device comparison for general FP16 inputs uses the measured FP32 error envelope.

Pack-level exactness is relative to the stored weights and declared arithmetic. Tier admission against
an engine's original weights requires its own evidence and quality gate.

## Container v2

The sidecar starts at byte 0 with this space-delimited text framing. Line endings MUST be single LF bytes.
The header, through the final LF of `end`, is limited to 64 MiB. Digest and header fields are lowercase hex
without a prefix. Numeric fields are unsigned decimal, with no signs.

```text
strix-lowbit-pack/v2 COUNT
checkpoint CHECKPOINT_SHA256
tensor NAME STORAGE REVISION N K OFFSET BYTES SOURCE_SHA256 [STRIX_HEADER_HEX]
end
```

There are exactly COUNT tensor lines. Names are unique, nonempty byte strings without bytes 0–32 or 127.
Registered v1 SP entries require the tenth field. The container also admits revision 1 `strix.q4_g64` and
`strix.q6_g64` (resident q4/q6 in the [calibration guide](lowbit-calibration.md)) with nine fields and their own
68/100-byte 128-element superblocks. The host reader admits all six storages. The resident reader admits
q4, q6 and SP4T.

Every absolute payload offset MUST be aligned to 4096 and at or beyond the first 4096 boundary at or after the
header. Bytes between the header and that boundary MUST be LF. Payload lengths MUST equal their checked
geometry and payloads MUST be in bounds and disjoint. The offset order may differ from header order. Other
gap and trailing bytes are ignored by the reader. The buffered writer uses zero gap bytes and the streaming
writer may reserve a larger initial header region filled with LF. Neither padding nor trailing bytes enter
the pack identity. The pack identity is SHA-256 of the exact header through `end\n`, then each complete
payload concatenated in tensor-line order.

The checkpoint digest must match a nonzero expected digest. It is SHA-256 of the whole single checkpoint,
or SHA-256 of `strix-pack-sharded-checkpoint/v1` (ASCII, no terminator), the 32 raw bytes of the manifest hash,
and the 32 raw bytes of each whole-shard hash in lexical filename order. Before loading a named matrix,
verify its shape, storage and source digest. BF16 source digest is SHA-256 of its raw payload. MXFP4
and block FP8 source digests hash `strix-pack-source/v1` (ASCII), one byte `m` or `f` respectively,
then the raw SHA-256 of the code plane and of the scale plane.

## Source decoding

- BF16 little-endian words widen by appending 16 low zero bits. Signed zeros, infinities and NaN payloads are
  preserved by this widening.
- MXFP4 takes low nibble first, E2M1 signed magnitudes `0,1/2,1,3/2,2,3,4,6`, and one E8M0 byte per 32 columns.
  E8M0 byte `e` is `2^(e-127)`, including the binary32 subnormal at byte 0. The decoder refuses byte 255
  (NaN) and any product outside the finite binary32 range.
- E4M3 and E4M3FN source names both use finite-only E4M3 here. For exponent field `e=0`, magnitude is
  `m * 2^-9`; otherwise it is `(8+m) * 2^(e-10)`. Bit 7 is the sign; both signs of `e=15,m=7`
  are NaN. Multiply by the little-endian binary32 scale for the 128×128 tile, rounded once to binary32.
  The scale plane is `ceil(N/128) x ceil(K/128)` and the code plane has no tail padding.
  NaNs and infinities can reach the coder through this decoder, where the coder refuses them.
- For MXFP4 and FP8 the scale plane is the tensor with suffix `_scale_inv`, which can be in another shard.
  Its values multiply the decoded code.

The [fixture instructions](../../tests/fixtures/strixpack/README.md) give the portable binary32 digest
format and the HIP and CUDA checks. The [synthetic study](strix-pack-synthetic.md) measures group-size
and calibration tradeoffs.

## Defects resolved while specifying v1

- The direct RTN helper previously truncated partial K and could enter scale search on NaN or infinity.
  Entry checks now agree with the calibrated coder; table fitting and header creation also check source inputs.
- Computing the tensor's maximum scale in binary32 could turn a subnormal weight into a zero maximum scale.
  Base selection uses binary64; it can also differ at a binade where the former FP32 division rounded up.
- The container reader accepted names the writer rejected and admitted overlapping payloads. Name rules are
  shared, and ranges must be mutually disjoint. The buffered writer now checks the reader's header bound and
  computes stable aligned offsets.
- A malformed MXFP4 source could throw an exception from a status-returning API. It now produces a malformed
  status and the decoder checks plane lengths before allocating or decoding.
- The old comments overstated FP16 and int8 representability and misstated the scale search interval.
  The rules above specify range, rounding and the separate head identity explicitly.
