# GDN FP16 chunked prefill: `qwen-gdn-fp16-chunk`, revision 2

Status: FAST (bit-changing) engine item, default OFF. `STRIX_QPF2K_GDN_FP16_CHUNK=1` replaces the conv, the recurrence
and the gated norm of the GDN prefill row route with a chunked route on FP16 WMMA operands with FP32 accumulation.
Unset or `0` keeps the exact production route. The item is ignored under the baseline and for GSQ checkpoints, and
shapes it cannot take stay exact. When it is on, stderr carries one line
`GDN FAST lever qwen-gdn-fp16-chunk rev 2: ON (...)` and the QPF2K witness names the route.

It is the prefill option `gdn_fp16_chunk` (identity tag `gdnchunk16`): `STRIX_PREFILL_EXACT_FP16=1` or
`--prefill-disable gdn_fp16_chunk` turn it off, and the GDN check modes run the exact route.

The mathematics is the public chunked (WY) form of the gated delta rule. The rounding points below define the identity.

Source: `src/hip_gdn_fp16_chunk.inc` (engine hook and norm epilogue in `src/hip_gdn_launcher.hip`).
Test: `tests/hip_gdn_fp16_chunk_tests.hip` (target `strix-hip-gdn-fp16-chunk-tests`).

## Numerical contract (revision 2)

Per value head `h`, key head `h % key_heads`, chunks of 32 rows from the call start, `m` valid rows. Padded rows have
q = k = v = 0, beta = 0 and carry the prefix of row `m - 1`.

- **Conv.** q, k and v are the production conv values: BF16 inputs (history below row 0), `sum = 0; sum += x_j * w_j`
  for taps 0..3, SiLU `sum / (1 + expf(-sum))`, `round_bf16`. q and k are L2-normed with the production tree, and q is
  scaled by `rsqrtf(128)`. `Q16 = fp16(q)`, `K16 = fp16(k)`. v stays exact (BF16).
- **Decay.** `g = gate`, with values below -110 raised to -110 (NaN kept). `G` is the inclusive prefix over the chunk
  (shuffle ladder 1, 2, 4, 8, 16). `e_r = expf(G_r)`, `f_r = expf(G_{m-1} - G_r)`, `eL = e_{m-1}`.
- **Prep.** `KK = W(K16, K16^T)`, `QK = W(Q16, K16^T)`. `W` is a native `v_wmma_f32_16x16x16_f16` chain with K16
  panels in ascending order. `D_rs = expf(G_r - G_s)`.
  - `A_rs = (beta_r * KK_rs) * D_rs` for `s < r`.
  - `P16_rs = fp16(QK_rs * D_rs)` for `s <= r`.
  - `T = (I + A)^-1` by FP32 forward substitution, pivots ascending, `x_i = fmaf(-A_ip, x_p, x_i)`.
  - `T'16_rs = fp16(T_rs * beta_s)`.
- **Scan.** The FP32 state `S[v][k]` is kept as `S^T` accumulator tiles.
  - `X16 = fp16(S)`; `KS = W(K16, X16^T)`; `QS = W(Q16, X16^T)`.
  - `Y16 = fp16(fmaf(-e_r, KS, v))`; `U = W(T'16, Y16)`; `U16 = fp16(U)`; `U2_16 = fp16(U * f_r)`.
  - `S^T <- W(K16^T, U2_16, seed eL * S^T)`.
  - `O = W(P16, U16, seed QS * e_r)`.
- **Output.** O (FP32) goes through the production gated-norm expression: `gdn_window_key_sum` of squares / 128 + eps,
  `rsqrtf`, times the norm weight and `sigmoid(z)`. The result then goes to CORE16, dense A8 or the FP32 core, as on
  the production route.
- **Bits.** Results are deterministic run to run. Splits at multiples of 32 are bitwise invariant.

## Kernels

| Kernel | Grid | Threads | Work |
|---|---|---:|---|
| `prep_kernel` | (key heads, chunks) | 256 | Conv of q, k and the group's v (sliding windows); Q16/K16/v to scratch; KK/QK; prefix; A, P16, (e, f); T'16 per value head |
| `scan_kernel` | value heads | 256 | 8 waves × 16 value columns. `S^T` held in registers. Each chunk is staged once in LDS (Q16, K16, K16^T, T'16, P16, v, e/f) from registers loaded during the previous chunk. z is loaded during the chunk. O is staged in the dead Q/K buffer for the norm epilogue. 41.7 KB LDS, ≤ 240 VGPR (6 waves/SIMD: all 48 heads resident) |
| `conv_history_kernel` | conv channel blocks | 128 | New BF16 conv history, after both readers |

**Persistent scratch** grows per (device, stream) and is never shrunk. It holds Q16/K16 and v rows, T'16/P16 and (e, f).

## Test

`strix-hip-gdn-fp16-chunk-tests` exits with 77 when no GPU is present.

**Checks against an FP64 token recurrence** (production conv in FP32, FP64 L2, recurrence and gated norm), for raw
core, normed output and final state:
- relL2 <= 3e-3 and max error <= 2e-2 relative to max(1, |ref|);
- exact conv history;
- untouched inputs.

**Cases:**
- lengths 1-1000 and every tail class;
- resets and tiny decays;
- both state layouts and several head groupings;
- bitwise run-to-run equality and aligned split invariance.

**`--time`** reports the median prep, scan + norm, raw scan and total at T = 8192 with 48/16 heads.
