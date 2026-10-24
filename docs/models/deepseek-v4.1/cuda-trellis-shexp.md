# CUDA trellis shared expert (split decode FRONT on the pair's mix-B identity)

The pair's production dense configuration ds41-trellis-mixB-v1 runs the trellis library at two decode sites: the shared
expert of every layer (K4 pack) and the LM head (K5 pack, TAIL only). The CUDA FRONT of a split decode runs layers
[0, K) without the head, so its only trellis site is the shared expert. `CudaPrefillOptions::trellis_shexp` (apps:
`--trellis-shexp PACK_DIR`) replaces the Q8_0 shared expert of `step()` with the pack's three linears:

    gate = L_w1(x), up = L_w3(x)         x = the FFN-normed hidden in F32 (no A8), rows R <= 8
    act  = swiglu_clamped(gate, up, limit)   F32, ds41_moe.hpp (declared exp)
    shared = L_w2(act)

`run()` (prefill) keeps the Q8_0 shared expert, as the pair's prefill does.

## The linear's arithmetic (one row; every row independent of R)

1. A = RN16(H128(RN16(RN32(x * suh)))): H128 = Sylvester butterflies at strides 1..64 in F32 (low + high, low - high),
   then one multiplication by RN32(1 / sqrt(128)).
2. Q = the procedural mul1 codebook of the circular K-bit state stream (16x16 tiles, input-major).
3. Splits: the K16 tiles [floor(T s / S), floor(T (s + 1) / S)) of split s; S = the pair library's vector policy for
   R <= 8 rows and K outside its measured K5 / K6 overrides: output <= 1024: 32; output <= 4096 and input >= 4096: 16;
   else 4; at most input / 16 (shared expert: gate / up 5120 -> 2304: 16, down 2304 -> 5120: 4).
4. Per output n and lane class j = 0..3, one F32 accumulator from +0; per tile k ascending, two pair steps
   acc = RN32(acc + RN32(Q[i] A[i] + Q[i+1] A[i+1])) for i = 16 k + 2 j, then i = 16 k + 2 j + 8 (the half products are
   exact in F32).
5. partial_s = (acc_0 + acc_2) + (acc_1 + acc_3); y = RN32(H128(sum over s ascending from +0 of partial_s)) * svh.

Step 4 is the pair library's GEMV order provided its packed half dot-accumulate rounds the pair sum before adding the
accumulator; that is checked on hardware (the dump check below), not assumed.

## Gates

- `strix-cuda-trellis-shexp-dump-check PACK DUMP`: every value of R = 1..8 of the shared expert linears of the dumped
  layers bitwise equal to the pair's adapter outputs on the same inputs (dump written on the AMD node).
- Split decode: CUDA FRONT [0, 22) (`--expert-tp --overlay c-u6 --trellis-shexp ours-K4`) + pair TAIL [22, 40) + head
  (mix-B) == pair-only mix-B: tokens and full-logits FNV per step (R1) and per row of forced verify windows, from the
  same imported state.

## Declared composed identity: ds41-hetero-mixb-k22-v1

Measured on the AMD node: the pair library's packed half dot-accumulate does not round like any IEEE model of
a0 b0 + a1 b1 + acc (neither one rounding of the exact sum nor RN(acc + RN(p0 + p1))), so step 4 above is NOT the pair's
bits: the CUDA shared expert differs from the pair's in the last bits of ~92 % of its outputs. The split decode with the
CUDA FRONT therefore has its own declared identity:

    ds41-hetero-mixb-k22-v1 = layers [0, 22) on the CUDA FRONT: ds41-expert-tp2-i1152-v1 on the c-u6 dense weights,
                              shared expert = this document's IEEE trellis K4 linear (pack ours-K4)
                            + layers [22, 40) and the head on the pair TAIL: ds41-trellis-mixB-v1 as the pair runs it
                              (trellis shared expert K4 and head K5 on the pair library, HEADOUT), expert TP, c-u6

It is deterministic and R-invariant (every row's bits are independent of the window size on both sides). It is gated
bitwise against itself (a live split run == the TAIL replay of an independent FRONT run's recorded messages; two live
runs equal, R1 and verify windows) and by quality against pair-only ds41-trellis-mixB-v1 (full-logit KL, top-1, greedy
token match), not bitwise against it.
