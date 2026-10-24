# ds41-expert-tp2-i1152-v1: routed experts split by the intermediate dimension over two ranks

Status: declared identity, the default of the two-rank decode since P16. `STRIX_DS41_EXPERT_TP` (read once at
startup): `1` on, `0` off; unset, the two-rank `DeviceForward` (decode steps and verify windows) and `BatchForward` take it
when the layout allows it (two ranks, resident experts, I = 2304; `expert_tp::decode_enabled`), and the server keeps it
off (its device prefill reads the decode weights in place). The host reference (`HostForward`,
`strix-deepseek41-trimmed`) follows only an explicit `1`. With `0` every path keeps the expert-parallel identity
(`combine-shared+low-half+high-half-ascending-id-v1`, `moe.md`) and its bits. Files: `include/strix/ds41_expert_tp.hpp`,
`src/ds41_expert_tp.cpp` (layout + host reference), `include/strix/hip_ds41_expert_tp.hpp`, `src/hip_ds41_expert_tp.hip`
(loader + down-row packing), `enqueue_routes_product_groups` in `src/hip_ds41_formats.hip`, tests
`tests/ds41_expert_tp_tests.cpp`.

## Why

The current two-rank decode is expert parallel: rank 0 owns experts [0, 192), rank 1 [192, 384). A row's six experts
split 3/3 only ~31 % of the time (W1 routes), so the rank with more experts makes the other wait at every combine
(~2 ms per token at R1). Here every rank computes **all** selected experts, each on half of its intermediate
dimension, so both ranks always read the same bytes.

## Layout (per expert, per rank)

Shapes of the target model: H = 5120, I = 2304, gate/up IQ2_XXS rows of H (1320 B), down Q2_K rows of I (9 blocks of
256, 756 B). Rank r holds 1152 intermediate rows of gate and up and, for every one of the H down rows, the 36 groups of
32 that multiply those 1152 activations.

| | rank 0 | rank 1 |
|---|---|---|
| intermediate rows (local j -> global) | j | 1280 + j (j < 1024); 1152 + (j - 1024) (j >= 1024) |
| down groups (local g -> global) | g | 40 + g (g < 32); 36 + (g - 32) (g >= 32) |
| down row stored (5 blocks, 420 B) | blocks 0..4 (block 4: groups 0..3 used) | blocks 5..8, then block 4 with scales 8..15 copied over scales 0..7 and the codes of groups 4..7 over those of groups 0..3 (d, dmin kept) |

Packed form (default on the device): the rank's 36 groups as 4 whole blocks (rank 0 blocks 0..3, rank 1 blocks 5..8) and
then its half of block 4 as one 44-byte half block (8 scale bytes, 32 code bytes, d, dmin): 380 B per down row, read by
the Tail form of the routes product (`enqueue_routes_product_groups(..., half_tail = true)`). `STRIX_DS41_EXPERT_TP_LAYOUT=dup`
keeps the 420 B form above (the first prototype). Both forms give every group the same codes, scales, d and dmin, so the
products and the bits are the same (gated both ways). Per expert and rank: gate + up 3,041,280 B + down 1,945,600 B =
4,986,880 B (+0.2 % over an exact half); resident routed experts per rank 71.34 GiB (dup form 74.27 GiB; EP today 71.19).

The local orders move whole 32-element groups, and rank 1's 128 elements of the boundary block are one Q2_K half
block (8 scale bytes, 32 code bytes, the block's d/dmin), so the existing Q2_K group reader reads a rank's 36 groups as a
row of width 1152. Boundary block 4 is stored on both ranks (the prototype "duplicate + mask": the unused half is never
read). Bytes per expert and rank: gate + up 3,041,280, down 2,150,400 (ideal 1,935,360), total 5,191,680 = +4.32 % over an
exact half; resident routed experts per rank 74.27 GiB (40 layers x 384 experts) against 71.19 GiB today (192 whole experts).

## Declared arithmetic

For a row x (FFN input, A8 as in `formats.md`), its six slots k (expert e_k, routing weight w_k from the router) and
rank r:

1. **Gate/up** (unchanged per output): for every local intermediate row j, `g = product_iq2_xxs(gate[e_k][inter_index(r, j)], x)`,
   `u` likewise, `a_j = swiglu_clamped(g, u, 10)`. The 1152 activations are A8-quantized in groups of 32 local rows; every
   group is a group of the unsplit expert, so codes, deltas and sums are those of the unsplit expert's activation.
2. **K-half down** (`product_q2_k_groups`, device `enqueue_routes_product_groups`): for every output o,
   `P_r[k][o]` = the declared Q2_K product over the rank's 36 local groups: group g contributes
   `(d * float(A) - dmin * float(B)) * delta_g` (A, B as in `product_q2_k`), lane l accumulates local groups l, l + 32 in
   ascending order from +0, then the butterfly `v[l] = v[l] + v[l ^ d]` for d = 16, 8, 4, 2, 1; the result is lane 0. (Rank 1's
   lanes 0..3 thus add global group 40 + l, then 36 + l.)
3. **Weighting and local fold** (`half_sum` with every slot in the low half): `S_r[o] = (((0 + w_(1) P_r[(1)][o]) + w_(2) P_r[(2)][o]) + ...) + w_(6) P_r[(6)][o]`
   over the row's six slots in **ascending expert id**, each term one FP32 product then one FP32 add. The routing weight
   is applied to each rank's part, after the down product (not before it).
4. **Exchange**: each rank sends S_r ([R][H] FP32, 20 KB per row: the existing half-sum exchange, still one per layer,
   81 exchanges per token with the attention ones; no new collective).
5. **Cross-rank sum and shared expert**: `y[o] = (shared[o] + S_0[o]) + S_1[o]` on both ranks (rank 0's sum first,
   whatever the local rank), `shared` the shared expert's output (`q8_0-a8-swiglu-clamp10-v1`, or its declared packed /
   native-FP8 forms), computed as today (replicated on both ranks and overlapped with the exchange, or split by rows with
   `STRIX_DS41_SPLIT_SHARED=1`; the shared expert's position in the sum is unchanged).

Everything else in the step (router, attention, hyper-connections, head) is unchanged. A row's y depends only on its own
x and routing, so the identity is the same for R1 decode steps, verify windows of 2..8 rows (incl. rejected rows and
`commit_window` rollback, which only concern attention state) and batch rows.

### Differences from the current identity

Current: `raw[k][o]` = product over 72 groups (lane l: groups l, l + 32, l + 64), `low` = fold of the slots with id < 192,
`high` the rest, `y = (shared + low) + high`. Here the per-output down product is split in two 36-group products with
their own lane sums, and both ranks fold all six experts before the cross-rank add. The values agree in exact arithmetic;
the FP32 bits differ (`combine@L:y` hashes differ from the first layer on).

## Reference and gates

- Host: `ds41::expert_tp::reference_window` (whole tensors in, the rank layouts built with `pack_down_row` /
  `inter_index`), used by `HostForward` under the flag (`strix-deepseek41-trimmed` prints `expert tp:
  ds41-expert-tp2-i1152-v1`). `routed_experts@L:raw` holds rank 0's parts (rank-local, not compared).
- `tests/ds41_expert_tp_tests.cpp`: local orders cover every row/group once; every packed group decodes to the unsplit
  group (codes, scales, d, dmin); the group reader over 72 groups equals `product_q2_k` bitwise; part 0 + part 1 equals the
  exact product up to rounding.
- Device: op hashes of every step equal the host reference's on both ranks (R1), fill windows of 2..8 rows and
  rejected windows (`--fill-keep`) equal the R1 fill, batch tokens/logits equal DeviceForward alone in the identity.

## Mixed identity ds41-prefill-ep+decode-tp

`DevicePrefill` implements only the expert-parallel identity and ignores `STRIX_DS41_EXPERT_TP`. A prompt filled by the
device prefill and continued by a decode forward with expert TP (`strix-deepseek41-longctx`, where the prefill object is
freed before the decode forward loads; a state imported from another prefill) runs the declared mixed identity
**ds41-prefill-ep+decode-tp**: prompt state of the expert-parallel prefill, every decode step and verify window in
ds41-expert-tp2-i1152-v1.

- The decode stays exact against its own CPU reference (this document's gates).
- The prefill is not bit-identical to an expert-TP prefill; it is validated by quality, as the RTX prefill is: on 3
  prompts, the first 64 generated tokens equal those of the expert-parallel prefill + expert-parallel decode, and the KL of
  the continuation's full logits (`strix-deepseek41-longctx --logits-out`) against that run, mean and max per step.
