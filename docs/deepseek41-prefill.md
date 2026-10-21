# deepseek41 prefill: routed experts and exact suffix replay

## Routed experts of T prompt rows

`hip_ds41_prefill.hpp` runs the routed (and shared) experts of a whole prompt block at once on
`v_wmma_i32_16x16x16_iu8`. It adds no numerical identity: every (row, output) value equals, bit for
bit, the one the decode window kernels (`hip_ds41_formats.hpp`) produce for that row alone, so prefill
results do not depend on T, on the batching or on the other rows of the block.

Routes (`enqueue_build_prefill_routes`, host reference `build_prefill_routes`):
- slot = row * top_k + k, as in the decode window;
- CSR dense over expert ids, the entries of an expert in ascending slot;
- each expert's entries cut into batches of 16 or 32 in ascending expert id, one workgroup per batch
  (the grid is sized by `prefill_max_batches`, extra workgroups exit);
- the combine's per-row order and low counts.

Arithmetic, per 32-weight group of a (row, output):
- the integer dot is two 16x16x16 iu8 WMMAs (K = 16 + 16) on an accumulator of 1.5 * 2^23, so one FP32
  subtraction yields float(dot) exactly (|dot| < 2^22 for every format);
- Q2_K weights enter the WMMA as `code * (scale & 15)` (at most 45 per byte), which gives A of the
  declared Q2_K term directly; B = sum over the two halves of `(scale >> 4) * sum x` is formed in
  integers from half sums staged with the activations;
- the group term is the declared one: `float(dot) * (scale * delta)` for IQ2_XXS, MXFP4 and Q8_0,
  `(d * float(A) - dmin * float(B)) * delta` for Q2_K;
- lane class l (groups l, l + 32, ...) accumulates in ascending order from 0, as one decode lane does;
- the 32 class sums are combined in the decode butterfly's tree (`v[l] += v[l ^ d]`, d = 16 .. 1): the
  classes are visited in bit-reversed order and folded with a five-level carry stack, which forms
  exactly the butterfly's pairs (l, l + 16), then (l, l + 8), and so on.

The gate/up epilogue (clamped SwiGLU with the declared exp, A8 per 32 outputs) and the raw slot
vectors of the down projection are the decode window's, so `enqueue_combine` (and the two-rank half
sums) apply unchanged.

Two kernel layouts, the same bits:
- class-split (batches of 16 entries, the default): 2 x 8 waves; class-wave q owns lane classes
  4q .. 4q + 3, i.e. one half of every 256-weight block it reads, so each weight row is read block by
  block in address order. Activations are staged in LDS one chunk of 32 groups at a time; the weights
  of a chunk arrive one chunk ahead. A wave covers 32 outputs: its two half-waves decode different
  outputs once and trade operands with `v_permlanex16`. The class sums meet in the butterfly tree
  between waves through LDS (levels 1-3) and in registers (levels 4-5). Gate/up covers 32 outputs of
  both matrices per workgroup, down 64.
- stage (batches of 32): 8 waves, each 16 outputs x two 16-row tiles of one matrix; activations staged
  eight lane classes at a time; classes in bit-reversed order folded with a carry stack. Gate/up
  covers 64 outputs, down 128. `STRIX_DS41_PREFILL_STAGE=1` sends batches of 16 here as well
  (comparison only).

Formats: gate/up IQ2_XXS, MXFP4, Q8_0 (shared expert); down Q2_K, MXFP4, Q8_0. Widths up to 5120
(multiples of 256 for the class-split layout). No register spills (at most 192 VGPRs in the
class-split kernels, two workgroups per WGP).

## Router of T rows

`enqueue_prefill_router` (RouterView of `hip_ds41_formats.hpp`, any row count) forms every
(row, expert) logit with the decode router's declared chain: lane l accumulates `acc = acc + w * x`
over the elements l, l + 32, ... in ascending order from 0, the lanes meet in the butterfly (lane 0's
value), then `router_score`. A block of 4 waves covers 16 rows x 16 experts (8 x 8 per wave) with x and
the weights staged in LDS 256 elements at a time. The top-k selection and the weights are the decode
router's. Scores, ids and weights equal `enqueue_router` row by row.

## Chunked MoE and expert halves

`PrefillMoe` runs the routed experts of T rows in chunks of at most `chunk_rows` rows: device routes,
gate/up, down, then either the combine (`enqueue_combine`, every expert) or the half sum of the owned
half (`enqueue_half`; half 0 = ids below split). Every step is per row, so any chunking gives the bits
of one pass; the scratch is sized by the chunk (raw slot vectors: chunk x top_k x hidden x 4 bytes,
240 MB for 2048 rows instead of ~1 GB for a whole 8K prompt). A rank that keeps only experts
[first, end) of the tensors passes weights at expert `first`; the routes then cover that range only,
renumbered from `first`, with the combine order and low counts of every slot. Two owners' half sums
and `enqueue_canonical_sum` give the single-node y.

`strix-ds41-prefill-pair` runs one MoE layer on two ranks (expert halves, router and shared expert
replicated): the half sums of chunk c go through the communicator while chunk c + 1 is computed, and
both ranks form y = (shared + low) + high per chunk. `--reference` compares y with the single-node
prefill bitwise; `--transport memory` runs both ranks in one process.

Chunks with few entries per owned expert (mean <= 5 by default, `STRIX_DS41_PREFILL_WINDOW_MEAN`)
run the decode window kernels over route batches of 8 entries (`batch_begin[batch_total]` holds the
entry count, so the batches are window routes); the same declared products, hence the same bits.
Measured on layer 20 (one node): T = 128 22.5 ms instead of 36.3, T = 256 33.6 instead of 43.7; from
T = 512 the WMMA kernels are faster.

## Activation and weight staging

The class-split kernels read the 64 weight rows of a chunk with coalesced dword loads into an LDS
tile (`STRIX_DS41_SPLIT_WEIGHT_LDS`, default on). Build-time alternatives kept for comparison, none
faster on gfx1151 in the measurements: activations of chunk c + 1 in registers while chunk c is
multiplied (`STRIX_DS41_SPLIT_DOUBLE=1`, two LDS stage buffers), per-lane weight chunks two ahead
(`STRIX_DS41_SPLIT_WEIGHT_DEPTH`).

## Exact suffix replay (CED prefill)

Every layer keeps its own sliding window (128 rows, the current one included) of K/V computed from
its own input; the compressed KV and index keys of the decoder layers (20..39) come from layer 20,
whose input is the encoder output. `plan_suffix_replay` gives, per layer, the rows to compute so that
the decode state (every layer's ring, the KV-source caches), the last row's logits and the drafter's
seed (inputs of layers 37..39 for the last 128 rows) equal those of a full prefill:

- encoder layers 0..19: every new row;
- decoder layer 39: the last row; layer 38 - m: the last 128 + 127 m rows (2,414 at layer 20);
- window K/V of a decoder layer: 127 rows below its first output row (and at least the last 128);
- layer 20's compressor and index keys: every new row (from the encoder output);
- with a cached prefix of C rows, rows before C are never recomputed: they come from the saved rings.

For a fresh 8K prompt the decoder runs 24,150 row-layers instead of 163,840 (57 % of all row-layers
of the model). `verify_suffix_replay` checks a plan against the brute-force dependency closure over the
per-layer windows; the CPU test runs it on 3,292 plans (fresh and cached prefixes, small geometries)
and rejects every one-row mutation of the 8K plan.

A prompt processed in several encoder chunks needs the encoder output (layer 20 input) of its last
`encoder_output_begin .. total` rows (2,541 for a fresh prompt) when the decoder cone runs at the end.
