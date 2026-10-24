# deepseek41 MoE layer (decode)

Status: kernels and host reference for the MoE layer (router, shared expert, routed experts, combine) over a
verify window of 1 to 8 rows; not yet wired into a model. Files: `include/strix/ds41_moe.hpp`, `src/ds41_moe.cpp` (host), the
window part of `src/hip_ds41_formats.hip` and `include/strix/hip_ds41_formats.hpp` (device), fixtures in
`tests/ds41_formats_tests.cpp` and `tests/hip_ds41_formats_tests.hip`.

## Window routes

`build_window_routes(ids, rows, top_k, split)` turns the routing of the window (row r's k-th expert
`ids[r * top_k + k]`, slot `r * top_k + k`) into CSR over the union of experts in ascending id; the entries of
an expert are its rows in ascending order. It also gives, per row, the slots in ascending expert id and how
many of them are below `split` (half the routed experts). Launches use one block row per union expert.

Routes can also be built on the device from the router's ids (`enqueue_build_routes`, equal to the host
routes): the expert count stays in device memory, launches cover every possible expert and blocks past the
count do nothing, and kernels stage an expert's entries a few at a time (each entry's result does not depend
on the others), so no host round trip is needed between the router and the experts. With `base`, the built
expert indices are `id - base`, for a rank that keeps only its own expert slice in memory.

## Pipeline and declared arithmetic

1. **Activations:** the A8 contract of `formats.md` on each row's MoE input.
2. **Fused gate/up** (`enqueue_gate_up`, IQ2_XXS for the target model, MXFP4 for the draft stages): for every
   (row, slot) and intermediate output, the gate and up declared products, then
   `swiglu_clamped(g, u, limit) = (min(g, limit) / (1 + exp(-min(g, limit)))) * clamp(u, -limit, limit)`, with
   `exp` the `declared_exp` of `ds41_moe.hpp` (Cody-Waite reduction, degree-7 polynomial, power-of-two scaling,
   basic IEEE operations only, so host and device agree bitwise). Each 32-output group of the activation is
   A8-quantized for the down projection. The FP32 gate/up values can be kept for checks.
3. **Down** (`enqueue_routes_product`, Q2_K or MXFP4): every (row, slot) gets its **unweighted** FP32 down
   vector (the raw slot vector).
4. **Combine** (`enqueue_combine`): per row and output,
   `y = (shared + low) + high`, where low sums `w * raw` over the row's slots with expert id below `split` in
   ascending id and high the rest likewise; every step is one FP32 product and one FP32 add.

Because the routing weight is applied in the combine and the raw slot vector of a (row, slot) does not depend
on the other entries of the window, a rank that computes a subset of the experts produces the same raw vectors
as a single node; exchanging them and running the same combine gives the single-node bits. A rank that owns
one half of the experts by id can equally send its finished half sum. (The model's reference applies the
routing weight before the down projection; in exact arithmetic the two are equal.)

All per-entry results are those of the declared products of `formats.md`: the lane, group order and
butterfly of an (entry, output) do not depend on how many entries an expert has or on the kernel's tiling.

## Router and shared expert

- **Router** (`reference_router`, `enqueue_router`): `logits = x . W` in FP32 with the declared lane order of the
  products, `score = sqrt(softplus(logit))` (softplus as `log1p(exp(z))` below the threshold 20, `z` above),
  with `declared_exp`, `declared_log1p` (`log(u) * e / (u - 1)`, `u = 1 + e`) and a correctly rounded square
  root (through double precision on the device, whose single-precision root is not correctly rounded over the
  whole range). The top-k experts maximize `score + bias` (the image-token bias for rows marked as image tokens),
  larger first and the lower id on ties; the weights are the unbiased scores divided by their sum in selection
  order plus 1e-20, times the routing scale.
  The score kernel gives each wave one expert and all window rows; a block of four waves stages the rows of `x`
  in LDS 512 elements at a time, and each lane loads its weights of the next chunk while it forms the products
  of the current one. Each lane still sums its elements `l, l + 32, ...` in ascending order, so the bits do not
  depend on the chunking (`RouterView::staged = false` selects the previous form, which read `x` from global
  memory inside the lane loop, for A/B). Layer 20 (384 experts, cold, median of 7): 1 row 0.12 ms (0.11 before),
  2 rows 0.14 (0.40), 3 rows 0.15 (0.64), 4 rows 0.16 (0.69), 6 rows 0.19 (0.78), 8 rows 0.21 (0.86).
- **Shared expert**: Q8_0 gate/up/down (`WeightFormat::q8_0`, the same integer form as MXFP4 with the block's FP16
  scale) through the same fused gate/up and down kernels, every row on one expert; its down output is the
  `shared` term of the combine.
- **Expert parallelism**: `restrict_routes` gives a rank the experts below or above the split; `enqueue_half_sum`
  forms its half of every row and `enqueue_canonical_sum` the combine from the two halves (see
  `../../engine/transport/communicator.md`); both ranks reproduce the single-node output bitwise.

## Kernels

A block owns one union expert and 32 outputs. It stages the A8 rows of the expert's entries in LDS; each wave
owns 32 / waves outputs (gate and up together in the fused kernel) and decodes every 32-weight group of its
weight rows once for all staged rows. The raw bytes of the next group of every weight row are loaded before the
current group is decoded. Blocks have 8 waves, or 16 when an expert has four or more entries. Q2_K uses 24-bit
products for its scale terms and computes the activation half sums once per row and group.

## Measurements (gfx1151, layer 20 and draft stage 0 of the pinned files, cold caches)

Each repetition first reads a 256 MB buffer (read-only, so no write-backs are pending), then times the three
launches; medians of 20. Bytes are the union experts' gate, up and down weights.

| Routing | Rows | Union experts | gate/up | down | window total |
| --- | --- | --- | --- | --- | --- |
| target, independent top-6 per row | 1 | 6 | 191 GB/s | 214 GB/s | 0.31 ms, 194 GB/s |
| | 2 | 12 | 212 | 225 | 0.56 ms, 214 GB/s |
| | 4 | 22 | 217 | 227 | 1.00 ms, 219 GB/s |
| | 6 | 36 | 218 | 233 | 1.61 ms, 222 GB/s |
| | 8 | 47 | 221 | 229 | 2.10 ms, 223 GB/s |
| target, the same top-6 for every row | 2 | 6 | 183 | 200 | 0.32 ms |
| | 4 | 6 | 186 | 141 | 0.37 ms |
| | 6 | 6 | 141 | 103 | 0.49 ms |
| | 8 | 6 | 120 | 80 | 0.60 ms |
| draft stage, independent top-3 | 1 / 2 / 4 / 6 / 8 | 3 / 6 / 12 / 17 / 24 | 202-214 | 191-228 | 193-217 GB/s |
| draft stage, the same top-3 | 2 / 4 / 6 / 8 | 3 | 200 / 185 / 176 / 169 | 190 / 178 / 147 / 126 | 0.29-0.38 ms |

With independent routing (the usual case with 384 experts) the window reads at the scattered-read rate of the
device. When many rows share experts each expert is still read once, so the window is faster in absolute time,
but the work per weight byte grows with the rows and the rate falls below the read limit.
