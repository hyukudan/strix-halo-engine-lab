# deepseek41: layer sites outside attention and the MoE

Status: host reference and device kernels, bitwise equal in fixtures (synthetic and real tensors of the
pinned file) and in the integrated trimmed-model lockstep (host forward = device forward, every op).

Files: `include/strix/ds41_layer.hpp` (declared arithmetic, host reference), `include/strix/hip_ds41_layer.hpp`
(kernels), `include/strix/ds41_engram.hpp` (n-gram hash and file reader), tests `ds41_layer_tests.cpp` (host,
double-precision oracles) and `hip_ds41_layer_tests.hip` (device = host).

## Identity and reduction order (`ds41-exact-v1`)

The residual stream is FP32, `hc_mult = 4` copies of `hidden` values, copy-major. Every reduction follows one of
these declared orders, which the kernels reproduce:

- **lane dot**: lane `l` of a 32-lane wave takes elements `l, l + 32, ...` in ascending order with `acc = acc + a * b`
  (one rounding each, no fused multiply-add), then the butterfly `v[l] += v[l ^ d]`, `d = 16 ... 1`; the result is
  lane 0. Used by the F16 projections.
- **chunked dot** (`dot_chunk = 640`): the elements are cut into chunks of 640; each chunk is a lane dot of its own
  elements, and `total = total + chunk_sum` in ascending chunk order from 0. Used by the long reductions of these
  sites (the RMSNorm sum of squares of `attn_norm`, `ffn_norm`, `q_a_norm` and `output_norm`, the hyper-connection
  mixes and their sum of squares, the three Engram gate sums), so that each chunk can run in its own wave.
- **ascending sums** over the four copies and over Sinkhorn rows and columns.

The attention piece declares its own 512-value RMSNorm (`rms_norm_512`, used by `kv_norm` and the compressor norm);
both declarations are part of `ds41-exact-v1`, each at its own sites.

Q8_0 linears take **A8 activations**: `quantize_a8` per 32 values (delta = amax / 127, code = round-to-nearest-even of
`y / delta` clamped to ±127) and `product_q8_0`: lane `l` takes the 32-weight groups `l, l + 32, ...` ascending, term
`float(integer dot) * (scale * delta)`, then the butterfly. This is the same contract as the MoE experts and the shared
expert, so a Q8_0 site has one identity in decode and prefill.

`exp`, `log`, `sqrt` are the declared functions shared with the MoE (basic IEEE operations only); `rsqrt(x) =
1 / sqrt(x)`, `sigmoid(x) = 1 / (1 + exp(-x))`. Division and square root are correctly rounded on the device.

## Sites

| Site | Arithmetic |
| --- | --- |
| Hyper-connection mixes | `m = chunked_dot(fn_m, stream) * rsqrt(chunked_dot(stream, stream) / (4 * hidden) + eps)` for the 24 rows of the F16 `hc_*_fn`; `pre = sigmoid(m * s0 + b) + hc_eps`, `post = 2 sigmoid(m * s1 + b)`, `comb` = row softmax + `hc_eps`, column normalisation, then 19 rounds of (row, column) normalisation, each `c / (sum + hc_eps)` |
| Collapse / expansion | `y = sum_c pre[c] x[c]`; `out[k] = post[k] sub + sum_j comb[j][k] residual[j]`. A sublayer collapses with the `pre` of the previous sublayer's mixes (the first layer with copy 0), as the model is defined |
| RMSNorm (attn, ffn, q_a 1280, output) | `w * (x * rsqrt(chunked_dot(x, x) / n + eps))`, eps `1e-20` |
| RoPE | last 64 dims, adjacent pairs; frequencies and angles in double on the host, `cos`/`sin` rounded once to FP32 per position; window-only layers use theta 10,000 without YaRN, the others theta 160,000 with YaRN (factor 16 over 65,536, beta 32/1); the inverse rotation for the attention output. In the integrated decode plan the attention piece applies the RoPE of `q` and of the kv row and the inverse RoPE of `o`; the RoPE kernels here serve prefill and tests |
| Q8_0 projections (`attn_q_a`, `attn_q_b`, `attn_kv`, `attn_output_b`, `output`) | A8 activations × Q8_0 (`product_q8_0`) |
| Grouped `attn_output_a` | 8 block-diagonal groups, group `g` reads the A8 codes of its 4,096 inputs and writes its 1,024 outputs |
| F16 projection (`engram_kv`) | lane dot of the F16 row |
| Embedding | the F16 row, exact in FP32, in every copy |
| Head and greedy selection | each rank computes its vocabulary slice; candidate = largest finite logit, ties to the lowest index; ranks exchange (value, index) and combine with the same rule, which equals the single-node selection; any non-finite logit marks the result unusable |

### Identity per site (`ds41-exact-v1`)

| Site | Weights | Activations | Input produced by |
| --- | --- | --- | --- |
| `hc_attn_fn`, `hc_ffn_fn`, `output_hc_fn` mixes | F16 | FP32 stream | chunked dot |
| `attn_q_a`, `attn_kv` | Q8_0 | A8 of `attn_norm(collapse(stream))` | site launch (collapse + norm + A8) |
| `attn_q_b` | Q8_0 | A8 of `q_a_norm(q_a)` | norm with FP32 and A8 outputs |
| `attn_output_a` (8 groups) | Q8_0 | A8 of `o` after its inverse RoPE (per 32 values, so per group) | `quantize_a8` (or the fused inverse RoPE + A8) |
| `attn_output_b` | Q8_0 | A8 of the 8,192 grouped outputs | `quantize_a8` |
| `output` (head, split by vocabulary) | Q8_0 | A8 of `output_norm(collapse(stream))` | site launch (expansion + collapse + norm + A8) |
| `engram_kv` | F16 | FP32 bfloat16-rounded rows | lane dot |

The FP32 `q` (after `attn_q_b`) and the FP32 `kv` row (after `kv_norm` + RoPE) go to attention.

## Kernel forms

All forms of a site give the same bits for a row, whatever the number of rows, the batch or the launch grouping.

- **Site launch (decode)**: `enqueue_hc_site` replaces the chain between two sublayers (expansion of the previous
  sublayer, this site's mixes + split + Sinkhorn, collapse + RMSNorm + A8 of the sublayer input) with one launch.
  Mix-partial blocks (one per 640-chunk and tile of 8 rows) and norm blocks (one per 640-chunk and row) run side by
  side and recompute the expansion they need; the last partial block of a tile, found with an atomic counter, runs
  the split + Sinkhorn after an agent-scope acquire (the other blocks release their partials with a fence before
  their increment). The same launch with no mixes is the head input
  (expansion + final collapse + `output_norm` + A8). Layers with an Engram injection keep the expansion separate.
- **Decode projections (1–8 rows)**: `enqueue_q8_0_a8` streams the weights once for all rows (one wave per 1–4
  output rows, next group in flight, `sudot4` integer dots). With 5–8 rows the block's waves share the activations
  staged in LDS by chunks of 128 groups; a chunk starts at a multiple of 32 groups, so every lane keeps its ascending
  group class.
- **Other small kernels**: RMSNorm (one wave per 640-chunk; up to 8 rows with the output apart from the input, each
  row spread over one block per chunk, every block recomputing the row's sum of squares in the declared order; in
  place, one block per row), inverse RoPE + A8, the mixes as chunk partials over row tiles plus a per-row split +
  Sinkhorn on 16 lanes.
- **Prefill (T rows)**: `enqueue_q8_0_a8_prefill` maps a site of width up to 5,120 onto the WMMA iu8 batch kernel of
  the prefill experts (dense site = one expert with slot `t`; grouped `attn_output_a` = 8 experts with slot
  `t * 8 + g`, which is the `[t][g][4096]` input and `[t][g][1024]` output layout). `enqueue_q8_0_a8_dense_prefill`
  is the own dense WMMA iu8 form for any width (`attn_output_b`, 8,192): two 16×16×16 iu8 WMMAs give each group's
  exact integer dot for a 16 × 16 tile; the four waves of a block own the lane classes `c mod 4` and walk stripes of
  32 consecutive groups together, so every class sums its groups in ascending order; the 32 class sums are then
  combined in the butterfly's addition tree (distances 16, 8, 4 inside a wave, 2 and 1 across waves through LDS). The
  head needs only the last row of a prompt and uses the decode form. Per-row kernels take any number of rows.

Aliasing: kernel pointers are `__restrict__` (in force-inlined bodies for the view kernels); the calls state where in
place is allowed (RMSNorm with `x == out`, Engram injection, RoPE). Every decode call has a `launch_access` description
of the device bytes it may read and write, for a launcher that tracks cache lines between dependent kernels.

## Engram

- **Hash** (host, from the admitted contract): compressed ids of the last 4 tokens (the compressed pad id before the
  sequence start), products with the per-layer multipliers in 64-bit integers, running XOR, and for each n-gram size
  and head `rolling % prime + offset` (offsets: running sum of the layer's primes, n-gram size major). 24 rows per
  token per Engram layer. `truncate` rolls the history back for verify windows.
- **Rows** come from the model file on disk through `EngramRowReader`: a pool of worker threads, `pread` of the
  aligned 4 KiB page(s) holding each 264-byte row with `O_DIRECT` (the page cache stays with the resident model),
  buffered `pread` as a fallback. A batch call blocks until every row is in place. The rows of a token depend only on
  the tokens, so a token's read is issued at its start on a host thread and overlaps the embedding and layer 0 (the
  Engram layers are 1 and 14).
- **Decode**: E4M3 times the E8M0 scale of each 32 values, then rounded to bfloat16 (as the model was trained).
- **Injection**: `kv = engram_kv * rows` gives a key per copy and one value; per copy
  `dot = (chunked_dot(h * (q * k), key) * (rh * rk)) * float(1 / sqrt(hidden))` with `rh`, `rk` the RMS reciprocals,
  gate `sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))`, `h = h + gate * value`.

## Checks

- Host: every site against a double-precision oracle (relative errors below 4e-6, Sinkhorn output doubly stochastic,
  inverse RoPE restores its input within FP32 rounding, split argmax equals single argmax including cross-rank ties
  and NaN); the hash against rows computed independently from the model definition; the reader against plain reads.
- Device: each kernel bitwise equal to the host on synthetic inputs (1–12 rows; the site launch with every variant at
  1, 3, 8 and 9 rows, twice, with its counters back at zero; RMSNorm in place; the dense prefill form at widths 8,192,
  1,280 and 1,024 with rows that are not a multiple of the tile) and, with `STRIX_DEEPSEEK41_GGUF`, a chain over real
  layer-1 weights in the decode data flow: embedding, Engram (hash, file rows, decode, projection, injection), mixes,
  collapse + norm + A8, `q_a`, `q_a` norm + A8, `q_b`, query RoPE, `kv`, kv norm + RoPE, a stand-in attention output
  through inverse RoPE + A8, the grouped `attn_output_a`, A8, `attn_output_b`, both expansions, the final collapse +
  norm + A8, the same three site boundaries as single site launches, and the two-rank head with its greedy selection.
  The prefill forms are also checked bitwise against `product_q8_0` on sampled rows by their benchmark.
- Kernels: no scratch, no spills.
- Benchmarks (not ctests): `strix-hip-ds41-layer-bench` (decode sites, cold weights), `strix-hip-ds41-layer-prefill-bench`
  (prefill sites, per-row kernels, site launch against the separate chain), `strix-hip-ds41-engram-bench` (cold row
  reads with the real per-token pattern, alone and next to a bandwidth-bound kernel).
