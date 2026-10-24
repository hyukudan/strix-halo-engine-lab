# deepseek41 DSpark drafter

Status: host reference and device kernels of the draft model's own layers, bitwise equal on the real
weights; greedy verify protocol and window choice; not yet wired to the target model (the target's
hidden states and verify windows come with the target runtime). Files: `include/strix/ds41_dspark.hpp`,
`src/ds41_dspark.cpp` (host), `include/strix/hip_ds41_dspark.hpp`, `src/hip_ds41_dspark.hip` (device),
fixtures `tests/ds41_dspark_tests.cpp` (CPU) and `tests/hip_ds41_dspark_tests.hip` (GPU). The routed and
shared experts are the MoE window of `moe.md`.

## Model interface

The draft file (`deepseek41-dspark`) holds three stages `mtp.0-2`; it has no embedding or output head and
uses the target's `token_embd` and `output`.

- **Input of a committed target row:** for each of target layers 37, 38 and 39, the mean over the four
  hyper-connection copies of that layer's input stream; concatenated (15,360 values). `main_x =
  main_norm(main_proj(.))` (Q8_0 15,360 -> 5,120, RMSNorm).
- **Per-stage ring:** every stage keeps `kv_norm(kv(main_x))` with RoPE at the row's position, as an FP8
  entry (512 E4M3 codes and 16 E8M0 scales, one per 32), in a ring of 128 entries (position p in slot
  p mod 128). All three stages read the same `main_x`; each has its own `kv` projection and ring.
- **Draft step** from the last committed token t at position p (the rings hold positions up to p - 1):
  - block of five rows at positions p..p+4: t, then four noise tokens (id 128799), embedded with the
    target's table and expanded to the four copies;
  - three stages, each a full block with the hyper-connection mixes (Sinkhorn), window-only attention
    (theta 10,000, no YaRN; 64 heads of 512, one shared K = V head, per-head sink) over the ring's
    positions plus the five block rows, every block row seeing all five (bidirectional inside the
    block), and the MoE (128 MXFP4 experts top-3 plus the Q8_0 shared expert);
  - head: collapse with the last stage's FFN pre coefficients, final RMSNorm, the target's output head;
    block row i (i = 0..4) gets the Markov bias `head_m * embed_m[token_i]` (rank 256, F16) where
    `token_0 = t` and `token_{i+1}` is the argmax of row i; five drafted tokens for p+1..p+5;
  - confidence of row i: `w . [hc_pre(x)_i, embed_m[token_i]]` (F32 weight of 5,376).
- The attention of a stage collapses the copies with the previous sub-layer's pre coefficients (identity
  for stage 0), as the target's blocks do.

## Numerical identity `ds41-dspark-v1`

Declared arithmetic, host = device bitwise:
- Q8_0 linears: the A8 activation contract and the declared Q8_0 product of `formats.md`. The grouped
  output projection is eight 4,096 -> 1,024 products, one per head group, run as a routes product.
- Short F16/F32 dots (confidence, attention scores): FP32 products in the wave order (lane l sums terms
  l, l+32, ... from 0 in ascending order, then the butterfly `v[l] += v[l ^ d]`, d = 16..1).
- Long sums (RMSNorm sums of squares, hyper-connection mix dots and stream statistic): the chunked order
  `chunk_sum` — chunks of 640 terms, each in the wave order (one wave each), chunk sums added in
  ascending order; `rstd = 1 / sqrt(mean + 1e-20)` with the correctly rounded root.
- Markov bias: `markov_dot` (lane l takes elements 8l..8l+7 ascending, then the butterfly).
- Sinkhorn split: `exp` is the declared exponential of `ds41_moe.hpp`; sums of four in the order
  `((a + b) + c) + d`; 20 column and 19 row normalizations as in the model.
- `hc_pre`: `((p0 x0 + p1 x1) + p2 x2) + p3 x3`; `hc_post`: `post_k x + (((c0k s0 + c1k s1) + c2k s2) + c3k s3)`.
- RoPE: cos/sin computed in double on the host and rounded once; pairs of adjacent elements, no fused
  operations.
- FP8 entries: per group of 32, scale `2^ceil(log2(max(amax, 1e-4) / 448))`, codes E4M3 round-to-nearest-even
  of `clamp(x / scale, +-448)`.
- Attention per (row, head): scores `wave_dot(q, k) * 512^-0.5` over the keys in order (ring positions
  ascending, then block rows); `m = max`; `e_j = exp(s_j - m)`; `sum = (sum_j e_j) + exp(sink - m)` in
  key order; `o[d] = (sum_j e_j v_j[d]) / sum`; inverse RoPE of the output.
- Argmax: larger value first, lower index on ties.

The drafter's numerics never change output tokens: every drafted token is verified by the target.

## Verify and rollback

A verify window has rows `t, d_1..d_k` at positions p..p+k (k <= 5, so up to six rows). With `g_j` the
target's greedy token after row j, the accepted count a is the longest prefix with `d_j = g_{j-1}`;
rows 0..a are committed, the target's state of rows a+1..k is rolled back, and `g_a` is the next t at
position p+a+1 (`verify_greedy`). Tokens equal target-only greedy decoding by construction.

The drafter needs no rollback: block rows never enter a ring, and the main rows of the window are
appended only after the verify, for the committed rows only (`append` / `append_hidden`). Rows of
target-only steps (no draft) are appended the same way, and can be batched until the next draft step.

`choose_drafts(confidence, cost, min)` picks k by expected tokens per cost, with the survival
`prod sigmoid(confidence)`: the confidence head is an input to the window choice, never the acceptor.

## Device path (round 2)

- **MoE routes on the device:** `build_routes_kernel` writes the CSR of `build_window_routes` (same order,
  same combine order) from the router's ids into persistent buffers; the union is padded to rows x top_k
  slots and the window kernels skip slots without entries. No host sync or allocation per draft step
  (`STRIX_DS41_DSPARK_GPU_ROUTES=0` restores the host routes).
- **Wide-row Q8_0 kernel:** when the A8 rows of a linear do not fit in LDS (main_proj, width 15,360; out_b,
  width 8,192 with five rows), `q8_rows_kernel` reads the weight once for up to 8 rows and the activations
  from memory; per (row, output) it is the declared Q8_0 product, so the bits do not change
  (`STRIX_DS41_DSPARK_WIDE_ROWS=0|1|2`: off, when needed (default), every multi-row linear).
- **Markov chain:** one fused pass per step over the vocabulary rows (one wave per row, a 16-byte load per
  lane: `markov_dot` order, lane l takes elements 8l..8l+7) with a per-block argmax and a pick kernel.
  Certified pruning by norm bounds was measured and dropped: the Markov bias is as large as the logit
  spread (std 3-5, up to 38) and nearly full rank, so Cauchy-Schwarz keeps all 129,280 rows and a rank-128
  split still keeps 18-116 K.
- **Draft vocabulary** (`Vocab`, `DeviceDrafter::set_vocabulary`): the ids below `core` plus up to 8,192
  extra ids (context or language-pack tokens, rows gathered from the resident head). A reduced vocabulary
  is a separate draft identity; output tokens stay the target's greedy tokens.
- **Appends from device memory:** `append_hidden_device` (the target window's captured rows).

## Draft vocabulary, two ranks and the survival controller

- `ds41_dspark_vocab.hpp`: draft vocabularies are id lists (core16k..64k, per-language packs). `auto:<dir>` picks
  the pack whose 16k list holds the most prompt tokens (at least 2 and 5 % of the prompt): core32k + pack;
  otherwise core48k. Prompt tokens and every committed token are added as context rows
  (`Drafter::add_context`, `DeviceDrafter::add_vocabulary`). `STRIX_DS41_DSPARK_VOCAB=full|<N>|auto:<dir>|files:...`.
  A restricted vocabulary is a draft-only identity; the target's output is unchanged.
- `DeviceDrafter::set_rank(rank, 2, exchange)`: each rank evaluates half of the vocabulary rows (head product and
  Markov pass) and exchanges an 8-byte candidate (value, id) per Markov step; both ranks keep the better one (larger
  value, lower id), so the tokens equal the one-rank draft. `strix-ds41-dspark-rank` runs one rank over the
  communicator (USB4 stream devices of the detected topology, or TCP for debugging).
- `SurvivalController` (`STRIX_DS41_DSPARK_WINDOW=survival[:c0,..,c5]`): per-depth P(draft correct | earlier
  drafts correct) from 8 confidence bins, learned online from verify outcomes; the window maximizes expected
  tokens per cost, with periodic full windows for exploration.

## Speculative driver and flags

`ds41_dspark_driver.hpp`: `SpeculativeTarget` (step a token, give the drafter's input row of that step),
`Drafter` (host `HostDrafter`, device `DeviceDrafterAdapter`), `generate_greedy` and
`generate_speculative`. The driver verifies a window row by row and feeds a draft only after the previous
draft matched, so the target needs no rollback; it is the protocol reference and the on/off test.
Flags: `STRIX_DS41_DSPARK=1` (drafts on), `STRIX_DS41_DSPARK_WINDOW=fixed:N` (N = 0..5, default 5) or
`auto[:c0,..,c5]` (`choose_drafts` on the confidence head with the window costs).

## Tests

- CPU (`strix-ds41-dspark-tests`): F16 and E4M3 codes, FP8 entries, the wave order, Sinkhorn and block
  attention against double-precision oracles, RoPE, the verify protocol and a speculative decode
  simulation (four window policies, tokens identical to greedy). With `STRIX_DS41_Q2_GGUF` and
  `STRIX_DS41_DSPARK_GGUF` it runs the host drafter on the real weights (`STRIX_DS41_DSPARK_DUMP=<dir>`
  writes inputs and outputs).
- GPU (`strix-hip-ds41-dspark-tests`, through the node's GPU turn): every kernel against the host on
  synthetic data; with the model files, the whole drafter on the real weights: main rows, ring seeding
  over 130 positions (wrap-around), draft steps at positions 130, 133 and 7 (stage streams, head input,
  Markov embeddings, logits, tokens, confidence), appends after a verify, and a smoke timing.
