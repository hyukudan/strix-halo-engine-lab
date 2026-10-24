# deepseek41 decode attention (CSA2)

Decode-time attention of the deepseek41 model family on gfx1151: host references in `src/ds41_attention.cpp`
(`include/strix/ds41_attention.hpp`) and kernels in `src/hip_ds41_attention.hip`
(`include/strix/hip_ds41_attention.hpp`). New targets only (`strix_ds41_attention`, `strix_ds41_attention_hip`);
nothing on the existing model paths changes. Every call takes 1..8 rows of one verify window (consecutive positions).
The kernels' arithmetic is declared below and the host references implement exactly that order, so device results
equal the host bit for bit; the published reference implementation of the model is the quality yardstick (a small
exported case is compared by `strix-ds41-attention-tests` with `STRIX_DS41_ATTN_CASE=DIR`).

## Shapes and layers

- 64 query heads of 512 (the last 64 dims carry RoPE), one shared KV head (MQA, K = V), a per-head sink logit,
  scale 512^-0.5.
- Window: the last 128 positions, every layer. Compressed KV: layers 2..19 pool 2 tokens per position (KV sources
  2, 8, 14, each serving the following layers until the next source), layers 20..39 one per token (source 20).
- Indexer: 32 heads of 128 at the index sources 2, 8, 14, 20, 24, 28, 32, 36; the other layers reuse the last
  source's top-k. Index keys exist only at KV sources. Layer 20 is the candidate source (2048 blocks of 8 compressed
  positions); 24..36 select inside its candidates. `released_plan()` gives every layer's sources and roles.
- RoPE pairs are (2i, 2i+1) of the last 64 dims; window-only layers use theta 10000, compressed layers theta 160000
  with YaRN (factor 16 over 65536, beta 32/1); frequencies and cos/sin are computed in double on the host
  (`rope_row`) and passed per row. A compressed position j carries the RoPE of its group's first token.

## Storage

| cache | entry | layout |
|---|---|---|
| window ring (per layer, `capacity >= 128 + rows` entries, slot = position mod capacity) | 528 B | 512 E4M3 codes, 16 E8M0 scales (one per 32) |
| compressed KV (per KV source, index = position / ratio) | 288 B | 256 B of E2M1 pairs (low nibble = even dim), 32 E4M3 scales (one per 16) |
| index keys (per KV source) | 68 B | 64 B of E2M1 pairs, 4 E8M0 scales (one per 32) |

Quantisation (float32, the scale rules of the reference):
- E4M3 window: amax over 32 (at least 1e-4); s = 2^ceil(log2(amax * (1/448))); code = E4M3(clamp(x / s, ±448)).
- E2M1 KV: amax over 16 (at least 6·2^-9); scale code = E4M3(amax / 6), s its value; code = E2M1(clamp(x / s, ±6)).
- E2M1 index keys and queries: amax over 32 (at least 6·2^-126); s = 2^ceil(log2(amax * (1/6))).
- E4M3 and E2M1 encoders round to nearest even and saturate.

## Declared arithmetic (all float32, no contraction except explicit fmaf)

A 512 vector lives in one wave: lane l owns dims 16l..16l+15. "Butterfly" is v = v + shfl_xor(v, d) for d = 16..1
(every lane ends with the same value). `exp_f` is a portable exp (rint, two-step Cody-Waite, degree-8 polynomial with
fmaf, power-of-two scaling; 0 below -87): host and device evaluate it identically. Divisions and square roots are
correctly rounded.

- RMS norm (512): lane chain acc = fmaf(x, x, acc) over its 16 dims; butterfly; r = 1 / sqrt(sum / 512 + 1e-20);
  y = (x * r) * w.
- Window write: norm(wkv(x)) with the layer's KV norm, RoPE, E4M3 quantisation.
- Compression, ratio 2: per dim m = max(s0, s1), e0 = exp_f(s0 - m), e1 = exp_f(s1 - m), sum = e0 + e1,
  pooled = kv0 * (e0 / sum) + kv1 * (e1 / sum) (token 0 is the group's first). The incomplete group is carried in a
  state buffer between calls. Ratio 1 takes wkv(x) itself. Then latent = norm(pooled) with the compressor norm.
  - Index key: k[o] = fmaf chain over d = 0..511 of f16(wk[o][d]) * latent[d] (lane l owns o = 4l..4l+3); RMS norm
    over 128 (lane chain over its 4, butterfly, / 128); RoPE on dims 64..127; E2M1 / E8M0 quantisation.
  - KV entry: RoPE on the latent's dims 448..511, then E2M1 / E4M3 quantisation.
- Index scores (row r at position p, count = (p + 1) / ratio): q = wq_b(qr) per head with RoPE on dims 64..127,
  quantised to E2M1 / E8M0 per 32; w_h = weights_proj(x)_h * 1/64. For every position t < count, per head h:
  dot_b = exact integer sum over the block's 32 products of the (2 · E2M1) integers;
  term_b = ((dot_b * sq_b) * sk_b) * 0.25; d_h = ((term_0 + term_1) + term_2) + term_3;
  score = fmaf(fmax(d_h, 0), w_h, score) for h = 0..31 starting from 0.
- Selection (every count; keys ordered as floats, ties by the smaller position):
  - candidate source: block score = max over its positions, the block holding the newest position pinned to +inf; the
    best 2048 blocks (those above -inf) become the candidates (one byte per block);
  - candidate users replace non-candidate scores by -inf;
  - top-k: all positions when count <= 512, else the 512 best; written in ascending position order.
  - The device finds the k-th key with a 4-pass radix histogram and compacts in position order with block scans
    (integer counts only), so the selection is exact.
- Attention (row r, head h): q with RoPE; visible keys = window positions max(0, p - 127)..p, then the selected
  compressed positions, in that order; keys dequantised exactly (code value × scale).
  - Chunks of 64 keys (the split across blocks), each processed in tiles of 16 keys with an online softmax:
    s_j = butterfly(lane fmaf chain over its 16 dims of q · k_j) * scale; tile max tm (j ascending);
    m' = max(m, tm), c = exp_f(m - m') (0 for the first tile); l = l * c, then l = l + p_j with p_j = exp_f(s_j - m')
    in key order; acc = acc * c, then acc = fmaf(p_j, k_j, acc) in key order; m = m'.
  - Combine in chunk order: M = max m_c; e_c = exp_f(m_c - M); L = fmaf(l_c, e_c, L), then L = L + exp_f(sink - M);
    o = (fmaf chain of acc_c * e_c) / L; inverse RoPE on dims 448..511.

## Kernels (gfx1151)

| kernel | launch | VGPR | LDS |
|---|---|---|---|
| window write | one wave per row | 42 | 0 |
| compress (pool, norm, KV entry; 128 index-key chains over 4 waves) | 128 threads per row | 154 | 2.5 KB |
| index query prep (RoPE, E2M1, scales, head weights) | 256 threads per row | 40 | 0 |
| index scores (thread = position, key decoded once for all rows, `v_dot4` with the queries as scalar operands) | 256 positions per block | 48 | 8 KB |
| select, rows up to 4096 positions (per-wave radix histograms, ordered compaction) | 1024 threads per row | 122 | 32 KB |
| select, longer rows: level one (exact top-k of each 4096-position slice, block maxima) | 1024 threads per slice and row | 106 | 48 KB |
| select, longer rows: level two (top-k over the slices' survivors; candidate blocks) | 1024 threads per row (and one more for the candidates) | 122 | 32 KB |
| attention chunk (tile of 16 keys dequantised once per block into LDS; wave = 4 heads; transposed DPP butterfly; the last block of a head group combines when tickets are given) | 10 chunks x (heads / 32) x rows, 256 threads | 240 | 33 KB |
| attention combine (when no tickets) | one wave per (head, row) | 168 | 0 |

How the kernels keep the declared order while running fast:
- **Attention scores** are reduced 8 keys at a time by a transposed xor butterfly (DPP `row_xmask` for xor 1-8,
  `permlanex16` for xor 16): at xor 16/8/4 every lane keeps half of its keys and adds the partner's partial of the
  same key, which are the additions of the per-key butterfly; key j ends in lanes 4j..4j+3.
- **LDS key rows** put float4 i of lane l at i * 32 + l, so a wave's reads are conflict-free.
- **Two-level selection** is exact: every element of a row's top-k is in its slice's top-k (fewer than k elements
  of the slice beat it), and slices are padded with key 0, below every float's key.
- **Tensor parallel:** `AttentionView::heads` (a multiple of 32) runs a subset of heads, with q, sink, o and the
  partials of those heads contiguous; per head the result is the same bits.

The timing build (`STRIX_DS41_ATTN_TIMING`, target `strix-hip-ds41-attention-timing`) records per block the wall
clock and shader cycles, plus phase marks inside the attention chunk kernel.

No kernel spills. Tests: `strix-ds41-attention-tests` (CPU: formats, exp, plan, selection against a sort, attention
against double precision on the same keys, optional exported reference case) and `strix-hip-ds41-attention-tests`
(device: every kernel equal to the host bit for bit; `STRIX_DS41_ATTN_BENCH=1` adds timings).
