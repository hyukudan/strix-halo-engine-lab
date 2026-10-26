# minimax-m3 sparse attention (MSA): semantics and kernel design (phase 1)

Status: host reference (`include/strix/minimax_msa.hpp`, `src/minimax_msa.cpp`) and decode kernels K1-K4
(`include/strix/hip_minimax_msa.hpp`, `src/hip_minimax_msa.hip`), bit-exact against the host on gfx1151
(`strix-hip-minimax-msa-tests`), CMake option `STRIX_FAMILY_MINIMAX`, OFF. Applies to MiniMax-M3.1-Flash: MSA in all 60 layers.

## Semantics (source of truth)

The checkpoint ships no modeling code. The reference is SGLang at revision
`299a75ac5d962087f8f09e19b23ff43548fc64c8` (read, not copied):

- `python/sglang/srt/models/minimax_m3.py`: `MiniMaxM3Attention` (q/k/v + `index_qkv_proj` with 4 index query heads and
  ONE index key head, `disable_index_value` = no index value/output), per-head Gemma RMS norm (scale 1 + w) on q, k,
  index q and index k, the same neox RoPE (`get_rope` default) on the first 64 of 128 dims for main and index heads,
  theta 1e7. Residual block: `h += attn(norm(h)); h += ffn(norm(h))`, Gemma norms.
- `layers/attention/minimax_sparse_backend.py` and `minimax_sparse_ops/minimax_sparse.py`: decode = index scores ->
  top-k block ids -> (reduce over index heads per KV head: group size 4/4 = 1, so index head g selects for KV head g)
  -> attention of the 16 query heads of KV head g over the selected blocks. GPU prefill uses `block_size_q = 1`
  (per-token selection).
- `kernels/ops/attention/minimax_sparse/decode/flash_with_topk_idx.py` (`_decode_score_kernel`,
  `_topk_index_*`) and `prefill/flash_with_topk_idx.py` (`_topk_index_kernel`): block score = max over the block's
  causal positions of q.k (times 128^-0.5 * log2 e, monotonic), blocks `< init_blocks` (0) pinned to 1e30, the
  newest block (the one holding the query) pinned to 1e29, top-16, FP32 scores. With 16 or fewer causal blocks every
  block is attended (dense attention up to 2048 tokens).
- Main attention: softmax scale 128^-0.5, no sink, causal inside the newest block.

Every point above is checked numerically by `tools/minimax/cpu_layer_reference.py` (oracle written from that code vs
our implementation on real activations of the EXL3 pack).

## Differences from the deepseek41 CSA2 path (docs/ds41-attention.md)

| piece | deepseek41 CSA2 | minimax-m3 MSA | reuse |
|---|---|---|---|
| index heads | 32 heads x 128, ReLU, per-head weights, ONE summed score | 4 heads x 128, plain q.k, one score PER HEAD | new score kernel (simpler) |
| index keys | FP4 E2M1 + E8M0/32, per compressed position, at KV sources only | one 128 key per token per layer, BF16 in the reference (FP8 optional) | new cache layout |
| q/k prep | RMS norm (512 latent), RoPE pairs (2i, 2i+1) on dims 64..127 | Gemma RMS norm per 128 head, neox halves on dims 0..63 | same lane/butterfly scheme, different pairing |
| selection unit | top-512 POSITIONS (two-level, candidates of 8) | top-16 BLOCKS of 128 per index head | `radix_threshold` + `ordered_select` as in the `candidate_source` pool (block max, newest block = +inf, ties by smaller id), k = 16 |
| selections per row | one shared by all heads | 4 (one per KV head) | per (row, head) launch index |
| main attention | MQA 64 x 512, K = V, sink, FP8/FP4 keys, window ring of 128 + selected | GQA 64/4 x 128, separate V, no sink, BF16 K/V, the selected 16 blocks only (the newest block replaces the window) | online softmax/tile/chunk/combine structure and `exp_f` reused; head 128 = 4 dims per lane |
| TP | heads split, indexer shared | rank r owns KV heads 2r, 2r+1, index heads 2r, 2r+1; index key replicated | **no exchange for MSA** |

## Declared arithmetic (host reference = future device)

All float32, no contraction except explicit `fmaf`. A 128 vector lives in one wave: lane l owns dims 4l..4l+3. Lane
chain = `acc = fmaf(a, b, acc)` over its 4 dims; butterfly = `v += v[lane ^ d]`, d = 16..1.

- Head norm + RoPE: `r = 1 / sqrt(butterfly(chain x*x) / 128 + 1e-6)`, `y = (x * r) * (1 + w)`; i < 32:
  `y_i' = fmaf(-y_{i+32}, sin_i, y_i cos_i)`, `y_{i+32}' = fmaf(y_i, sin_i, y_{i+32} cos_i)`; cos/sin in double per
  position.
- Index block score: `max_t dot128(iq_h, ik_t) * 128^-0.5` over the block's visible positions (max is exact, order
  free).
- Selection: all blocks if at most 16; else the newest block + the best 15 others by score, ties by the smaller block
  id; ascending output.
- Attention per (row, KV head g, query head): keys in ascending block then position order, chunks of ONE block, tiles
  of 16 keys, online softmax with `exp_f`, chunk combine in chunk order (header comment has the exact formulas).

## Kernels (gfx1151, wave32), decode R1..R8

Per layer and rank (2 KV heads, 2 index heads, 32 query heads). All equal to the host bit for bit; no spills.

| # | kernel | launch | VGPR / LDS |
|---|---|---|---|
| K1 | `prep`: head norm + RoPE for q, k, index q, index k; K/V/index-K written to the BF16 caches | one wave per (row, head) | 19 / 0 |
| K2 | `scores`: thread = position, the key read once for every row and index head, dot128 evaluated in the declared order by one thread, block max by wave shuffles + LDS | one 128-thread block per 128 positions | 195 / R x heads x 512 B |
| K3 | `select`: exact radix threshold on order-preserving keys (-0 canonicalised), newest block forced, ties by smaller id, ascending compaction by block scans | one 256-thread block per (row, head) | 34 / 33 KB |
| K4 | `attention_chunk`: one block per (row, KV head, selected block), wave = query head; lane = key for the scores (dot128 in declared order, no shuffles), tiles of 16 keys walked in order with broadcasts; `attention_combine` merges the 16 chunk partials | R x kv x 16 blocks of 512 threads; one wave per (row, head) | 115 / 8 KB |

Measured kernel time per layer and rank (one observation, `STRIX_MINIMAX_MSA_BENCH=1`, one node, no model):

| context | rows | prep | scores | select | attention | total |
|---|---|---|---|---|---|---|
| 8K | 1 | 5.1 us | 8.5 | 9.9 | 48.2 | 71.8 |
| 8K | 8 | 5.1 | 23.2 | 10.1 | 189.3 | 227.7 |
| 32K | 1 | 5.1 | 16.4 | 25.6 | 62.9 | 109.9 |
| 128K | 1 | 5.4 | 82.0 | 241.8 | 83.6 | 412.7 |
| 128K | 8 | 5.6 | 240.3 | 258.1 | 299.0 | 803.0 |

With the radix K3 (measurement): select 19.6 us at 8K R1, 19.6 at 32K, 21.1 at 128K (R1 128K total 181 us). Rank selection
was faster below ~2K blocks of 4 passes' fixed cost; a size switch (both exact) is the next small step.

Next: K4 with the 16 heads sharing one K/V tile load per key group; K1 fused into the projection epilogues.

Bytes per decoded token and node at 8K, BF16 caches: index scan 0.126 GB + selected K/V 0.126 GB over 60 layers;
at 128K 2.0 + 0.13 GB (the scan dominates beyond ~32K: an FP8 index key (E4M3 + E8M0/32, 132 B) halves it; FP4 like
DS41 needs a quality gate since the M3.1 indexer is not QAT'd for it).

## Prefill

- Index scores as a WMMA GEMM tile: [query tile x 128] x [128 x key tile] with an in-register block max -> scores
  [T x ceil(T/128)] per index head (at 32K: 32768 x 256 floats per head, 32 MB; stream it per query tile).
- Selection per token (K3, rows = query tile).
- Main attention: per query tile of 16 tokens and KV head, the union of the tile's selected blocks (the newest 1-2
  blocks are shared; the rest overlap strongly at long context), keys loaded once per union block, a per-token mask
  for blocks the token did not select. The per-token result must equal the decode declared order (same chunking of the
  token's own ascending block list), so the exact prefill uses per-token chunk partials; a faster non-exact tile
  order needs its own gate.

### Window batching (draft; qualification pending)

The engine retains windows starting at chunk rows 0, 16, 32, ... . Two startup switches default off:
STRIX_MM_PF_ATTEND4 selects attend v4 with two heads per wave in fused prefill windows;
STRIX_MM_PF_MSA_BATCH selects fused::enqueue_prefill. Engine setters expose both switches.
The pfattn4 and pfmsa A/B flags run seven alternating prefills with cache-state and final-row FNV checks
enabled. Decode scheduling is unchanged.

The batched schedule first writes every chunk K/V/index-key cache row with the existing lane norm, RoPE and
BF16 packing expressions. One persistent index launch then scores all sparse windows, and one select/attend
launch processes them. Dense windows (last position below 2048) use a separate attend launch. Each window has
its own score, selection, count, partial and ticket regions; combining still visits partials in chunk order.
No inter-window ticket or counter is shared. Subsequent scratch batches run on the same stream.

The index kernel still loads a tile and overwrites its own window's slots in LDS using the original norm/RoPE/pack
operations. Earlier visible slots already have their final bits. Slots after the window's last position are zeroed
by the load predicate; each row's score predicate further excludes keys after that row. Thus preparing later
windows first does not change a scored operand or the per-block maximum chains. Cache stores in the scoring
launch are disabled for batching to avoid cross-window writes.

Attend loads only the selected block's keys below the row's visible count. In the sparse path, K/V were already
prepared before attend in the window schedule. In the dense path, the old kernel overwrote stale LDS slots with
its window's packed rows; batching loads those same packed bits directly and suppresses the redundant cache/LDS
writes. The newest block, ascending selected blocks, softmax tiles, partial addresses and combine order remain
the same. V4 shares reads across heads while retaining each head's arithmetic order.

Startup scratch is capped at 512 MiB and printed. One 16-row window needs
16 * (2 * max_blocks + 2 * 16 + 2 + 32 * 16 * 130 + 2 * 4) * 4 bytes. Oversized chunks are processed in
whole-window scratch batches, with one cache-prep launch for the entire chunk. For an 8192-position capacity,
125 windows fit (533840000 bytes). An 8K prompt has 128 dense and 384 sparse windows: the old schedule uses
896 launches per layer, or 53760 over 60 layers. With default 4096-row chunks, batching uses 14 per layer (840
total); a single 8192-row chunk uses 11 (660 total). With enough scratch for every window, a mixed chunk uses
four launches. These counts describe MSA launches only and do not establish a measured speedup.

The fused HIP tests compare v3 with v4/hpw2 for every window length 1..16 around positions 127/128/129 and 2048.
Batched tests compare every output/cache byte for starts 0, 1000, 2040, 4096, 8190 and lengths
1, 15, 16, 17, 100, 1024, 4096, using both attend versions, bounded scratch and forced one-window scratch.
They restore finite stale cache bits before each arm, check ticket reset and print chunk times. HIP compilation,
native bitwise execution, model fidelity and complete-workload qualification remain pending; neither switch
is qualified by launch-count estimates.

## Exactness gates for the device work

1. Device = host bit for bit on synthetic and on the real exported case (`tools/minimax/export_msa_case.py`,
   `STRIX_MINIMAX_MSA_CASE=DIR strix-minimax-msa-tests`).
2. Host reference vs the float64 oracle on real data: selection identical when the oracle's 16th/17th block margin is
   above the float32 noise; attention rel error <= 1e-5.
3. Model level: the selection is discrete, so FP16 activations / BF16 caches flip some selections (measured in the
   phase 1 report); the model gate must be teacher-forced KL / top-1 against the oracle, not bitwise.

## Phase 3 additions (all bit-exact against the host references)

- **Hybrid selection (K3).** At most 256 candidates: exact rank by counting (one thread per block, the AMD/ATOM
  count-based top-k); more: the 4-pass radix threshold. Same total order (score desc, block id asc) either way.
  Measured select at 8K R1: 19.6 -> 6.4 us (measurement).
- **Context-parallel exact indexer.** For long contexts on the pair: each rank scores ALL index heads over the blocks
  `b = j * 2 + rank` (`ScoreView::shard_rank/shard_count`), keeps its best 15 blocks per (row, head) as
  (order key, global id) (`enqueue_candidates`), the head's owner merges both lists and adds the newest block
  (`enqueue_merge`). Exact because every block of the global top 15 is in its shard's top 15. Tested: CP selection ==
  single-rank selection bit for bit up to 512K positions. Exchange: 15 x 8 B per row and head. Threshold to switch it
  on (a long-context flag, OFF by default) comes from the cost model in the article review: about 55-65K tokens with
  a BF16 index key.
- **Shared K tile in LDS (tried, rejected).** One LDS key tile per (row, KV head, block) for the 16 heads was slower
  (R1 8K 48 -> 55 us, R8 189 -> 209 us): the keys were already L2 hits and the barrier cost more. The useful sharing
  for R8 is across rows that selected the same block; that needs a (KV head, block) -> rows schedule, not done yet.
- **Fused post-exchange (`hip_minimax_moe.hpp`).** One kernel per row after each TP exchange: h += part0 + part1
  (fixed rank order, so both ranks keep the same residual bits), the next sublayer's Gemma norm, and for MoE layers
  the FP32 router, sigmoid, + bias, top-4 by (choice desc, id asc) and the renormalised weights x 2. Used by the G4
  gate. Rows are independent: the same row gives the same bits in a call of 1 or 8 rows (tested).
- **Correctness rules from the article review.** (11) Batch invariance: every new kernel computes a row from that
  row's data only (MSA, post-exchange); the MoE sums of the gates run per expert in ascending expert id per token.
  (12) No kernel publishes a flag to another block: completion is always the kernel boundary on the stream (the
  combine is its own launch), so no store-drain/fence protocol is needed yet; a ticket-based fused combine would have
  to drain its stores (`s_waitcnt vmcnt(0)` + release fence at agent scope) before the ticket.

## Draft prefill index query tiles

`STRIX_MM_PF_IDX_TILE=1` (default off), or `Engine::set_prefill_index_tile(true)`, enables
batched prefill for chunks larger than eight rows and reuses each loaded index-key tile across consecutive query windows.
`STRIX_MM_PF_IDX_ROWS` is read once at engine startup: 16, 32 (default), or 64; other values fail closed.
The effective cap is 32 rows: 32 KB of keys plus 32 KB of normalized FP32 index queries fill 64 KB of LDS.
After all threads read their packed key into registers, the key LDS region holds the wave maxima.
A barrier separates those reads from the maxima writes. Requested Q=64 therefore executes Q=32.

Each output retains the same four-FMA lane chains, dot reduction, scale, shuffle-max tree and
wave-max order. Per-row causal masks and the skip for wholly invisible blocks are preserved.
The prepared cache supplies the same packed index keys; score addresses remain in each original
16-row window scratch region. Decode kernels and their dispatch are unchanged.

`--ab pfidx:PROMPT.u32:BLOCK` runs seven alternating prefills, with batched prefill in both arms,
state and final-row hashes, and restores the prior switches afterward. It is excluded from `pfall`.
The HIP tests compare score bytes (including untouched slots), selected IDs, counts, outputs and
cache bytes at Q=16/32/64, including tied maxima, boundary positions, scratch reuse and 30K offsets.
They print per-chunk sparse-index launch time for each arm, excluding prep and attention.
This path is a draft pending compiler, native execution, model fidelity and complete-workload
qualification on both ranks; no local GPU qualification or performance claim is made.
