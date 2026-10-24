# deepseek41: long context (128K-512K) on two nodes

How a long prompt reaches the decode forward, what it costs in memory, and how several requests share a KV budget.
Nothing here changes the arithmetic: every row is the decode step of its position (ds41-exact-v1), whichever path
filled the context.

## What a sequence's attention state is

The decode attention of position `p` reads, from earlier positions only:

- per layer, the window ring: the last 128 positions' K/V (FP8 E4M3 with E8M0 per 32, 528 B), in a ring of
  `AttentionHost::ring_capacity` = 136 slots (slot = position % 136);
- on the four KV sources (layers 2, 8, 14 at ratio 2 and layer 20 at ratio 1), the compressed KV (FP4, 288 B per entry)
  and the index keys (FP4, 68 B) of entries `[0, p / ratio)`, and the ratio-2 compressor carry (the pending group's
  first row);
- the Engram n-gram history (the token ids before `p`).

The index sources' top-k of the last row is not read by the next step (it selects again); it is kept only so that state
hashes compare with the existing per-step hashes.

`AttentionState` (`include/strix/deepseek41_attention_state.hpp`) holds exactly this, plus the position and the greedy
token after the last position. Per position it is **890 B** (3 × 356 / 2 + 356: 720 B of compressed KV and 170 B of
index keys); the rings are 2.9 MB whatever the length. 128K = 114 MB, 512K = 448 MB, 1M = 893 MB of device state.

- `DevicePrefill::export_state()` / `import_state()`, `DeviceForward::export_state()` / `import_state()`,
  `BatchForward::import_state(slot, state)` move it between the paths. After `DeviceForward::import_state(s)`,
  `step(s.next_token)` is the step of position `s.position`, bitwise.
- `save_attention_state` / `load_attention_state`: a binary file (magic `DS41ATST`, version 1) with every piece, the
  token history, the state digest and an FNV checksum of every byte; load refuses a corrupt or truncated file.
- `AttentionState::digest(true)` leaves the top-k out: the ranks of a row-split prefill (only the last row's owner
  knows it) and one node then agree.

## Prefill in runs (`DevicePrefill::extend`)

The prefill computes all rows of a run layer by layer, so its row buffers grow with the rows of a run: about 366 KB per
row on a row-split rank (two residual streams of 4 × 5120 floats are 160 KB of it), plus a candidate table of
`rows × positions / 8` bytes. A whole 128K prompt would need ~50 GB next to the 80.2 GiB of weights.

`PrefillOptions::max_rows` sizes the row buffers for one run and `max_positions` the caches for the whole sequence;
`run(tokens)` starts a sequence, `extend(tokens)` continues it at `positions()`:

- the attention state persists between runs (no clearing), the position tables are re-based to the run's first
  position, the Engram history continues (`EngramHasher` kept in the prefill);
- before the window writes of each layer, the layer's ring (the 128 positions before the run) is copied into the run's
  ring (capacity rows + 8 + 128 + 32, so the restored window and the run's rows never share a slot);
- the index counts and RoPE rows use global positions; the candidate table's row stride is per position of the
  sequence;
- the suffix-replay cone is planned per run: decoder layers compute only the rows the run's final state and last
  logits depend on; window attention of the cone's first rows reads the restored ring.

Every row and the final state equal those of one run of the whole prompt. A run's start must be a multiple of
`window_rows` (8). `import_state` lets the next run start from a file, so a 512K prompt fills over several GPU turns.

## Memory per node (weights 80.2 GiB; device used, GiB of 117.2, measured on two nodes)

| stage | 16K | 64K | 128K | 256K | 512K |
|---|---|---|---|---|---|
| decode forward with the state imported (TP attention) | 80.82 | 80.88 | 80.94 | 81.05 | 81.29 |
| prefill, runs of 4K rows (row split) | 82.76 | | | | |
| prefill, runs of 16K rows | 86.93 | 87.10 | | | |
| prefill, runs of 32K rows | | | 93.05 | 93.69 | 94.97 |

The decode state of one 512K sequence is 0.47 GiB. The prefill's peak is set by the rows of a run (~366 KB per row on a
row-split rank) plus the candidate table (rows x positions / 8 bytes); a whole 128K prompt in one run would need ~50 GB
next to the weights, which does not fit, so long prompts run in pieces. The decode forward and the prefill are never
resident together (the prefill's weights are freed before the decode loads).

## Shared KV pool for the batch slots (`BatchForwardOptions::pool_positions`)

Without a pool every slot owns caches for `max_positions`. With `pool_positions = N`, each KV source has one arena of
`(N + 16 × slots) / ratio` entries (KV and index keys); `reserve(slot, positions)` gives a slot a range (first fit,
rounded up to a multiple of 8 plus 8 position units, so every ratio's range starts aligned) and resets it,
`release(slot)` returns it (adjacent free ranges merge). Rings and carries stay per slot (2.9 MB each). Placement in
the arena never changes a value: a slot's kernels see their own range from entry 0.

The server (`strix-deepseek41-serve --pool-positions N`) reserves `prompt + max_tokens + 1` (capped by the context) at
admission and releases at the end; a request that does not fit waits (FIFO) for room, one that can never fit is
refused. Rank 0 sends its reserve/release operations, in order, in the control block; rank 1 replays them before the
step, so both ranks hold the same ranges. A 1M-position pool is 0.9 GB per node (replicated, like the caches).
