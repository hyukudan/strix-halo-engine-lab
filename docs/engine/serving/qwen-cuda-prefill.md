# CUDA Qwen prefill scheduling and tracing

The resident service computes rows `[0,P-1)` under the A8 prefill identity. The
node executes the final prompt row. The repeating block sizes are
`1024,2048,4096,8192`. Scheduling controls never change numerical identity:
every schedule produces bit-identical state and greedy continuations.

The default block schedule is `tail-large`: a short final remainder block moves
to the front when it is smaller than its predecessor, so the last block (whose
transfer cannot overlap later compute) is a large one. Row order stays
chronological: only the sizes move. With 32767 rows the plan becomes
`1023,1024,2048,4096,8192,8192,8192`; with 8191 rows it becomes
`1023,1024,2048,4096`. The block count, total rows and maximum capacity are preserved.
An already larger final block stays in place. Measured on the RTX PRO with the
node import (same greedy IDs as node-only prefill in every request), TTFT drops
12.7 % at 1K, 7.9 % at 8K and 5.0 % at 32K against the original order.
`--block-schedule default` (or `"block_schedule":"default"` per request) turns it
off and restores the original order.

`--final-block-rows N` optionally reserves a specific tail size after that
transformation, moving rows between existing blocks without adding blocks. Zero
disables the reservation. The feasible tail range is
`max(1,position-(count-1)*capacity)` through
`min(capacity,position-count+1)`; the target is clamped to that range. A one-block
prompt stays one block. A target above the loaded maximum is rejected. For
example, `--blocks 1024,8192 --final-block-rows 7167` gives `1024,7167` at 8191
rows. Moving from the default four-block 8191-row plan to two blocks would change
the block count, so that example requires this explicit base schedule.

POST `/v1/prefill` accepts `"block_schedule":"default"|"tail-large"` and
`"final_block_rows":N`. Omitted fields inherit startup defaults; explicit
`"default"` plus `0` restores the baseline within the same loaded service.
Overrides cannot enlarge the loaded capacity. Invalid overrides return 400
before contacting the node. `/health` reports the startup policy and capacity.

Every block contains 1 through the configured maximum real rows, bounded by
65520 (the padded row grids must fit CUDA grid y, whose limit is 65535). Row-independent operations pad to at least
16 rows and a multiple of 8; stateful operations see only real rows. Activation,
GDN scratch, MoE CSR and MTP workspaces are allocated once from the padded maximum
of `--blocks`. QSA uses a bounded row sub-batch workspace with full position
capacity. Scheduling cannot grow these allocations.

Boundary invariance follows the A8 row-local projections and fixed reduction
order, chronological GDN recurrence, and persistent KV/index and convolution
state. GDN carries its FP32 matrices and four BF16 convolution history entries;
PLE carries two token history words and nine convolution entries. Neither is
reset between blocks. QSA stores and attends at absolute positions. MTP pairs
use token `i+1` even when it lies across a block boundary, store at absolute `i`,
and export exactly `position-1` pairs plus the final held wide row. Padding never
advances those histories. CPU scheduling tests establish coverage and capacity,
not device bit equality. Complete fixture comparisons and greedy workload gates
are required for each candidate schedule.

Set `"trace":true` to receive a `trace` object. All event `ms` values share a
request-local `steady_clock` epoch created at worker entry. No trace clock reads,
callbacks or event storage occur for requests without tracing; existing aggregate
service/driver timers are retained.

* `gpu_ready` and `d2h_done` use stream-ordered host callbacks. Per-layer readiness
  is observed after the existing layer release point, not the earliest possible
  internal operator release. MTP readiness follows its compute span; the held row
  is marked after main-model compute. Other readiness precedes D2H at the current
  block/fixed release point; host history words use equivalent host marks.
* `piece` supplies component, first row/byte offset, row count and byte size.
  Its `id` joins readiness, D2H, enqueue and chunk membership. Split sender pieces
  share the producer id and carry their own range/bytes.
* `chunk_piece` has a `chunk` sequence number; encode and POST events use that
  sequence as `id`. `encode_start/end` include packing, codec and SHA work.
  `post_start/ack` bracket the complete HTTP exchange.
* `commit_wait_start` precedes drain; `commit_start/end` bracket commit work after
  all chunk acknowledgements.
* `block_start/end` give absolute boundaries. `trace.blocks` contains `elapsed_ms`
  and `compute_ms`: the sum of main-model and MTP compute-stream callback spans,
  excluding subsequent D2H and the emitter join between those spans. These are
  observed stream wall intervals, including launch starvation and instrumentation,
  rather than a sum of kernel durations. `elapsed_ms` includes copies and waits.

Callbacks and serialization perturb traced requests. Use untraced interleaved
runs for TTFT comparisons and separate traced runs to explain overlap. A nonzero
`dropped_events` means the trace is incomplete: an allocation failed inside a
callback, or the request exceeded the 1,048,576-event cap. Timestamps from different machines must not be subtracted.

The isolated Windows build uses `STRIX_CUDA_QWEN_ONLY=ON`, CUDA architecture 120
(pass `-DCMAKE_CUDA_ARCHITECTURES=120`; the top-level default also lists 89 and 86),
and the exact-arithmetic flags `--fmad=false --prec-div=true --prec-sqrt=true
--ftz=false`. It builds `strix-cuda-qwen-prefill-service`, `qwen_prefill_e2e`, and
CPU-only `qwen_prefill_schedule_test` and `qwen_handoff_codec_test`. CTest runs
only those two CPU checks. The gate runs the tail-large default;
`STRIX_QWEN_TAIL_LARGE=0` restores the original order and
`STRIX_QWEN_FINAL_BLOCK_ROWS=N` sets a reservation, alongside its existing block
and asset settings. The `xsched:TOKENS:N[:R]` job runs one prompt under the
original order, tail-large and (R > 0) an R-row final-block reservation, and
requires every exported component to be byte-identical across them.
