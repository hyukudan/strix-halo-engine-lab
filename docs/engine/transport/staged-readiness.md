# Coherent staged readiness

`STRIX_DIST_STAGED_READY_BYTES=N` admits the host path for payloads up to N
bytes, at most 1 MiB. The default is zero. This is an opt-in transport interface;
model producers and consumers require their own qualification before enabling it.

The communicator allocates separate coherent send/receive buffers and cache-line
aligned control records once. Ordinary host exchanges keep their original
buffers. A device-only counter is initialized once and reset by the last block
of each producer/consumer. There is no per-exchange memset, copy, HIP event,
event query or stream synchronization in the readiness methods.

1. `prepare_staged_ready(stream, bytes)` returns a `ProducerStage`, or an empty
   view when the size/route is not admitted. Kernels write the payload directly
   into `data`. The view's `flag` and `token` belong to this generation.
2. All threads of **every** block call
   `publish_staged_block(flag, token, gridDim.x * gridDim.y * gridDim.z)` exactly
   once, after their final payload access. Include padding/empty blocks. The
   helper checks the grid size, releases writes to system scope and publishes
   completion only after the last block arrives.
3. `start_staged_ready(stream, bytes, tag)` posts the progress job. The CPU
   observes the producer's system-release token with a lock-free acquire load,
   then computes the unchanged CRC32C/trailer and runs the host transport.
4. `finish_staged_ready()` validates the unchanged received identity and CRC
   before returning a `ConsumerStage`. Every consumer block calls
   `staged_acquire(flag, token)` **before any early return or payload access**.
   This does one system-acquire load per block and broadcasts it through a
   workgroup barrier.
5. All consumer threads publish at their final access with the same block
   helper. After enqueueing that consumer on the same stream, the caller invokes
   `release_staged_ready(stream)`. RX reuse waits for its consumed token. Merely
   returning from `finish_staged_ready` does not release the buffer.

For a consumer whose only shared-memory obligation is to finish reading RX,
`publish_staged_reads_block(flag, token, total_blocks)` retains the workgroup
barrier and agent-scoped last-block chain, then releases the consumed token to
system scope. It omits the per-thread system fence of the general helper.
Outputs must remain GPU-owned, with subsequent work on the communicator's same
stream. This helper does not publish output stores to CPU or external DMA and
must never replace the producer helper. The general release remains the default.

### Optional attention output epilogue

Both decode forwards accept two independent runtime switches, unset or `0` by
default; exactly `1` enables each switch:

| Switch | Effect on an admitted ready exchange |
| --- | --- |
| `STRIX_DS41_OB_READY_EPILOGUE=1` | The actual Q8 x A8 `o_b` product publishes TX completion from every block, removing the separate publish launch. |
| `STRIX_DS41_READY_READS_ONLY=1` | The real add of the rank halves releases RX with the reads-only helper. Its result stays GPU-owned on the same stream. |

The transport must also admit the view: for R1/R6/N8 host payloads, use
`STRIX_DIST_STAGED_READY_BYTES=262144`. With the default byte limit of zero the
switches have no effect. GPU-direct routes, split `o_b`, and trellis attention
keep their existing paths. CRC32C, trailer identity, acquire and RX/TX lifetimes
are unchanged.

`enqueue_q8_0_a8_ready` uses the selected base Q8/CB4/U5/U6 product or the
production RN form (one output per wave, 16 waves, LDS). It shares the original
product body, including the ascending group accumulation and butterfly. Every
lane reconverges after the body before the system release; padding waves and
partial last blocks participate. There is no per-generation counter clear.
Other RN shapes, native FP8, output A8, accumulated products and more than eight
rows return false without launching; the caller keeps the ordinary product and
separate publish. Switching on the epilogue never silently selects another
product shape. Switching both flags off retains the original behavior.

### Lean releases and the MoE ready exchange

| Switch | Effect |
| --- | --- |
| `STRIX_DS41_READY_LEAN=1` | Every ds41 ready kernel (the three `o_b` epilogues, the separate publish, the add) uses the lean releases below. |
| `STRIX_DS41_MOE_READY=1` | The decode/batch MoE half-sum exchange uses the ready views: the half sum is written into TX, the combine `(shared + low) + high` reads RX, both with the lean releases. No D2H/H2D copies or events. shexp-tp, split-shared and messages above the view limit keep the previous path. |

Both also need `STRIX_DIST_STAGED_READY_BYTES` (262144 covers R1/R6/N8). Why
the general producer release made the fused `o_b` product slower (66 us at R1):
thread 0 of every block loaded `ready` and `block_counter` from the control
line while the progress thread polled `done` on that same 64-byte line, and
each lane's seq_cst `__threadfence_system()` also invalidated GL0/GL1 under
every wave. The lean producer release (`publish_staged_block_lean`) uses a
release-only lane fence, takes the counter from the view
(`ProducerStage::counter`), and lets only the last block read the CPU token,
once, before it publishes `done`. The lean consumer entry
(`staged_acquire_lean`) is a system acquire fence without a control-line load.
The view exists only after `finish_staged_ready()` validated RX and set
`ready`, before the consumer is enqueued. The last block checks the token in
its release. `strix-hip-ds41-ready-bench` times the real products and
consumers in both forms, with and without a CPU thread polling the line.

The GPU-only `strix-hip-ds41-ready-tests` checks coherent TX bytes immediately
after observing the token without a HIP event or stream synchronization: 42
shapes, four generations each, including K-half addressing and partial
blocks. It also checks the production add with both release helpers, 64
asynchronous RX generations each. `strix-hip-staged-ready-tests` exercises both
release helpers over 512 asynchronous generations, padding/empty blocks through
257 blocks, mixed ready/copy lifetimes and injected integrity failures. Run GPU
tests only through the site's GPU reservation mechanism. These fixtures alone
do not establish model exactness or throughput; model logits and state must
also be compared with the switches off before promotion.

Use one stream for the lifetime of a communicator using these views. The stream
must outlive it. Views cannot overlap another prepared/in-flight exchange, and
an unreleased consumer prevents a new view. A consumer may run asynchronously;
its token, rather than a HIP event, protects RX from overwrite. Normal exchanges
and ready exchanges retain independent buffer lifetimes.

The send token is a generation identifier, not payload integrity. CRC32C and all
trailer identity checks remain in place. Missing/wrong tokens fail with a bounded
deadline and poison the communicator. An invalid payload is never exposed as a
consumer view. CPU/GPU native host atomics are required. Teardown synchronizes
the view's stream before freeing memory, including prepared/abandoned views;
that synchronization is outside the exchange hot path.

The existing GPU-event trace mode makes this interface unavailable so diagnostic
event insertion cannot silently become part of the event-free contract. The
optional monotonic CPU phase trace remains available. Probe output distinguishes
GPU-visible availability from total time including its standalone qualification
consumer. Neither alone qualifies model throughput or a production default.

For isolated idle-latency diagnosis, `STRIX_DIST_PROGRESS_SPIN_US=0..10000`
overrides the idle spin window of the communicator's progress thread only.
Pin it explicitly with `STRIX_DIST_PROGRESS_CPU`. Per-stream workers still use
`STRIX_DIST_SPIN_US`; no CPU QoS or idle-state setting changes. Unset preserves
the previous behavior. A longer spin consumes CPU/package power and needs
separate model qualification before adoption.
