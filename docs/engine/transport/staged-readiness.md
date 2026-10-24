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
