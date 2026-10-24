# CUDA prefill with bounded passes

This is an experimental, opt-in path. Complete-model qualification at a million tokens and production handoff
qualification remain pending. Existing behavior is selected when `pass_rows` is zero (the default).

`CudaPrefillOptions::max_positions` is the global context capacity. `pass_rows` bounds row buffers independently
of that capacity. The applications accept `--context N` (an alias for `--positions N`) and `--pass-rows P`.
Pass starts are aligned to the attention window. A short final pass is rebalanced so the global decoder suffix
fits in it. The experimental pass limit is 128K rows, and chunk rows must be positive and at most 65535. Device
identity is checked before profiling or allocating on the selected device. Graphs, closed batches, and streamed
Engram uploads are not supported with bounded passes.

The encoder processes every pass through its layers. Compressed KV, index keys, compressor carries, and window
state persist at global positions. The decoder dependency cone is planned for the entire request; intermediate
passes only run its KV-source compressor. The decoder outputs and head run in the final pass. Expert uses form
one request schedule, avoiding idle preloads between passes. Two mapped Engram buffers prefetch one pass ahead.
Activations are retained only for one pass; they are not spilled to disk.

The attention driver supplies its own workspace. Decoder selections and candidates have a compact logical row
origin, and fast-index scores use at most 256 simultaneous queries. Selection above 512K entries requires the
explicit `SelectView::long_context` flag; its extended merge supports up to 2M entries. The ordinary merge and
entry limit remain unchanged without that flag.

`run(tokens)` starts a fresh request. `extend(tokens)` continues the previous or imported sequence and requires
bounded passes. `import_state(state)` validates the history and requires known producer selections. The caller
must bind the state to the same effective weights and numerical configuration. The standalone application exposes
`--state-in FILE`, interpreting the supplied prompt as the complete history plus suffix, and `--check-extend N`
for a fresh-versus-checkpoint comparison.

The exact path retains the existing kernels and reduction order. Local checks cover serialized-byte identity
against the original single-pass path at 16K with three layers and at 8K with all 40 layers, including full logits
and per-layer stream hashes. A three-layer checkpoint at 8K also extends to 16K with identical bytes. Direct
selection checks cover 1M entries, ties, candidate masks, and rejection before launch. These checks do not qualify
the fast path, complete-model long-context fidelity, memory admission, or end-to-end serving.

For continuation qualification, `strix-deepseek41-longctx --logits-out FILE` optionally writes complete raw FP32
logit rows in teacher-forced order followed by greedy order. A two-rank run appends `.r0` and `.r1` to the path;
both dumps require gathered logits. Empty or changing vocabulary sizes and write failures fail the run. The option
is off by default and changes no arithmetic. Dump-assisted native continuation checks remain pending.

Device and host memory must be budgeted separately, including the platform's commit charge for device allocations.
The description exposes row capacity, device state, expert tiers, Engram pinned bytes, and total pinned bytes.
Long-context performance depends on expert residency, file traffic, query count, and Engram I/O; short-context
throughput is not a qualification result for this path.

Bounded passes default to a 38 GiB device cap when none is supplied and preserve at least 4 GiB of free host commit.
On Windows, load admission includes the device cap, requested host experts, both Engram buffers, and a conservative
CPU reservation. Request admission and every pass boundary probe free commit again. The reservation includes hash
IDs, state copies, queued requests, and runtime margin (between 8 and 9 GiB at 1M positions). It is an admission
estimate; external allocations can still reduce available commit after a probe. CPU boundary tests cover the floor,
one-byte deficits, and overflow. End-to-end memory qualification is pending.

With `--pass-rows`, the service limits its queue to at most 16 jobs, bounds concurrent connections and JSON body
size, and releases the parsed token array before waiting in the queue. Insufficient commit returns HTTP 503.
`/health` adds `position`, `total`, `elapsed_s`, `eta_s`, `cancel_pending`, and host free commit. ETA is a rough
linear estimate from completed passes; it is null until progress is available. `POST /v1/cancel` with a non-null
`session` removes a matching queued job or requests cancellation before the active job's next pass. Session values
should be unique. A completed prefill proceeds through export and handoff. Interrupted prefills return HTTP 503
and leave the service available for another request; device failures still require a restart. These HTTP paths
require native service qualification before deployment.

`set_state_stream` attaches an optional observer for immutable compressed KV and index-key entries after each pass.
Two pinned 4 MiB buffers are reused only after the observer's completion futures finish. Completed groups retain
their original bytes; mutable rings, carries, selections and tokens are exported at the end. Transfer failures clear
the partial request and are reported separately from device failures. The observer requires bounded passes and
must complete every future it returns. Its device execution still requires native qualification.

The service's experimental `--handoff-v2 --state-identity ID` enables the framed transport with bounded passes.
See [the v2 contract](ds41-state-transfer-v2.md). Its Windows SHA provider is checked against known answers before
load. CPU transport tests preserve original state hashes through complete reconstruction and lost acknowledgements.
These transport checks do not qualify model execution, a decode import, or final handoff latency.
