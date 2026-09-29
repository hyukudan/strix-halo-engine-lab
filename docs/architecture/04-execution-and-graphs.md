# 04 — Execution Model and Graphs

## Decision

The hot path is a static model executor with phase-specific execution plans.

A generic dynamic operator runtime MUST NOT sit in the critical decode path.

## Execution phases

The executor MUST distinguish at least:

- prefill;
- serial autoregressive decode;
- speculative draft;
- speculative verify;
- accepted-state commit;
- rollback or tail restoration;
- resumed-prefix catch-up.

These phases may use different kernels and workspace plans.

## Static execution

For a qualified model, operator topology SHOULD be fixed after load.

Dynamic decisions in the hot path SHOULD be limited to:

- sequence position;
- active sessions;
- routing results;
- selected speculative depth;
- bounded shape variants;
- kernel variant selected from a precomputed registry.

## Stable buffers

Repeated decode SHOULD use stable-address input, state and output buffers where this improves graph reuse.

Reusing a buffer before a prior launch has finished is forbidden.

Versioned slots or generation counters SHOULD be used wherever asynchronous replay could observe stale input.

A captured graph MUST be bound to an allocation generation for every buffer whose address, size or lifetime affects replay.

Growing, replacing or repacking such a buffer MUST invalidate affected graph instances unless a supported graph-update path has been proven correct for the exact change.


## Buffer lifetime planning

Side values and branch outputs MUST have explicit lifetimes.

The first serial decode plan reserves separate logical classes for:

- two expanded hyper-state buffers used as ping-pong storage across reinjection boundaries;
- hidden-width scratch for attention/MLP intermediates;
- simultaneous routed-expert and shared-expert outputs;
- hyper-connection injection weights;
- route indices and route weights;
- final logits when they are materialized.

A buffer may be physically reused only after every DAG consumer of its current generation has completed.

The initial conservative plan keeps three hidden-width buffers available during the MoE branch so route input, routed output and shared-expert output cannot alias accidentally.

Full-vocabulary logits are accounted separately from internal decode scratch. The peak with materialized logits is the maximum of the internal execution peak and the final-hidden-plus-logits stage, not their sum.

Future LM-head/sampling fusion MAY avoid materializing the full logits buffer, but that optimization must preserve the sampling contract.


## Compiled execution identity

Before a serial plan can execute, the runtime SHOULD compile the following into one immutable execution object:

- static operator DAG;
- selected kernel ids;
- versioned buffer reads and writes;
- liveness-derived arena offsets;
- persistent-state access masks;
- logical row and concurrency shape.

The compiled identity MUST change when any execution-relevant model geometry, kernel identity, buffer layout, numerical-plan input or arena layout changes.

Measured benchmark time MUST NOT contribute to this identity.

Snapshots and future graph instances MAY reference the compiled execution identity rather than reconstructing compatibility from individual flags.

## Buffer generations and arena layout

Physical scratch reuse MUST be derived from logical buffer generations.

Each write creates a new generation. Every read binds to the exact generation it consumes.

The static arena allocator MAY alias two generations only when their live intervals do not overlap.

A dispatcher MUST receive both the logical generation and the resolved arena plan so a reused physical address cannot be mistaken for the previous contents of that slot.

## Eager failure containment

Eager execution performs a complete structural/state preflight before the first dispatch.

Preflight failure leaves the session reusable.

Once dispatch begins, a device kernel may mutate persistent KV, recurrent or sparse state before a later step fails. Unless an explicit rollback snapshot has been restored, such a session MUST be marked poisoned and MUST NOT continue decoding.

Position, committed-token counters and published generation metadata advance only after the complete serial DAG succeeds.


## Runtime resource identities

The execution plan identity and the loaded runtime identity are separate.

A runtime execution binds three resource classes:

- scratch arena;
- persistent per-session state;
- model tensors/weights.

The model manifest layout identity describes tensor structure and placement. It MUST NOT be presented as a content hash.

A separate model-content identity MUST be supplied by the loader after hashing all execution-relevant model files.

Two derived identities serve different purposes:

### Compatibility identity

Includes:

- compiled execution identity;
- manifest layout identity;
- model-content identity;
- session-state layout identity.

It excludes physical addresses.

Snapshots and semantic replay compatibility SHOULD use this identity so relocation after restart does not invalidate otherwise compatible state.

### Binding identity

Extends the compatibility identity with:

- scratch base address and allocation generation;
- host/device session-state bases and allocation generations;
- every local model-tensor address/allocation generation;
- remote tensor ownership descriptors.

Graph captures and any object containing baked pointers MUST use the binding identity.

A physical relocation MUST invalidate binding identity without changing compatibility identity.

## Graph capture

HIP graph capture SHOULD be used for stable decode and verification shapes when it measurably reduces launch overhead.

A kernel or operator MUST declare whether it is graph-safe.

Graph capture MUST be disabled for paths that:

- allocate during capture or replay;
- require host synchronization inside the captured region;
- depend on changing host-side pointer structures;
- perform unsupported dynamic control.

The eager path MUST remain correct and available as a fallback.

Graph capture and explicit graph construction SHOULD be compared when both are available. A captured execution that preserves unnecessary cross-stream ordering or retains excessive buffers MUST NOT be preferred merely because it uses graph replay.

## Shape policy

The engine SHOULD support a finite set of decode/verify shapes rather than unrestricted dynamic shapes.

Typical shape dimensions include:

- active session count;
- verify width;
- token rows;
- expert-row count class;
- attention key-range class.

A shape variant should exist only if traces show it is useful.

## Prefill

Prefill SHOULD prioritize throughput and bounded working memory rather than graph reuse.

A retained prefix or snapshot MUST carry the identity of any prefill arithmetic plan whose chunk decomposition, reduction order or kernel selection can affect state.

The scheduler MAY yield between canonical chunks to protect decode latency, but it MUST NOT silently change the numerical prefill plan merely because another request arrived.

Chunk size MUST be selected with total memory pressure in mind.

A prefill plan SHOULD bound live activations and temporary expert buffers even if a larger monolithic launch is faster in isolation.

## Synchronization

The runtime MUST avoid unnecessary device-wide synchronization.

Dependencies SHOULD use streams and events with explicit ownership.

Every overlap optimization MUST preserve:

- state ordering;
- allocator lifetime;
- replay safety;
- deterministic result semantics.
