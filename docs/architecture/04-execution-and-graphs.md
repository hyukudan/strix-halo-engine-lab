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

## Graph capture

HIP graph capture SHOULD be used for stable decode and verification shapes when it measurably reduces launch overhead.

A kernel or operator MUST declare whether it is graph-safe.

Graph capture MUST be disabled for paths that:

- allocate during capture or replay;
- require host synchronization inside the captured region;
- depend on changing host-side pointer structures;
- perform unsupported dynamic control.

The eager path MUST remain correct and available as a fallback.

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
