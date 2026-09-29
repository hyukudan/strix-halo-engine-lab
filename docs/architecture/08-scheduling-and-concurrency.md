# 08 — Scheduling and Concurrency

## Decision

Concurrency is small, explicit and memory-aware.

The engine is optimized first for C1 and then for a small number of concurrent streams.

Local compute scheduling and inter-node transport scheduling are separate concerns with one coordinated admission policy.

## C1 fast path

Concurrency one MUST be a first-class execution path.

C1 MUST NOT pay avoidable costs for:

- generalized batching metadata;
- large dynamic queues;
- unused synchronization;
- multi-session reductions;
- distributed scheduling when the placement plan is fully local.

## Small-batch execution

C2 and C4 SHOULD be the first shared execution targets.

Higher concurrency MAY be added only when:

- memory admission is reliable;
- per-request latency remains acceptable;
- batching produces real end-to-end throughput gains.

## Compute scheduler

The compute scheduler MUST consider:

- persistent session bytes;
- projected reply growth;
- prefill workspace;
- speculative width;
- active graph shape;
- waiting time;
- placement of the next execution stage.

A request that cannot be safely admitted MUST remain queued or be rejected before partial execution.

## Transport scheduler

When a placement plan contains remote boundaries, a dedicated transport scheduler SHOULD select how each transfer uses the available links.

The transport scheduler SHOULD consider:

- tensor byte size;
- destination node;
- criticality to the token path;
- transfer deadline;
- current in-flight bytes;
- recent latency per link;
- recent bandwidth per link;
- directional congestion;
- whether striping would require reassembly;
- whether another independent tensor can use the second link instead.

The scheduler MAY choose between:

- one-link transfer;
- direction-dedicated link;
- flow-level distribution;
- message striping;
- role-specific link assignment.

Static round-robin SHOULD be a baseline only.

## Prefill versus decode

Long prefill MUST NOT indefinitely stall active decoders.

The engine SHOULD chunk prefill and allow decode opportunities between chunks when doing so improves tail latency without excessive throughput loss.

The policy should be controlled by measured time share, not by arbitrary token counts alone.

In multi-node mode, the scheduler SHOULD also avoid letting large prefill transfers block latency-critical decode transfers when separate link capacity is available.

## Shared rounds

Requests MAY share a decode or verify round when:

- their operator path is compatible;
- state remains logically independent;
- each row reproduces its solo arithmetic contract;
- rollback remains per request.

Shared execution MUST NOT imply shared state.

## Communication/computation overlap

The runtime SHOULD overlap communication and compute when ownership and dependencies are explicit.

Useful mechanisms MAY include:

- double-buffered tensor boundaries;
- pre-posted receive buffers;
- independent GPU streams;
- separate USB4 links for independent transfers;
- computation of the next local stage while an unrelated boundary transfer is in flight.

Overlap MUST NOT change the deterministic execution contract.

## Fairness

FIFO is an acceptable baseline.

The scheduler MAY add bounded prioritization for active decoders over new long prefills to protect interactive latency.

Starvation is forbidden.

## Backpressure

Backpressure MUST occur before unsafe memory or transport commitments.

The public server should expose queueing separately from execution and transport time so latency can be diagnosed correctly.
