# 08 — Scheduling and Concurrency

## Decision

Concurrency is small, explicit and memory-aware.

The engine is optimized first for C1 and then for a small number of concurrent streams.

## C1 fast path

Concurrency one MUST be a first-class execution path.

C1 MUST NOT pay avoidable costs for:

- generalized batching metadata;
- large dynamic queues;
- unused synchronization;
- multi-session reductions.

## Small-batch execution

C2 and C4 SHOULD be the first shared execution targets.

Higher concurrency MAY be added only when:

- memory admission is reliable;
- per-request latency remains acceptable;
- batching produces real end-to-end throughput gains.

## Scheduler

The scheduler MUST consider:

- persistent session bytes;
- projected reply growth;
- prefill workspace;
- speculative width;
- active graph shape;
- waiting time.

A request that cannot be safely admitted MUST remain queued or be rejected before partial execution.

## Prefill versus decode

Long prefill MUST NOT indefinitely stall active decoders.

The engine SHOULD chunk prefill and allow decode opportunities between chunks when doing so improves tail latency without excessive throughput loss.

The policy should be controlled by measured time share, not by arbitrary token counts alone.

## Shared rounds

Requests MAY share a decode or verify round when:

- their operator path is compatible;
- state remains logically independent;
- each row reproduces its solo arithmetic contract;
- rollback remains per request.

Shared execution MUST NOT imply shared state.

## Fairness

FIFO is an acceptable baseline.

The scheduler MAY add bounded prioritization for active decoders over new long prefills to protect interactive latency.

Starvation is forbidden.

## Backpressure

Backpressure MUST occur before unsafe memory commitments.

The public server should expose queueing separately from execution time so latency can be diagnosed correctly.
