# 11 — Runtime Observability and Recovery

## Decision

Operational visibility is part of the engine architecture.

The runtime MUST make memory, execution, transport and recovery state inspectable without attaching a debugger.

## Required telemetry

The engine SHOULD expose:

### Memory
- physical memory total and available;
- configured host reserve;
- mapped GPU memory limit;
- mapped bytes in use;
- registered/pinned bytes;
- file-backed model bytes;
- estimated resident file-backed bytes;
- KV and recurrent-state bytes;
- scratch and reusable-cache bytes;
- transport-buffer bytes.

### Execution
- active requests;
- queued requests;
- phase per request;
- graph versus eager execution;
- active kernel-plan identity;
- active placement-plan identity;
- prompt and decode throughput;
- speculative acceptance;
- prefill chunk progress.

### GPU
- utilization;
- clocks;
- power;
- temperature;
- throttling indicators;
- recoverable runtime errors.

### Transport
For every physical and logical link:

- peer node;
- physical port/router identity;
- negotiated link rate;
- health state;
- bytes sent and received;
- current in-flight bytes;
- completions;
- errors and retries;
- recent payload bandwidth;
- p50/p95/p99 transfer latency;
- interrupt rate;
- active transport mode;
- mapped/copy/staging mode;
- current scheduler role.

When two links connect the same node pair, telemetry MUST keep them independently visible.

## Structured logs

Critical events MUST be structured and include:

- timestamp;
- request/session identity where appropriate;
- model identity;
- phase;
- node identity;
- placement plan;
- relevant link identity;
- memory snapshot;
- error class;
- recovery action.

## Failure containment

A single malformed request MUST NOT corrupt shared engine state.

A failed session SHOULD be isolated and destroyed while other sessions continue when device and transport state remain valid.

A failed secondary transport link SHOULD be isolated without terminating the node when the active placement can continue correctly over another qualified path.

## Dirty termination

The supervisor MUST distinguish:

- clean engine exit;
- process crash;
- device/runtime failure;
- inconsistent mapping state;
- transport disconnect;
- peer disappearance;
- stale generation/completion state.

After an unclean termination, automatic model reload SHOULD require a health check.

If GPU mapping or driver accounting is inconsistent, the system SHOULD refuse repeated reload attempts and escalate to a controlled host restart rather than entering an OOM/crash loop.

If transport generation or ownership state is uncertain, affected in-flight tensors MUST be discarded rather than reused.

## Administrative boundary

The inference process SHOULD have a narrow control interface.

Administrative actions SHOULD be typed operations such as:

- load model;
- unload model;
- drain;
- restart engine;
- connect peer;
- disconnect peer;
- requalify transport;
- apply approved runtime profile;
- run qualification benchmark.

The administrative interface MUST NOT expose arbitrary shell execution.

## Reproducible runtime profiles

A runtime profile SHOULD version:

- GPU memory-mapping policy;
- host reserve;
- cache limits;
- concurrency limits;
- graph policy;
- kernel-plan identity;
- placement-plan identity;
- transport policy;
- dual-link policy;
- power profile;
- supported context policy.

Changes SHOULD be applied atomically and be reversible.

## Principle

A fast engine that cannot explain its current memory state, execution mode, transport state or recovery path is not production-ready.
