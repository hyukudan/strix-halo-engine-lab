# Strix Halo Architecture Guide

This directory is the normative architecture guide for the inference engine.

It intentionally contains no project comparisons, repository names, benchmark anecdotes, reverse-engineering notes, or source-provenance discussion. It describes only the technical decisions that should shape an inference runtime designed specifically for AMD Strix Halo-class systems.

The target is a Linux-native, ROCm/HIP inference engine optimized first for `gfx1151`, unified physical memory, high memory capacity, low-latency local serving, and optional multi-node execution over direct high-speed host links.

## Decision language

The documents use the following terms:

- **MUST**: required unless a later architecture decision explicitly replaces it.
- **SHOULD**: preferred default; deviation requires measurement.
- **MAY**: optional experiment or implementation choice.
- **MUST NOT**: prohibited because it violates correctness, memory safety, reproducibility, or the target architecture.

## Architecture priorities

In order:

1. Correctness and deterministic replay.
2. Predictable unified-memory behavior.
3. Decode latency at concurrency one.
4. Sustainable throughput at small concurrency.
5. Long-context capacity without destabilizing the host.
6. Reproducible kernel selection and tuning.
7. Multi-node scaling without penalizing the single-node fast path.
8. Topology-aware use of one or more direct USB4 links.
9. Operational recoverability.

Peak microbenchmark throughput is never sufficient by itself to justify a design.

## Optimization objective

The engine SHOULD optimize **end-to-end cost per committed token** under the required numerical and memory contract.

Optimization decisions MUST consider the complete path, including packed-weight traffic, persistent-state traffic, launch and synchronization cost, speculative proposal/verification/commit work, expert-routing locality, page faults and staging, host participation, uncovered inter-node communication, and memory retained to enable the fast path.

A wider kernel, larger speculative window, second node or second physical link is useful only when the complete qualified workload improves.

## Documents

- [01 — Target platform](architecture/01-target-platform.md)
- [02 — Memory and residency](architecture/02-memory-and-residency.md)
- [03 — Model loading and weight layout](architecture/03-model-loading-and-weight-layout.md)
- [04 — Execution model and graphs](architecture/04-execution-and-graphs.md)
- [05 — Kernel design](architecture/05-kernel-design.md)
- [06 — Quantization and numerical rules](architecture/06-quantization-and-numerics.md)
- [07 — KV, recurrent state and snapshots](architecture/07-state-kv-and-recurrence.md)
- [08 — Scheduling and concurrency](architecture/08-scheduling-and-concurrency.md)
- [09 — Speculative decoding](architecture/09-speculative-decoding.md)
- [10 — Benchmarking and qualification](architecture/10-benchmarking-and-qualification.md)
- [11 — Runtime observability and recovery](architecture/11-runtime-observability-and-recovery.md)
- [12 — Multi-node and USB4 interconnect](architecture/12-multi-node-and-usb4.md)
- [13 — Platform discovery and capability negotiation](architecture/13-platform-discovery.md)
- [14 — Implementation roadmap](architecture/14-implementation-roadmap.md)
- [15 — Optimization strategy](architecture/15-optimization-strategy.md)
- [16 — Direct GPU transport over USB4](architecture/16-direct-gpu-transport.md)

## Draft investigations

- [Current qualification ledger](qualification-ledger.md).
- [Portable correctness and precision foundations](implementation-2026-10-11.md).
- [Optimization lifetime implementation](implementation-2026-10-12-optimization-lifetimes.md).
- [Pre-hardware experiments](implementation-2026-10-12-prehardware-experiments.md).
- [Reusable execution components](implementation-2026-10-12-execution-components.md).
- [Complete C1 graph and GPU export prerequisites](implementation-2026-10-12-complete-graph-and-export.md).
- [Device-dynamic GDN and indexed attention](implementation-2026-10-12-dynamic-state-operators.md).
- [Experimental tiled integer LM-head consumer](implementation-2026-10-12-integer-runtime.md).
- [Phase specific packed weights and integer execution](proposals/2026-10-11-phase-specific-low-precision.md).
- [Execution optimization gates](proposals/2026-10-12-execution-optimization-gates.md).

The proposal is not native qualification or an adopted numerical mode.

## Change rule

Architecture changes MUST identify:

1. the decision being replaced;
2. the measured problem;
3. the alternative;
4. correctness impact;
5. memory impact;
6. latency and throughput impact;
7. rollback conditions.

A faster result that changes output semantics, hides memory pressure, or cannot be reproduced MUST NOT become the default.
