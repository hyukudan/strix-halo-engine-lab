# Strix Halo Architecture Guide

This directory is the normative architecture guide for the inference engine.

It intentionally contains no project comparisons, repository names, benchmark anecdotes, reverse-engineering notes, or source-provenance discussion. It describes only the technical decisions that should shape an inference runtime designed specifically for AMD Strix Halo-class systems.

The target is a Linux-native, ROCm/HIP inference engine optimized first for `gfx1151`, unified physical memory, high memory capacity, low-latency local serving, and an optional multi-node extension over high-speed direct host links.

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
8. Operational recoverability.

Peak microbenchmark throughput is never sufficient by itself to justify a design.

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
