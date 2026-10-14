# Strix Halo Architecture Guide

This directory contains the normative architecture guide, dated implementation
reports and draft investigations for the inference engine.

The architecture describes the project's own technical decisions for an inference
runtime designed specifically for AMD Strix Halo-class systems. Comparative
evaluations, external benchmark records and source-identified inspection artifacts
remain outside Git. Required attribution, licenses and notices for adopted
components and official references must still be preserved; see the
[publication policy](../AGENTS.md) and [sources ledger](../SOURCES_LEDGER.md).

The target is a Linux-native, ROCm/HIP inference engine optimized first for `gfx1151`, unified physical memory, high memory capacity, low-latency local serving, and optional multi-node execution over direct high-speed host links.

## Start here

| Your task | Reading path |
| --- | --- |
| Build or inspect the host tools | [Repository setup and examples](../README.md#build-and-test-the-host-tools), then the relevant [laboratory](#implementation-and-laboratory-entry-points) |
| Understand what has actually been demonstrated | [Current qualification ledger](qualification-ledger.md), then the dated implementation report for that area |
| Change memory, model or execution behavior | [Memory and residency](architecture/02-memory-and-residency.md), [model loading](architecture/03-model-loading-and-weight-layout.md), [execution and graphs](architecture/04-execution-and-graphs.md) and [numerical rules](architecture/06-quantization-and-numerics.md) |
| Plan native model experiments | [Benchmarking and qualification](architecture/10-benchmarking-and-qualification.md), [hardware campaign preparation](implementation-2026-10-12-hardware-campaign-preparation.md) and [native numerical preparation](implementation-2026-10-12-native-numerical-campaign.md) |
| Work on distributed execution | [Multi-node and USB4](architecture/12-multi-node-and-usb4.md), [platform discovery](architecture/13-platform-discovery.md) and [direct GPU transport](architecture/16-direct-gpu-transport.md) |
| Choose the next implementation slice | [Implementation roadmap](architecture/14-implementation-roadmap.md), [optimization strategy](architecture/15-optimization-strategy.md) and the current ledger |

## How to read status and evidence

The files under `architecture/` state design requirements. A requirement is not a
claim that the corresponding native path is implemented or qualified. Dated
`implementation-*` reports describe particular development slices; later work may
supersede their open-feature lists. Consult the current ledger alongside
[implementation history](../IMPLEMENTATION_STATUS.md) before drawing a current
status conclusion.

Files under `proposals/` remain draft investigations until their stated gates
pass. Portable contract tests establish host-side behavior, offline HIP builds
establish compilation, and native execution evidence establishes only the exact
hardware/runtime/workload bindings tested. Full-model fidelity and complete-workload
performance require their own evidence. These levels must not be substituted for
one another.

In these documents, **C1** means concurrency one, **GGUF** identifies the model
container being inspected, and **RAW** refers to the diagnostic recording path.
Eager floating execution, floating graph replay and eager integer LM-head execution
are separate experimental modes. A result for one does not qualify another.

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

## Implementation reports and draft investigations

- [Current qualification ledger](qualification-ledger.md).
- [Portable correctness and precision foundations](implementation-2026-10-11.md).
- [Optimization lifetime implementation](implementation-2026-10-12-optimization-lifetimes.md).
- [Pre-hardware experiments](implementation-2026-10-12-prehardware-experiments.md).
- [Reusable execution components](implementation-2026-10-12-execution-components.md).
- [Complete C1 graph and GPU export prerequisites](implementation-2026-10-12-complete-graph-and-export.md).
- [Device-dynamic GDN and indexed attention](implementation-2026-10-12-dynamic-state-operators.md).
- [Experimental tiled integer LM-head consumer](implementation-2026-10-12-integer-runtime.md).
- [Hardware campaign preparation](implementation-2026-10-12-hardware-campaign-preparation.md).
- [Native numerical campaign preparation](implementation-2026-10-12-native-numerical-campaign.md).
- [Eager RAW baseline recording](implementation-2026-10-12-raw-baseline-recording.md).
- [Complete-model experimental decode recording](implementation-2026-10-12-decode-execution-modes.md).
- [Phase specific packed weights and integer execution](proposals/2026-10-11-phase-specific-low-precision.md).
- [Execution optimization gates](proposals/2026-10-12-execution-optimization-gates.md).

The proposals are not native qualification or adopted numerical modes.

## Implementation and laboratory entry points

Run build and test commands from the repository root using the
[host setup instructions](../README.md#build-and-test-the-host-tools). The default
build does not enable HIP. Before comparing test runs, inspect their configured
inventory with `ctest --test-dir build -N`; optional dependencies and native
configuration can change which checks are available.

| Experiment | Practical instructions | Related design |
| --- | --- | --- |
| UMA access and residency | [Residency lab](../lab/uma_residency/README.md) | [Memory and residency](architecture/02-memory-and-residency.md) |
| Packed weights and integer arithmetic | [Packed projection](../lab/packed_projection/README.md), [low precision](../lab/low_precision/README.md) and [integer projection](../lab/integer_projection/README.md) | [Kernel design](architecture/05-kernel-design.md) and [numerics](architecture/06-quantization-and-numerics.md) |
| Recurrent state and sparse lookup | [GDN lab](../lab/recurrent_gdn/README.md) and [sparse lookup](../lab/sparse_lookup/README.md) | [State, KV and recurrence](architecture/07-state-kv-and-recurrence.md) |
| Model recordings and host evidence checks | [Numerical campaign runbook](../lab/native_campaign/numerical/README.md) | [RAW baseline recording](implementation-2026-10-12-raw-baseline-recording.md) and [decode modes](implementation-2026-10-12-decode-execution-modes.md) |
| Copied transport and physical-link diagnostics | [Transport lab](../lab/transport/README.md) | [Multi-node and USB4](architecture/12-multi-node-and-usb4.md) |
| Experimental GPU-owned transport | [Commissioning runbook](../lab/direct_gpu_stream/COMMISSIONING.md) and [private kernel builder](../lab/direct_gpu_stream/kernel/README.md) | [Direct GPU transport](architecture/16-direct-gpu-transport.md) |

Use each runbook's prerequisites and rejection criteria before attempting native
work. Kernel/module preparation, commissioning and model recording are separate
operator workflows; the host test suite does not perform or qualify them. Keep
models, target inventories, detailed receipts and deployment results in private
storage as required by the publication policy.

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
