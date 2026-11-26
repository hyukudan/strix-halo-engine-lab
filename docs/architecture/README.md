# Strix Halo Architecture Guide

This directory is the normative architecture guide for the inference engine.

The architecture prioritizes low-latency local inference, capacity on unified memory, and optional multi-node execution. AMD Strix Halo (`gfx1151`, ROCm/HIP on Linux) is a primary target. The runtime also includes CUDA and other backend bindings, with coverage listed per family in the [support matrix](../support-matrix.md).

Public contracts live in `include/strix/`. Shared implementations in `src/` implement those contracts and backend mechanisms. Family adapters bind them to model data and stages. Executables in `apps/` compose the adapters and shared runtime. The [shared-core map](../engine/shared-cores.md) locates these components. The layering ratchet in `tools/ci/check_layering.py` tracks existing exceptions and prevents their growth.

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

- [Engine reference](../getting-started/engine-reference.md): build options, complete command line, prompt prefill, draft vocabularies,
  environment switches and repository layout (moved from the top-level README).

- [01 — Target platform](01-target-platform.md)
- [02 — Memory and residency](02-memory-and-residency.md)
- [03 — Model loading and weight layout](03-model-loading-and-weight-layout.md)
- [04 — Execution model and graphs](04-execution-and-graphs.md)
- [05 — Kernel design](05-kernel-design.md)
- [06 — Quantization and numerical rules](06-quantization-and-numerics.md)
- [07 — KV, recurrent state and snapshots](07-state-kv-and-recurrence.md)
- [08 — Scheduling and concurrency](08-scheduling-and-concurrency.md)
- [09 — Speculative decoding](09-speculative-decoding.md)
- [10 — Benchmarking and qualification](10-benchmarking-and-qualification.md)
- [11 — Runtime observability and recovery](11-runtime-observability-and-recovery.md)
- [12 — Multi-node and USB4 interconnect](12-multi-node-and-usb4.md)
- [13 — Platform discovery and capability negotiation](13-platform-discovery.md)
- [14 — Implementation roadmap](14-implementation-roadmap.md)
- [15 — Optimization strategy](15-optimization-strategy.md)
- [16 — Direct GPU transport over USB4](16-direct-gpu-transport.md)

## Speculative decode path

- [Multi-row verify window: design and status](../engine/speculative/verify-window-plan.md).
- [MTP command line](../engine/speculative/mtp-native-cli.md) and [MTP runtime contract](../engine/speculative/mtp-native-runtime.md).
- [Decode defaults](../engine/speculative/decode-defaults.md): draft profiles (`--mtp-draft-profile`), min-p by draft identity and the n-gram extension in `auto` (D-055).
- [Known-prefix window](../engine/prefill/known-input-windows.md).
- [Automatic decode optimization profile](../engine/decode/optimization-profile.md).
- [Several requests in one decode loop (cohort windows)](../engine/scheduling/multi-request-cohort.md).
- [Serving over HTTP](../engine/serving/serving.md).

The decisions behind this path (D-021 to D-031, and the measured rejections) are in
[`docs/decisions/decision-register.md`](../decisions/decision-register.md).

## Draft investigations

- [Current qualification ledger](../reference/qualification-ledger.md).
- [Phase specific packed weights and integer execution](../research/proposals/2026-10-11-phase-specific-low-precision.md).
- [Execution optimization gates](../research/proposals/2026-10-12-execution-optimization-gates.md).

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
