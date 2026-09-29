# 01 — Target Platform

## Decision

The first production target is AMD Strix Halo on Linux, compiled specifically for the `gfx1151` GPU target through ROCm/HIP.

Portability is secondary to extracting the best behavior from this memory and execution architecture.

## Platform assumptions

The design assumes:

- CPU and GPU share the same physical DRAM pool;
- GPU-visible memory is limited by mapping policy as well as physical RAM;
- host activity, page cache, GPU allocations, registered memory and file-backed mappings all compete for the same physical capacity;
- CPU-to-GPU copies are not automatically useful merely because conventional discrete-GPU runtimes use them;
- low concurrency and decode-heavy workloads are first-class use cases.

## Required consequences

### Native Linux

The primary runtime MUST be Linux-native.

Subsystems that depend on GPU memory registration, page residency, huge pages, NUMA behavior, kernel parameters, power state, or device telemetry MUST be validated on the real native Linux host.

Virtualized environments MAY be used for host-only development, parsers, tests and cross-compilation, but MUST NOT qualify performance or residency behavior.

### gfx1151 specialization

Kernels MAY specialize for `gfx1151` without introducing abstractions for other GPUs in the hot path.

The engine SHOULD prefer compile-time specialization over runtime generality when the shape set is finite and known.

GPU architecture identity MUST be part of every tuning-plan and cache fingerprint.

### Wave size

Wave32 and wave64 MUST be treated as independent implementation candidates.

No family-wide rule may assume that one wave size is universally better. The winning choice MUST be established per kernel family and shape class.

### CPU participation

The CPU is not merely a launcher.

It MAY perform:

- model-file parsing;
- page preparation and bounded prefetch;
- sparse host-table lookup;
- admission control;
- request scheduling;
- tokenization and template work;
- asynchronous I/O;
- background checkpoint persistence.

CPU work MUST NOT create an unbounded dependency chain in the decode critical path.

### NPU

The integrated NPU is outside the initial hot path.

It MAY be explored later for auxiliary work only if transfer, synchronization and software-stack overhead are measured end to end.

## Non-goals

The first architecture MUST NOT be distorted to support:

- arbitrary GPU vendors;
- generic graph execution frameworks;
- arbitrary model architectures;
- large distributed deployments.

Those can be added only after the single-node Strix Halo fast path is mature.
