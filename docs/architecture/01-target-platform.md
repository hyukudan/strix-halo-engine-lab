# 01 — Target Platform

## Decision

The first production target is AMD Strix Halo on Linux, compiled specifically for the `gfx1151` GPU target through ROCm/HIP.

Portability is secondary to extracting the best behavior from this memory and execution architecture.

The single-node path is the first qualification target, but the architecture MUST preserve a clean path to multi-node execution across direct high-speed host links.

## Platform assumptions

The design assumes:

- CPU and GPU share the same physical DRAM pool;
- GPU-visible memory is limited by mapping policy as well as physical RAM;
- host activity, page cache, GPU allocations, registered memory and file-backed mappings all compete for the same physical capacity;
- CPU-to-GPU copies are not automatically useful merely because conventional discrete-GPU runtimes use them;
- low concurrency and decode-heavy workloads are first-class use cases;
- many Strix Halo systems expose two native USB4 links that can potentially be used independently or together for direct node-to-node transport.

## Required consequences

### Native Linux

The primary runtime MUST be Linux-native.

Subsystems that depend on GPU memory registration, page residency, huge pages, NUMA behavior, kernel parameters, power state, device telemetry, or direct host-to-host transport MUST be validated on the real native Linux host.

Virtualized environments MAY be used for host-only development, parsers, tests and cross-compilation, but MUST NOT qualify performance, residency, or interconnect behavior.

### gfx1151 specialization

Kernels MAY specialize for `gfx1151` without introducing abstractions for other GPUs in the hot path.

The engine SHOULD prefer compile-time specialization over runtime generality when the shape set is finite and known.

GPU architecture identity MUST be part of every tuning-plan and cache fingerprint.

### Target-specific cost model

Performance thresholds, tile sizes, wave choices, speculative widths, CPU helper counts and transport policies MUST be derived from measurements on the target system.

Values learned on another GPU, memory architecture or interconnect MAY be used only as experiment seeds. They MUST NOT become production defaults without qualification on the target.

Generated GPU code SHOULD be inspectable for the selected `gfx1151` plan, including instruction selection, register pressure, LDS use and spills.

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
- inter-node control and transport coordination;
- background checkpoint persistence.

CPU work MUST NOT create an unbounded dependency chain in the decode critical path.

### Multi-node readiness

The core executor MUST NOT depend on multi-node execution, but model partitioning, tensor ownership and scheduler interfaces SHOULD avoid assumptions that make a later two-node fast path invasive.

Distributed execution MUST be an optional layer over the single-node engine.

The initial interconnect target is direct USB4 between Strix Halo hosts. A transport abstraction MUST allow:

- standard network transport as a baseline and control path;
- direct USB4 streaming when supported by the host kernel;
- one or more independent physical links;
- future lower-latency transports without changing model semantics.

### NPU

The integrated NPU is outside the initial hot path.

It MAY be explored later for auxiliary work only if transfer, synchronization and software-stack overhead are measured end to end.

## Non-goals

The first architecture MUST NOT be distorted to support:

- arbitrary GPU vendors;
- generic graph execution frameworks;
- arbitrary model architectures;
- datacenter-scale distributed fabrics.

Small multi-node Strix Halo deployments are explicitly in scope, provided they remain optional and do not slow the single-node path.
