# 14 — Implementation Roadmap

## Decision

Implementation proceeds in slices that each produce a testable artifact.

The order establishes model structure, platform truth and memory behavior before complex GPU kernels, then establishes a correct single-node executor before distributed model execution.

Multi-node capability is designed in from the beginning but does not block the single-node fast path.

## A0 — Host-side architecture and platform truth

### A0.4 — ModelManifest and placement

Status: **implemented, initial version**.

Current deliverables:

- logical tensor manifest derived from validated model metadata;
- conservative tensor-role classification;
- per-tensor physical bytes and offsets;
- explicit residency policy;
- explicit local-node placement;
- machine-readable inspection.

Next gate:

- fix the canonical development checkpoint;
- complete architecture-specific tensor-name and shape validation;
- extend placement representation only as required by the first distributed partition.

### A0.5 — Platform and topology probe

Status: **implemented for static discovery; target-hardware qualification pending**.

Current deliverables:

- dependency-free Linux host probe;
- physical RAM discovery;
- KFD GPU topology;
- GPU-target derivation;
- compute-unit discovery when exposed;
- ROCm/toolchain discovery;
- USB4 domain and host-router discovery;
- direct-stream device discovery;
- machine-readable profile.

Next gate on native Strix Halo hardware:

- real port-to-router mapping;
- negotiated link rate;
- dual-link independence;
- HIP runtime/device capabilities;
- effective mapping limits;
- graph and library capability inventory.

### A0.6 — UMA residency lab

Compare:

- file mapping;
- registered mapping;
- anonymous/copied buffers;
- optional huge-page buffers;
- random file-backed lookup;
- bounded staging.

Measure:

- startup;
- residency;
- page faults;
- bandwidth;
- latency;
- physical-memory pressure;
- cleanup.

Exit gate:

Default residency policies are selected from real Strix Halo measurements.

### A0.7 — Transport micro-lab

Build the transport abstraction before distributed model execution.

Qualify:

- ordinary network baseline;
- direct USB4 stream path when available;
- one-link latency and bandwidth;
- simultaneous bidirectional traffic;
- dual-link simultaneous traffic;
- directional split;
- flow-level distribution;
- large-message striping.

The first version MAY use copies. Mapped or external-DMA modes are later capability tiers.

Exit gate:

The runtime has measured transport capabilities and can schedule independent transfers across one or two links.

## A1 — Single-node execution

### A1.0 — Kernel registry and basic exact kernels

Implement:

- kernel registry;
- offline plan identity;
- RMSNorm;
- residual operations;
- positional transform;
- deterministic argmax/top-k;
- row-invariance tests;
- eager and graph-safe declarations.

Exit gate:

Basic optimized GPU operations match their references and can be selected reproducibly.

### A1.1 — Packed decode projection

Implement the first low-row-count projection kernel that consumes packed weights directly.

Target order:

1. C1;
2. C2;
3. C4;
4. prefill form separately.

Compare wave sizes and tiling from end-to-end operator cost.

Exit gate:

Packed decode beats a conservative baseline without model-wide weight expansion.

### A1.2 — Recurrent fast path

Implement the recurrent layer contract and optimize state traffic.

Explore:

- register-retained state;
- LDS;
- DPP;
- tiled multi-update execution;
- minimal rollback state.

Exit gate:

The recurrent operator is correct, replayable and has an identified `gfx1151` fast path.

### A1.3 — Sparse lookup path

Implement large lookup-table access as a separate subsystem.

Compare:

- direct mapped access;
- CPU gather;
- asynchronous read/staging;
- bounded row cache;
- access-hint policies.

Exit gate:

The lookup path has predictable memory use and does not destabilize long-context admission.

### A1.4 — Correct serial executor

Integrate:

- model manifest;
- loader;
- residency;
- operator graph;
- persistent state;
- KV;
- recurrent state;
- sampling;
- prefix lifecycle.

Exit gate:

A complete serial request runs through the local engine with reproducible output and full memory accounting.

### A1.5 — Greedy speculative execution

Add:

- draft state;
- verify windows;
- minimal snapshot/recompute;
- exact accepted-state commit;
- HIP graph replay where safe.

Exit gate:

Speculative greedy output matches the serial contract and improves completed-token time on qualified workloads.

## A2 — Multi-node execution

### A2.0 — Distributed placement executor

Activate the placement fields already present in `ModelManifest`.

Implement:

- remote tensor boundary;
- node identity;
- generation/sequence safety;
- control/data-plane separation;
- one-link transport backend;
- deterministic fallback.

Exit gate:

A two-node execution crosses a planned tensor boundary and preserves the qualified numerical contract.

### A2.1 — Two-node partition search

Evaluate partition candidates by:

- memory saved per node;
- bytes transferred per token;
- synchronization count;
- compute balance;
- transport overlap.

Do not select by parameter balance alone.

Exit gate:

At least one useful two-node placement is identified or the evidence shows that the tested partition is not worthwhile.

### A2.2 — Dual-link USB4 scheduler

Enable:

- independent physical links;
- directional assignment;
- flow-level assignment;
- large-message striping;
- live link metrics.

Exit gate:

The scheduler selects the best qualified policy by workload class and falls back cleanly to one link.

### A2.3 — Three-node topology

Only after two-node operation is stable.

Evaluate:

- direct ring;
- direct triangle where physically possible;
- topology-aware placement;
- asymmetric link cost.

Exit gate:

Three-node execution improves a defined capacity or performance target without sacrificing correctness or recoverability.

## Deferred work

The following remain deliberately later unless measurements move them forward:

- aggressive KV quantization;
- complex sparse-attention specialization;
- large-node-count orchestration;
- generic multi-vendor support;
- arbitrary model-family abstraction;
- transport-specific kernel logic.

## Ordering rule

A later slice MAY begin early for isolated experimentation, but it MUST NOT become a production dependency before the exit gates of the earlier architectural prerequisites are satisfied.

## Principle

Every slice should either remove an architectural uncertainty or create a measurable fast path.
