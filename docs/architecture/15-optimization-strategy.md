# 15 — Optimization Strategy

## Decision

The engine is optimized by reducing **end-to-end cost per committed token** on the Strix Halo execution and memory architecture.

No optimization mechanism is assumed beneficial merely because it reduces kernel count, increases theoretical FLOPS, widens a speculative window, adds a second node, or aggregates physical links.

The qualified plan is the fastest complete path that preserves its declared numerical contract, memory safety and recovery behavior.

## Primary cost model

For interactive decode, the runtime SHOULD reason about a cost equivalent to:

```text
committed_token_cost =
    compute
  + weight_and_state_memory_traffic
  + launch_and_synchronization
  + proposal_and_verification
  + commit_or_replay
  + uncovered_transport
  + required_host_work
```

Overlapped work SHOULD be charged only where it remains exposed on the critical path.

Memory retained to enable a fast path is a first-class cost because it reduces context and concurrency capacity.

## Local optimization hierarchy

The first performance work SHOULD proceed in this order:

1. establish model-semantic and numerical references;
2. qualify physical memory and GPU mapping behavior on the real host;
3. implement packed one-row decode;
4. add low-row variants that reuse weight traffic for verification;
5. minimize recurrent-state traffic while preserving update order;
6. keep MoE routing on device and choose direct versus grouped expert execution from measured route distributions;
7. integrate a correct serial executor;
8. add graph replay only for stable shapes and stable buffer generations;
9. add adaptive speculation that can choose depth zero;
10. qualify prefix reuse, long-context behavior and mixed prefill/decode scheduling.

A later step MAY be prototyped earlier, but it MUST NOT become a production dependency before the correctness and measurement prerequisites are satisfied.

## Low-row verification

Speculative verification creates an opportunity to amortize packed-weight traffic across multiple logical rows.

The engine SHOULD:

- retain a first-class single-row path;
- add only useful low-row shape classes observed in traces;
- share weight loads and dequantization without silently changing row arithmetic;
- compare vector and matrix-instruction implementations;
- keep prefill on a separate plan.

The winning verify width is a runtime cost decision, not a fixed architecture constant.

## Recurrent-state traffic

For large recurrent state, writing a complete post-update state for every speculative candidate can dominate the value of keeping arithmetic on chip.

The engine SHOULD compare:

- full candidate snapshots;
- one pre-window snapshot plus accepted-prefix replay;
- sparse checkpoints with bounded replay.

The decision MUST include state reads, state writes, retained bytes, replay work and rejection position.

Sequential recurrence order remains canonical unless a different algebraic schedule receives its own numerical qualification.

## MoE-aware verification

Verification cost depends on the union and distribution of routed experts, not only on row count.

The runtime SHOULD measure:

- distinct experts touched;
- assignments per expert;
- adjacent-row route overlap;
- grouping/padding overhead;
- effective expert-weight traffic.

A direct small-row path and a grouped path SHOULD coexist until measurements demonstrate that one dominates the relevant workload.

## Adaptive speculation

The speculative controller MUST compare serial execution with all qualified positive draft depths.

A candidate depth SHOULD be chosen from expected committed tokens divided by the measured cost of proposal, verification, commit/replay, uncovered communication and control.

Acceptance rate alone MUST NOT drive the decision.

Proposal mechanisms MAY be cheaper than the target model, but they MUST NOT alter the target model's logits, vocabulary or sampling distribution.

## Graphs and retained state

Graph replay is a launch optimization, not a semantic layer.

Graph instances MUST be tied to stable allocation generations and invalidated or safely updated when referenced buffers change.

Prefix snapshots MUST identify all arithmetic plans required to reproduce their state, including prefill decomposition where it affects results.

Fair scheduling MAY pause between canonical chunks but MUST NOT silently rewrite a cached conversation's numerical plan.

## Unified-memory specialization

The engine SHOULD optimize allocations by purpose rather than seeking one universal zero-copy mode.

Hot packed weights, persistent state, sparse file-backed tables, transport staging and control metadata have different caching, reclaimability and coherence requirements.

Pinned or strongly coherent memory MUST be budgeted explicitly. A direct mapping is a candidate optimization only when it improves the complete producer-to-consumer path.

CPU helper activity is part of the shared-memory cost model and MUST be tuned under GPU load.

## Distributed optimization hierarchy

Distributed execution follows local qualification.

The progression SHOULD be:

1. measure one physical direct link;
2. measure full-duplex behavior;
3. prove whether two physical links are independent;
4. map logical streams onto measured physical links;
5. qualify flow-level scheduling before aggressive striping for small messages;
6. establish a whole-layer/block partition as a capacity baseline;
7. evaluate finer tensor/operator partitions when compute savings may exceed communication;
8. enable topology-aware placement only from qualified link profiles.

A second node and a second link are schedulable resources, not multiplicative speedup assumptions.

## Capacity versus latency

Every distributed placement MUST state its objective.

A placement may be valuable because it:

- makes a larger model fit;
- increases context or concurrency capacity;
- improves aggregate throughput;
- reduces single-request latency.

These are separate claims and require separate measurements.

For latency, the planner SHOULD compare:

```text
two_node_critical_path =
    concurrent_compute
  + uncovered_communication
  + synchronization_and_control
```

Additional synchronization does not by itself invalidate a partition when parallel compute or reduced local memory traffic compensates for it.

## Physical-link scheduling

Logical streams sharing one physical USB4 path share its measured capacity.

For large messages, striping SHOULD be selected only when:

```text
dual_setup + reassembly + payload / measured_dual_bandwidth
    <
single_setup + payload / measured_single_bandwidth
```

The dual bandwidth term MUST be measured during simultaneous operation.

Small latency-sensitive tensors SHOULD normally use one link or independent flow scheduling until measurements show that striping wins end to end.

## Qualification rule

A candidate optimization becomes preferred only when:

- its model-fidelity obligations pass;
- its within-plan equivalence obligations pass;
- memory use and recovery behavior remain safe;
- repeated end-to-end measurements beat the relevant baseline for a named objective;
- the benefit survives representative contexts, routing distributions and thermal/cache states.

The engine SHOULD retain a simpler fallback whenever a faster path depends on a capability that can disappear after a driver, kernel, firmware, topology or model change.

## Principle

Optimize the bytes, state transitions and synchronization required for each committed token. Use the architecture aggressively, but promote only behavior that the target hardware has actually proven.
