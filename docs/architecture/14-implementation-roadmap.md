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

### A0.5b — Semantic and numerical oracle

Status: **partially implemented**.

The first dependency-free oracle slice now covers centred/grouped RMSNorm, partial RoPE, sequential recurrent Gated DeltaNet, PLE/n-gram row hashing, deterministic candidate top-k, and fresh-versus-resumed state equivalence.

Still open before the slice is complete: sampled RNG semantics, MTP acceptance, complete QSA selection/tie fixtures, Gated Residual matrix fixtures, tokenizer/template fixtures and first-layer/logit replay from the canonical development checkpoint.

Before optimized model kernels become authoritative, establish independent fixtures or a trusted reference for:

- model-specific operator semantics;
- recurrent update order;
- sparse-selection boundaries and deterministic ties;
- sampling semantics;
- fresh versus resumed state;
- serial row behavior for the first target checkpoint.

Exit gate:

Optimized paths can fail against an independent reference rather than only against another path in the same engine.

### A0.6 — UMA residency lab

Status: **harness implemented; native-hardware qualification pending**.

The standalone lab contains a Linux host path for mmap, prefault, page-fault and access-pattern measurements, plus an optional HIP path for registered mapped memory versus explicit staging.

Compare:

- file mapping;
- registered mapping;
- anonymous/copied buffers;
- optional huge-page buffers;
- random file-backed lookup;
- bounded staging;
- buffer-purpose-specific coherence and cache policies;
- producer/consumer visibility for any transport-visible allocation.

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

Status: **transport core, TCP baseline and direct-stream harness implemented; physical-link qualification pending**.

The core now defines explicit buffer ownership transitions, bounded transfer descriptors with connection epochs, measured link profiles, single-link cost estimation and a dual-link scheduler that only stripes links already qualified as independent. A standalone TCP ping-pong/stream harness provides the network-stack baseline, and a direct-stream harness now exercises configured USB4 character devices through the documented read/write interface while recording configfs tuning state. Real host-to-host qualification, dual-link measurements and higher transport-buffer capability tiers remain pending.

Build the transport abstraction before distributed model execution.

Qualify:

- ordinary network baseline;
- direct USB4 stream path when available;
- one-link latency and bandwidth;
- simultaneous bidirectional traffic;
- dual-link simultaneous traffic;
- directional split;
- flow-level distribution;
- large-message striping;
- logical-stream to physical-link mapping;
- interrupt/event-driven versus dedicated polling progress.

The first version MAY use copies. Mapped or external-DMA modes are later capability tiers.

Exit gate:

The runtime has measured transport capabilities and can schedule independent transfers across one or two links.

## A1 — Single-node execution

### A1.0 — Kernel registry and basic exact kernels

Status: **registry and basic gfx1151 kernel harness implemented; native qualification pending**.

The host-side registry now records operator family, execution phase, exact shape class, wave size, workgroup/resource metadata, numerical contract, graph-safety status, qualification state, compiler identity and measured cost. Selection is deterministic and plan identity excludes benchmark noise.

The standalone HIP lab now contains FP32 RMSNorm, residual add, partial RoPE, deterministic argmax and deterministic small-k top-k. It builds native Wave32 and explicit Wave64 candidates for `gfx1151`, verifies the compiled device `warpSize`, tests true solo-versus-batched row invariance, and emits timing plus function-resource metadata. No variant is target-qualified until it runs on the native Strix Halo host.

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

Status: **packed-projection planner, Q8_0 reference path and IQ4_XS compact candidate implemented; native qualification and canonical format decision pending**.

The core now exposes GGML physical block geometry and a low-row planner for row classes 1/2/4/8. It reports exact packed bytes per output row, total tensor bytes, activation bytes, serial repeated-weight traffic and the ideal lower bound when one packed weight stream is reused across rows.

The standalone HIP lab implements Q8_0 as an auditable packed-format reference. One workgroup owns one output feature and decodes each packed weight once while accumulating all compile-time logical rows. Separate Wave32 and Wave64 builds verify row invariance, report timing/resource metadata and expose how register/LDS cost grows with row count. Q8_0 is not the canonical single-node checkpoint decision.\n\nAn independent IQ4_XS byte-layout/dequant contract is now pinned in the host core and a matching packed HIP candidate applies the same C1/2/4/8 weight-reuse structure to the compact 136-byte/256-element block. IQ4_XS remains experimental until quality and native hardware measurements close the checkpoint decision.

Implement the first low-row-count projection kernel that consumes packed weights directly.

Target order:

1. one logical row;
2. two logical rows;
3. four logical rows;
4. eight logical rows where useful;
5. prefill form separately.

These row classes are not concurrency classes. Qualify the local C1 request path first, then test how C2/C4 scheduling composes multiple request rows with the same kernels or separate shared-round variants.

Compare wave sizes and tiling from end-to-end operator cost.

Exit gate:

Packed decode beats a conservative baseline without model-wide weight expansion.

Qualification MUST include row classes relevant to serial decode and speculative verification, effective weight traffic, dequantization/repack cost, register/LDS pressure and spill checks.

### A1.2 — Recurrent fast path

Status: **recurrent planner and tiled HIP harness implemented; native qualification pending**.

The host planner now models retained-state traffic and explicit rollback strategies: full candidate snapshots, one pre-window snapshot plus replay, and sparse checkpoints with bounded replay. The standalone HIP lab implements the pinned sequential recurrent state update with 128 key lanes and 32-value tiles, retaining each state tile across compile-time 1/2/4/8-token windows before one writeback. It compares that path with repeated one-token launches and checks output/state against a CPU reference.

Implement the recurrent layer contract and optimize state traffic.

Explore:

- register-retained state;
- LDS;
- DPP;
- tiled multi-update execution;
- minimal rollback state.

Exit gate:

The recurrent operator is correct, replayable and has an identified `gfx1151` fast path.

Full snapshots, pre-window snapshot plus replay, and sparse-checkpoint replay have been compared by complete-cycle time, retained bytes and state-write traffic.

### A1.3 — MoE routing and expert execution

Status: **route planner, routing baseline and expert-execution correctness harness implemented; packed grouped reuse pending**.

The host planner now measures touched-expert union, assignments per expert, adjacent-row route overlap, per-projection storage compatibility, realizable 8/4/2/1 expert batches, and packed expert-weight traffic saved by grouping. It also emits a deterministic CSR-style expert grouping plan with expert offsets plus original row/top-k-slot identity for every assignment. The semantic oracle covers FP32 softmax, no-tie top-k and selected-weight renormalization. A separate FP32 expert-execution harness compares direct assignment order with deterministic expert-major grouping and validates both against a CPU reference. The packed IQ4_XS MoE harness then exercises real shared decode of gate/up and down weights across 1/2/4/8-row expert chunks on synthetic compatible geometry; it is a reuse experiment, not yet the canonical expert-format decision. A standalone HIP baseline routes explicit 1/2/4/8-row classes and reports route latency, index agreement, weight error, expert union and resource metadata. Cutoff-tie semantics remain intentionally unresolved until a canonical contract is fixed.

Implement and compare:

- device-resident deterministic routing;
- a direct tiny-row expert path;
- grouped expert execution for reusable routes;
- row restoration without host synchronization;
- routing histograms and expert-union metrics.

Exit gate:

The engine selects expert execution from measured route distributions and complete operator cost rather than row count alone.

### A1.4 — Sparse lookup path

Status: **lookup planner and host benchmark harness implemented; native qualification pending**.

The host planner now models requested rows, unique rows, contiguous I/O runs, duplicate elimination and bounded LRU cache behavior. A standalone Linux benchmark compares mmap, one-pread-per-row, bounded row cache and per-window coalesced pread on identical deterministic access traces. The primary synthetic PLE pattern uses 16 row requests per token window and a 5120-byte initial row model derived from the pinned 2560-wide embedding geometry.

Implement large lookup-table access as a separate subsystem.

Compare:

- direct mapped access;
- CPU gather;
- asynchronous read/staging;
- bounded row cache;
- access-hint policies.

Exit gate:

The lookup path has predictable memory use and does not destabilize long-context admission.

### A1.5 — Correct serial executor

Status: **complete experimental C1/BF16 HIP serial path assembled; native GGUF token/logit replay pending**.

The HIP launcher catalog now covers every serial DAG step with zero missing launcher entries in the structural audit. A raw-token runner assembles the full path from GGUF manifest/tensor access through selective model residency, scratch/state allocation, fresh-state initialization, PLE file lookup, eager dispatch and FP32 LM-head logits. The path remains experimental until it compiles/runs on native Strix Halo and reproduces canonical logits.

The host-side plan now encodes the pinned layer order as a dependency DAG rather than a generic dynamic operator list. A canonical tensor-requirement generator now maps the Qwen4Exp GGUF tensor families into stable per-step binding slots, including hyper-connections, PLE, recurrent/indexed attention, routed/shared experts and the output head. The resulting tensor-access plan identity already contributes to runtime compatibility. The compiled plan now freezes DAG topology, kernel bindings, versioned buffer IO, liveness-derived arena offsets and a stable execution identity. The eager runtime performs complete preflight before dispatch, exposes exact buffer generations and physical arena layout to the backend, and tracks persistent-state generations. A C1 serial-session wrapper carries token id/position and commits metadata only after the whole DAG succeeds; mid-dispatch failure poisons the session because persistent GPU state may have been modified in place. Runtime resources are split into three independently versioned bindings: liveness-aliased scratch arena, persistent session state, and model tensors. The model manifest has a deterministic layout identity separate from the loader-supplied content identity. Runtime compatibility identity excludes physical addresses so snapshots can survive relocation, while binding identity includes addresses/allocation generations for graph and pointer-stability checks. It represents token embedding, position preparation, hyper expansion, the single PLE insertion, attention hyper-mix, recurrent or indexed attention by exact layer pattern, attention reinjection, MLP hyper-mix, MoE routing, routed and shared expert branches, MoE combine, MLP reinjection, final hyper mixing and LM head. Persistent-state read/write masks are explicit for KV, recurrent matrix, convolution history, sparse index, n-gram history and position history. Route/shared-expert parallelism is represented through dependencies rather than hidden scheduling rules.

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

### A1.6 — Greedy speculative execution

Status: **host-side greedy verify/snapshot/replay executor, compiled backend, bounded draft-proposal seam, draft lifecycle metadata and MTP checkpoint binding implemented; concrete MTP proposal compute and native qualification pending**.

The host-side contract implements contiguous target-argmax matching, the mandatory target bonus token, EOS/max-length cropping, recomputed accepted-match count and exact speculative-tail crop count. Positive-depth cycles capture a pre-window snapshot, execute a compiled multi-row target verify path, retain the verified target state on all-match cycles, and otherwise restore plus replay exactly the accepted candidate prefix through the serial target path. Restore failures leave the session poisoned; successful restore can recover a previously poisoned session. Serial and verify executions are bound to the same model content and persistent target-state allocation, dispatch rejects foreign execution identities, and candidate/target token IDs are range-checked before they can become visible output. The depth controller requires a qualified depth-zero baseline, compares complete cycle cost per expected committed token across qualified positive depths, and applies switch hysteresis so noisy marginal gains do not flap the runtime policy. Sampled speculative mathematics is identified separately, but exact sampled replay remains blocked on RNG identity/consumption. A bounded draft-proposal backend seam now feeds candidates into the unchanged target cycle while receiving committed target session metadata read-only. Depth zero bypasses proposal generation entirely, and over-depth proposal results fail before target verification. The runtime now owns draft lifecycle metadata with explicit synchronization, active, reconciliation-required and poisoned phases; positive-depth proposal/reconciliation failures cannot silently reuse stale state, while depth zero remains independent of drafter health. MTP proposal state is bound to the existing model manifest/content identity through the tensors already classified as `mtp`; compatibility identity is relocation-stable and physical binding identity tracks concrete storage/address generations. Concrete MTP proposal compute/state storage, prompt-lookup implementation and proposal qualification remain pending.

Add:

- draft state;
- verify windows;
- minimal snapshot/recompute;
- exact accepted-state commit;
- HIP graph replay where safe;
- a controller that compares depth zero with qualified positive depths;
- cost accounting for proposal, verification, commit/replay and exposed communication;
- bounded exploration and hysteresis;
- optional cheaper proposal paths that never alter target verification semantics.

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

The evaluation MUST distinguish:

- a capacity-oriented whole-layer/block baseline;
- latency candidates that may use finer tensor/operator partitioning;
- critical-path compute saved;
- uncovered communication;
- synchronization/control cost.

Synchronization count alone MUST NOT reject a candidate.

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

## Integrated optimization sequence

The preferred progression is:

1. semantic oracle and real platform truth;
2. packed low-row projection;
3. recurrent state-traffic optimization;
4. MoE routing/expert-path selection;
5. correct serial executor;
6. bounded graph replay and adaptive speculative execution;
7. prefix/fairness qualification;
8. one-link and dual-link transport qualification;
9. distributed placement search.

Transport discovery and microbenchmarks MAY proceed earlier as isolated host-side work, but distributed execution MUST NOT become a dependency of the single-node fast path.

## Ordering rule

A later slice MAY begin early for isolated experimentation, but it MUST NOT become a production dependency before the exit gates of the earlier architectural prerequisites are satisfied.

## Principle

Every slice should either remove an architectural uncertainty or create a measurable fast path.
