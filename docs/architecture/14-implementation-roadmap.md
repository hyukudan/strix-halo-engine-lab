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


### A0.7b — Direct GPU DMA transport tier

Status: **architecture/core synchronization contract implemented; kernel/runtime ABI qualification pending**.

The pool planner now has a concrete move-only ownership layer: every physical link receives separate TX/RX GPU-domain allocations with independent allocation generations and a physical binding identity. A portable exclusive-lease registry binds that physical pool identity to one link/epoch at a time and rejects duplicate/stale lease use. Portable RX/TX completion bridges now keep DMA receive completion, consumer ownership, repost completion, local send reuse, transport ACK and remote consumer-complete credit as distinct milestones. A repeated re-arm gate requires multiple clean stop/arm/first-completion cycles with exact epoch/sequence/byte matching before the lifecycle can publish completion-rearm qualification. These layers are allocator/backend-agnostic and deliberately do not claim uncached/DMA-BUF semantics; the native allocator/backend must satisfy those properties before direct-DMA qualification.

Implement as an optional tier above the existing copy/direct-stream baselines:

- dedicated GPU-owned TX/RX pools;
- GPU-exported DMA-buffer capability probe;
- stream-side imported-pool backend;
- terminal commit record with epoch/sequence/stripe validation;
- system-scope producer-release / consumer-acquire qualification;
- explicit consumer-complete credit;
- RX repost lifecycle and bounded deferred retirement;
- exclusive per-link lease;
- repeated cleanup/re-arm reliability gate;
- operation-specific adapters with portable fallback.

For two physical links:

- separate pools/epochs/leases per link;
- independent-flow qualification first;
- direction/role specialization second;
- single-message striping only after simultaneous independence is proven.

Exit gate:

A direct GPU path improves completed useful GPU-operation latency and the real distributed model workload while preserving deterministic cleanup, recovery and fallback.

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

Status: **route planner, serial-selection and Wave32 hierarchical HIP routing candidates, expert-execution correctness harness, host-side complete-cost selector and target-dimension Q8_0 direct/grouped packed-reuse candidate implemented; native qualification and production integration pending**.

The host planner now measures touched-expert union, assignments per expert, adjacent-row route overlap, per-projection storage compatibility, realizable 8/4/2/1 expert batches, and packed expert-weight traffic saved by grouping. It also emits a deterministic CSR-style expert grouping plan with expert offsets plus original row/top-k-slot identity for every assignment. The semantic oracle and portable C++ selector cover FP32 softmax, deterministic top-k, selected-weight renormalization and an explicit cutoff-tie contract: equal FP32 softmax probabilities prefer the lower expert index. A Wave32-shaped hierarchical host candidate now splits the 512 experts into sixteen 32-wide groups, selects deterministic local candidates, reduces them globally, and matches the canonical serial selector at target top-k=10 geometry; it also exposes exact candidate-work counts. A matching standalone `gfx1151` HIP candidate maps one 512-thread block to a row, performs sixteen 32-lane local reductions followed by a global Wave32 reduction for each top-k slot, validates the deterministic cutoff-tie contract against the CPU reference, and reports the same work counters for native comparison. A separate FP32 expert-execution harness compares direct assignment order with deterministic expert-major grouping and validates both against a CPU reference. The packed IQ4_XS MoE harness then exercises real shared decode of gate/up and down weights across 1/2/4/8-row expert chunks on synthetic compatible geometry; it is a reuse experiment, not yet the canonical expert-format decision. A separate Q8_0 candidate now uses the real routed-expert dimensions (2560 hidden, 640 intermediate), compares direct one-assignment batches with deterministic expert-major 8/4/2/1 batches, reuses each packed scalar across active rows in both projections, restores original rows exactly and checks both paths against the same CPU packed-weight reference. Its native qualification runner gates C1/C2/C4/C8 on geometry/device identity, numerical error and byte-accounting consistency before reporting direct/grouped latency medians. A standalone HIP baseline routes explicit 1/2/4/8-row classes and reports route latency, index agreement, weight error, expert union and resource metadata. Both routing binaries now use the same versioned deterministic cutoff-tie fixture and emit parity metadata. A dependency-free native qualification runner interleaves serial and hierarchical repetitions for C1/C2/C4/C8, rejects device/fixture/architecture mismatches before comparing timings, and emits median latency, resource metadata and speedup in a machine-readable report.

A target-independent execution-policy selector now consumes an explicitly qualified cost model and the actual route distribution. It compares complete direct and grouped costs using assignment count, realizable expert-batch count, packed-weight bytes traversed and row-restoration cost; equal costs deterministically retain the direct path. This closes the host-side policy contract without inventing target measurements: the coefficients still have to be calibrated on native Strix Halo before production selection can be enabled.

Implement and compare:

- target-calibrated direct/grouped complete-cost coefficients for the host-side selector;
- device-resident deterministic routing;
- native C1/C2/C4/C8 qualification of the target-dimension Q8_0 direct/grouped expert path;
- device-resident grouping and row restoration without host synchronization;
- promotion/integration only after measured direct-versus-grouped complete cost agrees with the runtime selector;
- routing histograms and expert-union metrics.

Exit gate:

The engine selects expert execution from measured route distributions and complete operator cost rather than row count alone.

The MoE execution policy now has a complete-layer cost surface: routed direct/grouped estimates are combined with qualified shared-expert and combine costs, including the DAG's routed/shared parallel branch. This prevents selecting grouped routed execution when it does not improve end-to-end MoE layer latency. A qualified layer-schedule builder turns that decision into explicit routed/shared/combine dependencies with a stable runtime-bound schedule identity. A portable schedule executor now materializes those dependencies through an async-capable backend seam, so qualified routed/shared parallelism can be implemented by native HIP streams without changing the policy contract.

### A1.4 — Sparse lookup path

Status: **lookup planner and host benchmark harness implemented; native qualification pending**.

The host planner now models requested rows, unique rows, contiguous I/O runs, duplicate elimination, bounded I/O chunking/inflight waves and bounded LRU cache behavior. A portable host executor now consumes the bounded plan with capped concurrent reads, deterministic result ordering and fail-closed error propagation. The sparse lab drives that same contract through real Linux pread calls with configurable chunk size and inflight depth. A planned PLE backend now applies tensor absolute offsets, launches bounded I/O asynchronously and restores requested rows into staging order. Serial execution can now begin a token/position-bound lookup immediately after preflight and before the DAG; the PLE launcher consumes that handle later, allowing real overlap with preceding embedding/hyper work while preserving synchronous fallback. Mapped model files expose bounded descriptor reads, and the native serial runner can select this planned pread path with configurable chunk/inflight limits. The runner now reports backend identity and complete per-token timing so a qualification harness can compare mapped versus planned paths on identical checkpoint/replay evidence. A policy-sweep harness can evaluate a declared chunk/inflight grid against one mapped baseline and rank only identity-compatible, replay-clean candidates; native execution remains pending. A standalone Linux benchmark compares mmap, one-pread-per-row, bounded row cache and per-window coalesced pread on identical deterministic access traces. The primary synthetic PLE pattern uses 16 row requests per token window and a 5120-byte initial row model derived from the pinned 2560-wide embedding geometry.

Implement large lookup-table access as a separate subsystem.

Compare:

- direct mapped access;
- CPU gather;
- deterministic bounded/coalesced I/O planning;
- asynchronous read/staging that executes the bounded plan;
- bounded row cache;
- access-hint policies.

Exit gate:

The lookup path has predictable memory use and does not destabilize long-context admission.

### A1.5 — Correct serial executor

Status: **complete experimental C1/BF16 HIP serial path assembled; native GGUF token/logit replay pending**.

The HIP launcher catalog now covers every serial DAG step with zero missing launcher entries in the structural audit. A raw-token runner assembles the full path from GGUF manifest/tensor access through selective model residency, scratch/state allocation, fresh-state initialization, PLE file lookup, eager dispatch and FP32 LM-head logits. It can also consume a versioned independent replay fixture, reject checkpoint/vocabulary mismatches before execution, and reject non-finite or mis-sized engine logits before exact-argmax and declared logit-probe qualification. Qualification evidence is bound to the exact checkpoint content, compiled execution and runtime-compatibility identities under assessment, so evidence from a foreign runtime plan cannot satisfy the policy. The path remains experimental until it compiles/runs on native Strix Halo and reproduces canonical logits.

The host-side plan now encodes the pinned layer order as a dependency DAG rather than a generic dynamic operator list. A canonical tensor-requirement generator now maps the Qwen4Exp GGUF tensor families into stable per-step binding slots, including hyper-connections, PLE, recurrent/indexed attention, routed/shared experts and the output head. The resulting tensor-access plan identity already contributes to runtime compatibility. The compiled plan now freezes DAG topology, kernel bindings, versioned buffer IO, liveness-derived arena offsets and a stable execution identity. The eager runtime performs complete preflight before dispatch, exposes exact buffer generations and physical arena layout to the backend, and tracks persistent-state generations. A C1 serial-session wrapper carries token id/position and commits metadata only after the whole DAG succeeds; mid-dispatch failure poisons the session because persistent GPU state may have been modified in place. A portable complete-request wrapper now drives prefix prefill followed by deterministic greedy generation, with lower-token-index argmax ties, EOS, token-budget and context-capacity stop rules. The native HIP runner exposes this path through `--generate N` while keeping replay qualification as a separate mode. Prefix lifecycle now has an explicit relocation-stable metadata contract: an exact prefix token sequence is bound to one validated snapshot/runtime identity, and restore planning is allowed only when execution, runtime and state-layout identities still match. Owned prefix-cache entries plus the HIP snapshot-transfer backend provide the physical capture/restore path, and the native runner can perform a destructive continuation/restore/replay round-trip that only passes when the restored continuation is token-equivalent. High-level prefix capture/restore operations enforce that contract before any physical snapshot transfer and republish session metadata only after a successful restore. A move-only prefix cache entry now owns the validated prefix metadata and compact snapshot allocation as one lifetime-bound object, so cache reuse cannot outlive or detach from its snapshot payload. A HIP transfer backend supplies the concrete device-to-device/host-to-host snapshot copies required to use the same cache object with native GPU session state. Runtime resources are split into three independently versioned bindings: liveness-aliased scratch arena, persistent session state, and model tensors. The model manifest has a deterministic layout identity separate from the loader-supplied content identity. Runtime compatibility identity excludes physical addresses so snapshots can survive relocation, while binding identity includes addresses/allocation generations for graph and pointer-stability checks. It represents token embedding, position preparation, hyper expansion, the single PLE insertion, attention hyper-mix, recurrent or indexed attention by exact layer pattern, attention reinjection, MLP hyper-mix, MoE routing, routed and shared expert branches, MoE combine, MLP reinjection, final hyper mixing and LM head. Persistent-state read/write masks are explicit for KV, recurrent matrix, convolution history, sparse index, n-gram history and position history. Route/shared-expert parallelism is represented through dependencies rather than hidden scheduling rules.

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

A proposal-identity-bound device-memory owner now provides concrete storage for the wide MTP next-step draft state while keeping its element type an explicit planning parameter; this does not open the unresolved forward-semantics gate. The source-level NextN ordering is now pinned separately in a versioned semantic contract and hashed into the compiled plan identity: hnorm per HC stream, repeated enorm(token), [embedding, hidden] concat through eh_proj, attention+combine, FFN+combine, next-state publication, no-injection head mix, then LM head. Replay remains the numerical execution gate. The input-fusion reference can now execute the checkpoint-native packed eh_proj tensor directly, preserving the exact [embedding, hidden] concat layout instead of relying only on an equivalent split-matrix formulation. The host-side contract implements contiguous target-argmax matching, the mandatory target bonus token, EOS/max-length cropping, recomputed accepted-match count and exact speculative-tail crop count. Positive-depth cycles capture a pre-window snapshot, execute a compiled multi-row target verify path, retain the verified target state on all-match cycles, and otherwise restore plus replay exactly the accepted candidate prefix through the serial target path. Restore failures leave the session poisoned; successful restore can recover a previously poisoned session. Serial and verify executions are bound to the same model content and persistent target-state allocation, dispatch rejects foreign execution identities, and candidate/target token IDs are range-checked before they can become visible output. The depth controller requires a qualified depth-zero baseline, compares complete cycle cost per expected committed token across qualified positive depths, and applies switch hysteresis so noisy marginal gains do not flap the runtime policy. Sampled speculative mathematics is identified separately; RNG identity/consumption and deterministic distribution qualification are pinned. A host-side sampled executor now applies the sampled-cycle plan to real session metadata/snapshot semantics with exact rollback and replay. A compiled sampled backend joins the already-materialized serial base distribution with D verify-row distributions, normalizes FP32 logits with stable softmax, and preserves the same persistent target allocation across serial/verify resources; a HIP logits reader supplies concrete device-to-host materialization. Native end-to-end sampled qualification remains pending. A bounded draft-proposal backend seam now feeds candidates into the unchanged target cycle while receiving committed target session metadata read-only. Depth zero bypasses proposal generation entirely, and over-depth proposal results fail before target verification. The runtime now owns draft lifecycle metadata with explicit synchronization, active, reconciliation-required and poisoned phases; positive-depth proposal/reconciliation failures cannot silently reuse stale state, while depth zero remains independent of drafter health. MTP proposal state is bound to the existing model manifest/content identity through the tensors already classified as `mtp`; compatibility identity is relocation-stable and physical binding identity tracks concrete storage/address generations. A deterministic bounded prompt-lookup drafter is now available as a concrete cheap proposal path; it uses the most recent non-overlapping suffix n-gram match and maintains explicitly bounded history. The wrapper preflights the target binding/session before invoking any drafter and clamps proposal depth to remaining target context and `token_budget - 1`. The controller now accepts multiple qualified proposal strategies at the same candidate depth by keying samples on `(proposal_identity, depth)`; hysteresis therefore applies to the concrete drafter/depth choice rather than depth alone. Repeated measurements can now be reduced into controller samples by a campaign-bound aggregator that rejects foreign campaigns, requires a minimum repetition count and disqualifies any proposal/depth group containing a correctness failure. The pinned official config is now represented by a non-executable MTP structural contract (one full-attention predictor layer, shared embeddings, no explicit hidden-source layer and planning-only KV accounting). An offline `strix-mtp-manifest-audit` checks the sidecar's combiner, hyper-connection, attention and MoE tensor families while preserving unknown tensors. MTP execution remains hard-gated while `forward_semantics_resolved=false`. Input-fusion semantics and next-step wide draft-state propagation are independently pinned and marked resolved. A composed forward contract now verifies that input, attention, wide draft state, output mixer/head binding and tensor-family geometry agree end-to-end; this closes the structural composition gap without pretending the numerical forward has been independently replay-qualified. A qualified forward runtime now dispatches the frozen 11-step predictor sequence through an explicit backend seam only after replay qualification and exact proposal/binding/state-memory preflight, so native kernels can be integrated step-by-step without weakening the gate. The next-step state is the post-block `hc_count * hidden_size` residual before final head collapse. Memory admission now accounts one such state row per session with an explicit configurable element-size assumption, avoiding an implicit dtype claim. Attention/QSA and output semantics remain separate open gates. A runtime adapter now connects replay-qualified MTP forward execution to the speculative proposal backend through explicit per-candidate cycle-input and token-reader seams; concrete native step kernels/numerics and measured proposal qualification remain pending.

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
- independent forward-replay qualification for any MTP plan before it can become executable;
- qualification evidence MUST match proposal/plan identity and compare multi-step tokens plus hidden/logit error against a declared policy;

Exit gate:

Speculative greedy output matches the serial contract and improves completed-token time on qualified workloads.

## A2 — Multi-node execution

### A2.0 — Distributed placement executor

Status: **host placement/boundary and control/session guards implemented; remote execution backend pending**.

The placement planner consumes the compiled serial DAG, exact tensor-access plan and manifest placement. Each step receives a deterministic owner node; tensor-free continuation inherits a single dependency owner where possible, mixed-node tensor requirements fail closed, and each cross-node dependency edge becomes a stable boundary with exact activation bytes derived from the compiled arena. Boundary descriptors bind placement-plan identity, request/generation/sequence and dtype. Runtime selection uses distributed execution only when every required remote node is available; local fallback requires an explicit local model copy.

Control-plane reconciliation additionally requires matching protocol, expected peer node, placement plan, runtime compatibility and model content. Receive-side request state enforces the exact request generation and monotonically increasing sequence, poisoning a request on stale or foreign boundary delivery.
A host data-plane package now binds each planned boundary to its stable distributed flow, request descriptor and live dual-link transfer plan; receive-side validation requires the matching inbound flow package before request sequence state advances.
Remote-step execution now has a concrete sequenced controller above an abstract control-wire backend; command/ACK identity is bound to placement/request/generation and any stale, foreign or negative acknowledgement poisons the controller before later remote work can proceed.
The control-wire format is now explicit and bounded: little-endian framed messages carry remote-step command/ACK and boundary-transfer metadata, including link-plan slices, over an abstract byte stream so TCP or native control channels can be substituted without changing control semantics.
A concrete POSIX TCP stream now implements that byte-stream with bounded length-prefix framing and RAII socket ownership, providing an immediately usable two-host control path while leaving future native/direct control channels interchangeable.
A server-side frame pump completes the portable control path: it dispatches validated command and boundary frames, mirrors command identity into ACKs and preserves explicit failure status rather than letting receiver-side execution errors masquerade as success.
The default receiver handler keeps the placement identity global while taking an explicit physical receiver-node id; it validates command ownership/sequence and queues only boundaries addressed to that node, avoiding per-peer re-planning that would create divergent plan identities. A portable end-to-end control test now runs the sequenced client through the actual framed socket stream, server pump and receiver executor, including boundary metadata on the same session.

Implement next:

- remote execution backend for planned boundaries;
- one-link data-plane integration with the native direct-transport backend;
- native two-node replay and deterministic fallback qualification.

Exit gate:

A two-node execution crosses a planned tensor boundary and preserves the qualified numerical contract.

### A2.1 — Two-node partition search

Status: **host candidate evaluator implemented; measured candidate generation pending**.

Candidate construction can now be driven directly from an executable placement plan, manifest residency/placement and complete measured step/boundary evidence, eliminating manual cost transcription. The search treats capacity and latency as separate objectives. Per-node memory budgets gate feasibility, while token latency is estimated from critical-path compute plus uncovered communication and control cost. Synchronization count is retained as evidence rather than used as an automatic veto. Every comparison requires at least one whole-layer/block baseline, but operator-level candidates may win the latency objective. Capacity selection minimizes the largest node footprint (then imbalance and token cost); latency selection requires an explicit improvement threshold over the measured local serial baseline.

Next:

- derive candidate costs from A2.0 boundaries and native per-operator timing;
- feed measured one-link/dual-link communication exposure rather than ideal bandwidth;
- evaluate canonical whole-layer and finer operator splits on the two Strix hosts.

Exit gate:

At least one useful two-node placement is identified or the evidence shows that the tested partition is not worthwhile.

The direct-native control path now has a lifetime-bound link session that acquires the exclusive lease, imports exactly the leased pool, owns send/receive completion bridges and guarantees pool release before lease release.

### A2.2 — Dual-link USB4 scheduler

Status: **directional live-metric scheduler with failover/recovery implemented; native telemetry qualification pending**.

A completion telemetry bridge now maps data-plane milestone timestamps onto directional live-link samples for the scheduler; native backends only need to supply monotonic timestamps and queue depth. The runtime scheduler maintains separate outbound/inbound profiles for each physical link, updates rate/latency/queue pressure from live samples, preserves flow affinity with switch hysteresis, and reuses the existing complete-cost transfer planner for large-message striping. Small or latency-critical traffic remains on one link, while bulk messages stripe only when two explicitly independent qualified links beat the best single-link estimate. Physical-link failure invalidates stale affinity and collapses striped traffic to the surviving link; explicit re-qualification plus fresh telemetry returns the recovered link to selection for new flows. Distributed placement boundaries now derive a stable flow id, local direction and deterministic size-class priority directly from the placement-plan/boundary identities, closing the A2-to-scheduler handoff.

Next:

- feed native completion/repost measurements into live profiles;
- qualify 1× versus 2× USB4 on the real two-Strix topology.

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
