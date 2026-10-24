# Strix Halo optimization review — 2026-09-29

**Status:** Research and proposed decision amendments. This document does not change the normative architecture guide or qualify an implementation.

**Target:** Native Linux, AMD Strix Halo / `gfx1151`, first one host and then two directly connected hosts. No additional vendor backend is proposed.

## 1. Scope and evidence boundary

The lab was inspected at `94d4333b7607343eabf64124ad0a1fbfd15b1861`. external execution study was inspected at `191188075bca56a7c71074a79375eb4c1cb22e1c`, release 0.3.6.3 dated 2026-09-29. Selected implementation files, the relevant model recipes, and the lab architecture documents were read. This is not an exhaustive audit of either repository.

The Linux streaming implementation was inspected at `72d3fcf802c45d00b300f25b848a93c3a2bd7c7e`, `drivers/thunderbolt/stream.c`, blob `25c259dd07602dfadab4a0f1449411d08a65ee3d`.

There were no native Strix Halo measurements in this review. GPU kernels, inference quality, USB4 throughput, transport recovery, and build/tests were not executed. Arithmetic examples below are calculations from declared shapes or explicitly stated assumptions, not benchmark results. External performance reports are evidence of promising mechanisms, not an AMD speedup forecast.

The lab currently contains a portable host-side foundation: defensive GGUF inspection, memory planning, a conservative model manifest, and static platform discovery. Its implementation status explicitly leaves HIP residency, `gfx1151` kernels, a complete executor, and several model-semantic contracts unimplemented or unqualified. Planning a fast path is not implementing it. [L1]

Research comparisons and provenance remain here, outside `docs/`. Any eventual normative amendment should describe technology and requirements in English without project comparisons or external repository names. [L2]

## 2. Executive decision

Preserve the current architecture. Its strongest decisions are already appropriate: explicit physical-memory accounting, packed weights, phase-specific execution, row invariance, bounded prefix state, minimal recurrent rollback, C1 priority, and measured rather than assumed USB4 independence.

The highest-value next step is not translating another backend. It is implementing the same *mechanisms* with a `gfx1151`-specific cost model:

| Mechanism | Native candidate | Main qualification question |
| --- | --- | --- |
| Low-row target verification | Packed HIP projections with fixed per-row reductions and shared weight tiles | Does a wider verify window reduce committed-token time without changing the serial contract? |
| Recurrent multi-update execution | Register/LDS state tiles, sequential updates, minimal snapshots plus accepted-prefix replay | Are saved state writes worth replay and register pressure? |
| MoE routing and grouping | Device-resident routing; a direct tiny-row path and a grouped larger-row path | Does grouping actually reuse experts, or mostly add sorting and padding? |
| Fused compatible projections | Lossless packed layout planning and phase-specific fusion | Is the whole operator chain faster after accounting for repack and workspace? |
| Adaptive drafting | Measured committed-token cost including a depth-zero alternative | Does speculation beat serial in the current context, route distribution and topology? |
| Prefix reuse and prefill scheduling | Canonical arithmetic plans, bounded caches, resumable work | Can service latency improve without cache-dependent numerical changes? |
| Graph replay | Finite HIP graph shapes, device routing, versioned buffers | Does replay remove launch cost without stale pointers or hidden memory growth? |
| UMA and sparse lookup | Separate hot compute buffers from bounded reclaimable lookup/staging pools | Which access/coherence policy wins on the actual APU? |
| Two-host execution | Explicit layer/state ownership and measured physical-link scheduling | Does added capacity or compute exceed communication and synchronization cost? |

These are hypotheses to test, not unconditional claims that a particular implementation wins.

## 3. What the inspected implementation actually contributes

### 3.1 Exactness is an execution contract, not a cross-platform claim

The inspected sampler keys random draws by seed, absolute position and token ID, resolves ties deterministically, and implements row-wise selection so speculative verification can reproduce the corresponding serial draw. Its narrow-nucleus path includes a fallback when its candidate set cannot safely determine the answer; a fixed truncated vocabulary is not a general exact target sampler. [T1]

The native design should separate three obligations:

1. **Model fidelity:** an independent, trusted implementation checks the mathematical model and reference numerical behavior.
2. **Execution equivalence:** optimized/serial, speculative/serial, solo/shared, and resumed/fresh agree within a fixed numerical plan where exactness is promised.
3. **Alternative numerical modes:** different quantization, accumulation, or distributed reduction plans receive separate identities and quality validation.

Internal equivalence alone can preserve the same bug in every path. Conversely, requiring AMD to reproduce another architecture's floating-point bits is not necessary to reproduce the intended model faithfully. Existing documents 06 and 10 should make these separate test obligations explicit, rather than collapsing them into one reference check. [L3]

The logical sampling key must not accidentally depend on a worker slot, transient HTTP request identifier, rejected work, transport epoch or which node executes the sampler. An explicit user seed and a well-defined logical request stream should reproduce the same run under the same numerical plan.

### 3.2 Low-row projection is the bridge between serial decoding and useful verification

The inspected projection dispatcher distinguishes exact low-row paths and checks or constrains shapes rather than assuming arbitrary matrix multiplication is row-invariant. Compatible projections sharing an input are stacked, including GDN projections and attention/indexer projections. The retained reference weights and additional execution layouts also demonstrate why repack memory must be counted explicitly. [T2]

Native candidate:

- Start with a conservative packed vector path for one row.
- Add two-, four-, and eight-row tiles only when traces justify them.
- Share packed weight loads and dequantization across rows without changing each row's reduction order.
- Keep source format geometry explicit: bit width alone does not identify block scales, offsets, signedness, packing or group size.
- Build a separate prefill path, rather than treating large prefill as repeated vector decoding.
- Compare a vector/VALU path with native WMMA candidates, including layout transformation, masked rows and accumulator behavior.

AMD documents RDNA WMMA through HIP compiler intrinsics, including FP16/BF16 inputs, FP32 accumulators, and wave32/wave64 forms. LLVM identifies `gfx1151` separately; targeting generic AMD GPU code is not equivalent to qualifying this target. Register mappings, instruction selection and resource use must be checked in the generated `gfx1151` code. [A1][A2]

Integer 4-bit matrix support does not mean an arbitrary block-quantized weight format with floating-point activations can be executed as an integer matrix product without an additional numerical transformation. Do not silently quantize activations to use a faster instruction. Dequantization into bounded register/LDS tiles is a separate candidate from full-model expansion.

Qualification should record total operator latency, effective weight bytes, dequantization cost, VGPRs, LDS, spills and code identity. No fixed wave size or occupancy target should replace this comparison.

### 3.3 Recurrent kernels: retain the idea, reconsider the write traffic

The inspected Flash Next `gdn_step` kernel loads recurrent state into thread-local storage, performs candidate-row updates in order, and writes a complete post-update state for each row. The pipelined variant separates preparation, state updates and output work, while still writing per-row states. This observation applies to that inspected path, not necessarily every recurrent family/backend. [T3]

The lab's existing preference for minimal snapshots and accepted-prefix replay is therefore not merely copying this implementation. It is a potentially better tradeoff for the target, but must win in measurements. [L4]

From the checked-in development config:

```text
36 recurrent layers
48 value heads per recurrent layer
128 key dimensions × 128 value dimensions
4 bytes per FP32 state element

36 × 48 × 128 × 128 × 4 = 113,246,208 bytes = 108 MiB/session
```

A strategy retaining one complete recurrent state for each of eight verified rows would materialize **864 MiB per session per window** in those snapshots alone. That is not the complete session footprint and excludes convolution state, KV, scratch, the initial state, and allocation rounding. [L5]

Compare three native policies:

- Full per-row snapshots as a simple correctness baseline.
- One pre-window snapshot plus saved update inputs and accepted-prefix replay.
- Sparse checkpoints inside the window plus shorter replay.

Measure both retained capacity and bytes written per cycle. Keeping a state in registers does not eliminate the writes used to retain all candidate outcomes. Minimal replay is not free either: it rereads the initial state and update inputs and repeats arithmetic. The decision is total cycle cost, not snapshot bytes alone.

Keep the canonical recurrence order. Replacing a sequential recurrence with an algebraically parallel scan may alter rounding and needs a separate numerical qualification. Copy neither another backend's threadgroup geometry nor its reduction tree as an architectural requirement.

### 3.4 MoE changes the speculation cost model

The checked-in config has 512 routed experts and selects 10 per token. A verify window does not necessarily reuse one fixed expert set across all rows. [L5]

For illustration only, if each row independently selected 10 uniformly distributed experts from 512, the expected union for eight rows would be:

```text
512 × [1 - (1 - 10/512)^8] = approximately 74.74 distinct experts
```

This is approximately 7.47 times one row's selected-expert count, not a measured routing distribution. Real adjacent tokens can be much more correlated, and caches, shared experts and non-expert layers change the final cost.

Consequently, a larger verify window is not equivalent to rereading the same expert weights once and obtaining eight tokens almost free. Log distinct experts, assignments per expert, overlap across rows, routed weight bytes and grouping/padding overhead. Select the direct tiny-row path versus grouped execution using the observed histogram, not only the total row count.

Keep routing and expert IDs on the device. Stable tie-breaking and row restoration are correctness requirements. A bounded device buffer can carry changing routing through a fixed graph; host-visible routing that forces a device synchronization defeats that benefit. The inspected implementation separates router work, expert gate/up work and expert reduction/write-back, providing useful fusion boundaries rather than requiring one monolithic kernel. [T2]

Do not make cold expert-weight fetches from disk or another node the default hot path merely because lookup tables can be file-backed. Expert execution placement and sparse-table lookup are different problems.

### 3.5 Drafting should optimize the whole cycle, including no drafting

The inspected depth controller estimates expected committed tokens from depth-conditional acceptance and divides by estimated or observed round cost. It updates estimates and occasionally explores a deeper choice. Its `_depth` method selects among positive depths; this does not prove there is no disable policy elsewhere. [T4]

Our global controller should explicitly compare depth zero with all qualified positive depths:

```text
cost_per_committed_token(d) =
    [draft + verify + commit/replay + uncovered_communication + control] /
    expected_committed_tokens
```

Use mutually exclusive timing categories to avoid counting a replay or communication interval twice. Startup transition cost, graph creation and ordinary steady-state cycles should be reported separately, without hiding startup from end-to-end latency.

The cost key should include context bucket, concurrency, numerical plan, topology and relevant MoE routing statistics. Do not import acceptance priors, fixed draft limits or extrapolated wide-window costs as measured AMD constants. Add hysteresis and bounded exploration to avoid oscillation.

The Flash Next runtime also contains a restricted-vocabulary draft head and queues draft generation. Restricting the *proposal* vocabulary is a legitimate experiment because the unchanged target still verifies emitted tokens. It must not restrict the target's own vocabulary or sampling distribution. Draft-head layout/cache bytes and acceptance changes belong in the comparison. [T5]

Start with the model's supported MTP mechanism and inexpensive context-copy proposals where appropriate. Additional trained draft heads are separate model artifacts, not a universally reusable backend optimization.

### 3.6 Prefix reuse, prefill and graph execution must share a plan identity

The inspected runtime fixes its prefill arithmetic identity before using snapshot keys and rejects a changed identity. It also distinguishes limiting queued layers from changing the mathematical prompt chunk plan. That is an important distinction for deterministic resume. [T5]

The lab already requires chunk-plan and kernel-plan identity and fair prefill/decode scheduling. The amendment should connect them explicitly: yielding execution is allowed, but a scheduler must not silently select a different rounding-sensitive prefill algorithm or chunk decomposition because another request arrived. Use canonical arithmetic chunks or prove finer subdivisions invariant. [L4][L6]

Cache admission should evaluate whether a new checkpoint can fit before evicting useful entries. Store valid state, not allocated capacity; drop temporary rollback and draft buffers when idle. Ensure all state classes agree at a retained boundary, including partial sparse-index blocks and short histories.

For HIP graphs, retain a correct eager baseline, finite qualified shapes, stable device inputs, and explicit dependencies. AMD documents both capture and explicit construction and notes that capture can preserve unnecessary stream ordering. Compare graph construction strategies only where the trace shows a benefit. [A3]

A graph must own an allocation generation. Growing or replacing a referenced buffer invalidates affected instances or requires an explicitly valid update. Graph pools, scratch and simultaneous live shapes count toward admission. Do not assume feature parity with another graph API or insert blocking USB4 reads/writes into a GPU graph.

## 4. UMA: specialization without magical zero-copy assumptions

The Ryzen AI Max+ 395 product specification lists a 256-bit LPDDR5x interface up to 8000 MT/s and two native 40 Gbit/s USB4 ports. The ideal memory-interface rate derived from those numbers is 256 GB/s; this is not a measured sustainable bandwidth. The mini PC vendor's product page also lists two USB4 Type-C ports. None of these specifications establishes dual-link payload independence. [A4][A5]

One physical memory pool does not imply identical performance for all mappings. HIP distinguishes coherence modes and notes that fine-grained coherence can constrain caching. Its documentation also distinguishes synchronization visibility and system-scope release behavior. Those descriptions are not proof that an arbitrary external USB4 DMA writer is synchronized with a GPU consumer. [A6]

Extend the UMA lab around buffer purpose:

| Purpose | Candidate policy | Required measurement |
| --- | --- | --- |
| Hot packed weights | Qualified cached compute mapping or copied hot buffers | Sustained operator bandwidth, startup duplication, mapping limits |
| GDN and KV state | GPU-efficient persistent storage with explicit ownership | Read/write traffic, append/update latency, retained capacity |
| Sparse lookup tables | Reclaimable file mapping plus bounded row gather/cache | Cold faults, warm locality, tail latency, page-cache pressure |
| Transport payload | Ordinary copy/staging baseline first | End-to-end GPU producer-to-consumer latency and CPU cost |
| Control flags | Minimal qualified coherent region if needed | Correct visibility, fence cost, polling cost |

Do not reserve the entire checkpoint as pinned memory by default. File-backed bytes can become resident and consume real RAM; a second repacked layout also has a peak and steady-state cost. Preserve source precision and quantization parameters during lossless repacking. [L7]

On this APU, evaluate CPU row-gather workers, transport progress threads and GPU work together. More CPU threads or permanent busy polling can consume shared memory bandwidth and package resources. Do not transplant a 16-thread gather setting or a spin-loop policy from a different machine. The NPU is not assumed to add its advertised throughput to this GPU executor; any future use needs an explicit operator, state and transfer contract.

## 5. Dual USB4: concrete implementation implications

### 5.1 Existing decisions are sound, but the probe is still discovery only

Documents 12 and 13 already distinguish controllers, routers, physical ports, peers, configured streams and qualified links, and explicitly require simultaneous measurements. The implementation currently enumerates domains, host routers and `tbstream` device names; it does not yet implement the full mapping or active qualification. This is an implementation gap, not a missing high-level architecture. [L8][L9]

Add a stable mapping:

```text
host identity
  -> host router / PCI identity
  -> physical port
  -> XDomain route and peer identity
  -> configured stream and device index
  -> measured physical-link profile
```

Multiple character devices can refer to logical streams over one physical connection. Two `tbstream` names must never become two independent bandwidth resources simply by counting device nodes. Linux explicitly supports multiple named streams on an XDomain connection and coexistence with networking. [K1]

### 5.2 The inspected driver is a copy-path baseline, not GPU RDMA

The reviewed driver exposes `read_iter`, `write_iter`, `poll`, `open` and `release`; the inspected file-operations table has no `mmap` operation. Its read path uses `copy_page_to_iter`. USB4STREAM defines DATA packets up to 4 KiB, mandatory end-to-end flow control, and CLOSE handling. [K2]

Consequences:

- Direct streaming means bypassing conventional IP transport, not automatically bypassing copies.
- Mapped-slot, DMA-BUF and direct GPU-DMA paths in the architecture remain future capability tiers, not properties of this backend.
- A userspace transfer needs length-bounded framing, partial-read/write handling, errors, EOF, cancellation and bounded buffering.
- Local write completion is not proof of peer GPU visibility or semantic commit.
- Use an explicitly supported kernel/runtime synchronization contract before enabling any external-DMA buffer path.

The protocol envelope should identify model/partition plan, request, connection epoch, speculative round, tensor role, dtype, shape, byte range and sequence number. Enforce bounds before allocating. After reconnect, stale or duplicate tensors must not advance committed state. Failover is safe only when ownership and commit position are known.

### 5.3 Interrupt mode and busy polling are different backends

The inspected implementation offers ring size, interrupt throttling and busy polling controls. Its `poll` handler returns `EPOLLERR` when busy polling is enabled; the busy-poll configuration cannot be changed while the stream has users. Therefore an epoll-based progress engine must not treat busy polling as a transparent tuning flag. [K2]

Qualify two explicit progress policies: interrupt/event-driven and dedicated polling. Record CPU time, IRQ rate, frequency/power behavior, p99 transfer latency and token cost under GPU load. Larger rings can improve bulk throughput while worsening queued latency; report queue depth and in-flight bytes.

Useful baseline matrix:

| Physical cables | Logical streams | Purpose |
| --- | --- | --- |
| One | One full-duplex stream | Simplest functional and performance baseline |
| One | Multiple streams | Separate flows and evaluate software contention without claiming more physical bandwidth |
| Two | One per cable | Prove simultaneous physical-link behavior |
| Two | Multiple, explicitly mapped | Evaluate control/bulk separation and flow scheduling |

### 5.4 Scheduling policy should follow message size and dependencies

One link already has bidirectional operation to qualify. Permanently assigning one cable to each direction can leave capacity idle and is not inherently an improvement. Independent flows and priority separation should be evaluated before splitting every tensor.

For a message of P bytes, a simple diagnostic model is:

```text
one_link = alpha_one + P / B_one
striped  = alpha_dual + reassembly + P / B_aggregate
```

Here B_aggregate is measured under concurrent transfer, not the sum of two isolated bandwidth results. Include GPU producer release, staging, queue wait, receiver acquire and exposed synchronization in the end-to-end comparison. Fit by message class; do not use a global threshold copied from another transport.

One-way latency requires a defensible clock synchronization/calibration method. Otherwise report RTT and local GPU-to-GPU cycle timing. Dividing RTT by two should be labelled an estimate, not a measurement of asymmetric one-way latency.

### 5.5 Derive boundaries from actual tensor shapes

The development config uses `hidden_size=2560` and four hyper-connection streams. The inspected runtime carries these residual streams before final mixing. A natural full-residual boundary between blocks therefore has, under BF16 storage:

```text
4 × 2560 × 2 = 20,480 bytes = 20 KiB per row
8 rows -> 160 KiB
```

A mixed-hidden boundary can differ. The planner must use the actual boundary tensor and any auxiliary state, not blindly assume `hidden_size × dtype_size`. These figures exclude protocol/control traffic and are not the total bytes per generated token. [L5][T5]

At a nominal 40 Gbit/s, an idealized 20 KiB payload serialization time is approximately 4.1 microseconds before all overheads. This is an arithmetic floor, not a USB4 measurement. It illustrates why small-message synchronization and launch cost may dominate; a second cable need not help a C1 layer boundary even when it helps prefill or larger verification windows.

Two native 40 Gbit/s ports have an ideal combined nominal rate of 10 GB/s if independently usable, far below the ideal local memory-interface rate. Remote memory must remain explicit storage/execution placement, not transparent local UMA. [A4]

## 6. Partitioning: distinguish capacity from latency

### Whole-layer or block pipeline partition

This is a useful first capacity baseline because layer-local KV and recurrent state stay with their owner and the number of boundaries is small. For a single autoregressive stream, the next token depends on completion of the previous token. Splitting sequential layers between hosts does not automatically overlap their work or double memory-read speed.

Pipeline utilization can improve with independent requests, prompt chunks or qualified verification work, but the planner must model real dependencies. Do not advertise a capacity gain as a C1 latency gain.

### Tensor-parallel partition

This introduces more collectives, but can also reduce local projection/expert work by running partitions concurrently. It should not be rejected merely because it adds synchronization. Compare:

```text
T_two_nodes = concurrent compute critical path
              + uncovered communication
              + synchronization/control

beneficial for latency when T_two_nodes < T_one_node
```

Measure actual collective phases, bytes, reductions and barriers. A model with large saved weight traffic may justify communication that a smaller model cannot. MoE imbalance and heterogeneous host power/thermal behavior must be included.

Partition/rank order and reduction arithmetic form part of the numerical-plan identity. Require speculative-vs-serial equivalence inside a fixed partition plan. When partitioning changes floating-point order, validate model fidelity separately rather than promising bitwise equality with an unpartitioned execution.

### Expert placement, lookup placement and remote drafting

Expert parallelism is a later candidate if routing locality, balance and payload histograms justify it. Prefer executing resident expert weights on their owner and moving activations, not repeatedly transporting weight blocks. Keep the state-owner relationship explicit.

Large sparse lookup tables can be placed or replicated independently when that removes a critical remote dependency; assess per-node memory and disk/cache state before replication.

A remote draft engine is only attractive if it removes enough local draft cost to pay for synchronization and any target-state inputs. Native MTP can consume target hidden/residual state, so sending only token IDs is not a sufficient general interface. Do not promise remote drafting as an immediate two-host speedup.

With exactly two ports per host, three hosts can form a triangle using both ports on every host. That topology does not also provide two cables per pair. Dual-link two-host and single-link triangle are distinct capacity/performance profiles.

## 7. Proposed amendments to the normative guide

The following are proposed clauses, not already-applied architecture changes. Preserve existing coverage instead of duplicating it.

| Document | Proposed amendment |
| --- | --- |
| 03 — Model loading | Replace synchronization-count-only rejection with total critical-path cost. Add explicit boundary shape/dtype/auxiliary-state bytes. |
| 04 — Execution | Bind graph instances to allocation generations; invalidate or safely update on growth; make prefill arithmetic identity immutable during cache use. |
| 05 — Kernels | Add expert-union/rows-per-expert metrics and direct-vs-grouped selection; distinguish logical low-row width from matrix-tile padding. |
| 06 — Numerics | Separate external model fidelity, within-plan exact equivalence, and separately qualified numerical/partition modes. |
| 07 — State | Compare full, minimal and sparse-checkpoint rollback using both retained bytes and per-cycle traffic; explicitly test partially completed sparse blocks. |
| 08 — Scheduling | Separate scheduling yields from arithmetic chunk changes; bind CPU progress/gather budgets to APU-load measurements. |
| 09 — Speculation | Include depth zero, exposed communication, route distribution and committed-token time in the global controller; qualify restricted draft vocabulary separately from target sampling. |
| 10 — Qualification | Add ablations, state-traffic and routing histograms, clock-method disclosure, and explicit numerical-plan identity. Consolidate repeated multi-node checklist text. |
| 12 — USB4 | Identify the implemented copy-based capability tier; distinguish interrupt/polling progress modes; formalize frame/epoch/commit protocol and measured striping thresholds. |
| 13 — Discovery | Implement XDomain-to-physical-link mapping; distinguish unknown/inaccessible evidence from proven absence; record driver API/progress capabilities. |
| 14 — Roadmap | Add explicit semantic-oracle and measured-cost deliverables before admitting optimized paths; retain isolated transport experiments without making them a single-node dependency. |

Suggested replacement for the over-broad rule in document 03:

> A placement MUST be rejected for the latency objective when its measured critical-path compute, uncovered communication and synchronization cost is worse than the qualified baseline. Additional synchronization points alone MUST NOT disqualify a placement. A capacity-oriented placement MAY remain available under an explicitly different objective.

Suggested numerical-contract clarification:

> Model fidelity and within-plan execution equivalence MUST be tested separately. A numerical plan MUST identify weight interpretation, accumulation, sampling, prefill decomposition and distributed reduction order. Changing the plan MUST invalidate incompatible retained state and MUST trigger the applicable fidelity qualification.

Suggested transport clarification:

> A logical stream MUST reference a physical-link identity. Logical streams sharing a physical connection MUST share its measured capacity budget. External-DMA or mapped-buffer capability MUST NOT be inferred from the availability of direct streaming.

## 8. Prioritized experiment plan

| ID | Experiment | Inputs and comparison | Admission/exit gate |
| --- | --- | --- | --- |
| E0 | Semantic oracle | Real development shapes; trusted forward/state; sparse block boundaries; deterministic ties | No unresolved interpretation of the optimized operator; independent fixtures retained |
| E1 | Native platform/residency | Real memory profiles; cached/coherent mappings; cold/warm file lookup; bounded staging | Verified `gfx1151`; stable physical-memory and mapping accounting; no host-pressure failure |
| E2 | Packed low-row projection | Same weights; rows 1/2/4/8; vector vs native matrix candidate; prefill separately | Row contract passes; no model-wide expansion; whole-chain benefit exceeds measured noise |
| E3 | Recurrent window | Full vs minimal vs sparse snapshots; multiple acceptance depths; long continuation | Correct accepted state after every rejection position; lower full-cycle cost on a defined workload |
| E4 | MoE paths | Same recorded routing; tiny-row vs grouped; correlated and dispersed routes | Same routing/output contract; measured union/assignment/padding costs; no host route synchronization |
| E5 | Graph replay | Eager vs finite graph plans; changing routes; buffer growth and model reload | Correct outputs/lifetimes; no stale pointers; bounded graph memory; measured launch-cost reduction |
| E6 | Speculative controller | Serial/MTP/context-copy where supported; depth zero and positive widths; draft-vocabulary ablation | Net committed-token benefit or reliable automatic fallback; same fixed-plan output |
| E7 | Prefix/fairness | Fresh/resumed; one long conversation; active decode plus incoming long prompts | Matching state/output contract; bounded prefix memory; better tail latency without starvation |
| E8 | USB4 micro-lab | One/two physical cables; one/multiple logical streams; QD1 and bulk; interrupt vs polling; GPU idle/load | Correct framing/recovery and physical mapping; complete latency/CPU/IRQ/bandwidth profiles |
| E9 | Distributed executor | Whole-layer baseline vs selected tensor partitions; one/two cables; decode/prefill/verify | Meets a named capacity or latency objective; numerical-plan validation; no stale state on failure |

Within each experiment, change one material factor at a time and alternate run order to expose thermal/cache bias. Run representative code, prose and multi-turn workloads rather than assuming one high-acceptance prompt is representative. Preserve tokenizer/rendered-token identity, output token count, sampling and cache conditions. Report TTFT separately from steady decode and report latency distributions, not only averages. Existing benchmark requirements remain applicable. [L10]

The sequence is: semantic and platform truth; packed/recurrent/MoE operator qualification; a correct local serial executor; bounded graph and speculative execution; then measured distributed execution. Transport discovery and microbenchmarks can proceed as isolated host-side work earlier. No production dependency should bypass the existing roadmap gates. [L11]

## 9. Explicit non-goals and claims not made

This review does not claim a particular speedup, sustainable memory bandwidth, USB4 payload rate, optimal draft depth, optimal CPU thread count, or optimal power profile. It does not claim that all external execution study paths use the same rollback strategy or that a selected controller method describes every outer disable policy.

No Apple/NVIDIA backend, native vendor-specific low-precision execution path, generic multi-vendor abstraction, or accelerator-to-accelerator coherent memory model is proposed. A storage format bearing a vendor name is not itself proof of an instruction requirement, but adding more formats is separate from optimizing the first supported native path.

Latent-only MLA storage is an architecture-specific algebraic optimization, not a generic replacement for GQA/QSA KV. Aggressive KV quantization, NPU integration, general remote-expert paging and large-cluster orchestration should remain separate experiments unless target measurements change their priority.

The practical objective is lower time and memory traffic per *committed* token on `gfx1151`, with a correct, reproducible model and useful long-context capacity. Two hosts and two cables are resources to schedule, not speedup multipliers.

## Sources

### Lab snapshot

All lab references below use commit `94d4333b7607343eabf64124ad0a1fbfd15b1861`.

- [L1] [Implementation status](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/IMPLEMENTATION_STATUS.md).
- [L2] [Architecture guide scope](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/README.md).
- [L3] [Numerical rules](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/06-quantization-and-numerics.md).
- [L4] [State and snapshots](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/07-state-kv-and-recurrence.md).
- [L5] [Checked-in development configuration](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/artifacts/qwen-official-config-de4b8e4.json).
- [L6] [Scheduling](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/08-scheduling-and-concurrency.md).
- [L7] [Memory](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/02-memory-and-residency.md) and [loading/layout](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/03-model-loading-and-weight-layout.md).
- [L8] [Multi-node/USB4](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/12-multi-node-and-usb4.md) and [discovery](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/13-platform-discovery.md).
- [L9] [Probe implementation](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/src/platform_probe.cpp).
- [L10] [Benchmark qualification](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/10-benchmarking-and-qualification.md).
- [L11] [Implementation roadmap](https://github.com/hyukudan/strix-halo-engine-lab/blob/94d4333b7607343eabf64124ad0a1fbfd15b1861/docs/architecture/14-implementation-roadmap.md).

### Inspected comparative implementation

- [T0] private source record.
- [T1] private source record.
- [T2] private source record.
- [T3] private source record.
- [T4] private source record.
- [T5] private source record.

### Platform and transport primary references

- [A1] [AMD: WMMA on RDNA 3](https://gpuopen.com/learn/wmma_on_rdna3/). Instruction concepts, not a Strix Halo benchmark.
- [A2] [LLVM AMDGPU backend guide](https://llvm.org/docs/AMDGPUUsage.html). Current target documentation; deployed compiler capabilities still require validation.
- [A3] [AMD HIP graph tutorial](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/graph_api.html).
- [A4] [AMD Ryzen AI Max+ 395 specifications](https://www.amd.com/es/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html).
- [A5] The mini PC vendor's specification of the Ryzen AI Max+ 395 system (128 GB LPDDR5x-8000, two USB4 ports).
- [A6] [AMD HIP coherence control](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html).
- [K1] [Linux USB4/Thunderbolt administration guide](https://docs.kernel.org/admin-guide/thunderbolt.html).
- [K2] [Linux USB4STREAM implementation](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/drivers/thunderbolt/stream.c). In particular: protocol description and ring/progress controls near the start; read/write/poll handling around lines 680–950; file operations and busy-poll configuration around lines 1005–1060.
