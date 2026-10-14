# Current qualification ledger

Updated: 2026-10-12. Earlier slices in
[IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) record historical
implementation steps. This table summarizes current gates without promoting
portable fixtures, declarations or offline compilation to native qualification.

Read each row across: the middle column describes available implementation and
checks, while the final column identifies the evidence still required on the
target. An implementation being available does not imply that its final-column
gate has passed. The date above belongs to the recorded qualification evidence;
editorial changes to this page do not represent a new hardware campaign.

| Area | Available implementation and checks | Required target evidence |
| --- | --- | --- |
| Memory/model contracts | Checked physical-pool accounting, GGUF metadata, tensor/runtime bindings and admission | Complete canonical checkpoint ownership and observed physical peak |
| Model execution | Portable semantic oracles, compiled serial runtime seams and experimental HIP implementation | Full-model tensor coverage, logits/tokens/state fidelity on gfx1151 |
| Low precision | Integer/affine/codebook references, bounded packed staging, opt-in Wave32 IU8/IU4 launcher and actual tiled LM-head runtime adapter with exact IEEE widening and atomic full-output publication | Native fragment correctness, preparation-inclusive projection costs and full-model quality; integer LM-head is eager C1 only |
| Instruction emission | Local ROCm 7.2.4 emitted Wave32 IU8/IU4 for gfx1151; artifacts retained privately | Native execution correctness, achieved throughput and complete-workload benefit |
| Distributed boundary transport | Exact identity/replay contracts, concrete Linux importer/event provider, GPU packing/full-payload acquire/unpacking, typed consumer completion and dedicated v2 control | Actual exporter/controller compatibility, external-DMA visibility, peer lifecycle and model continuity; first provider is serialized C1 |
| Platform/link discovery | Linux/sysfs discovery and bounded CPU baseline tooling | Two-peer ABI, port/router mapping, negotiated links and physical independence |
| Character-device control/payload | Owned POSIX FD, bounded framing, v2 control and C1 host-copy payload; separate actual opt-in DMA-BUF kernel extension with strict enabled-object compilation | Configured two-peer behavior, throughput and model continuity; no module load or physical DMA execution yet |
| Native interface preflight | Explicit experimental UAPI, concrete syscall/HIP providers, uncached allocator, guarded imports, operation-bound event proofs and same-epoch sequence history | Independent exact-binding target evidence and real driver/GPU visibility/drain observations; default provider remains unavailable |
| Real two-node model | Portable placement and control contracts | State continuity, numerical replay, failure containment and single/two-link measurements |
| Prefix lineage | Actual serial completion history, complete chunk/row provenance, mismatch cache misses and failure taint | Qualified chunked-prefill/verify histories and native fresh/resumed state correspondence |
| Graph eligibility | Separate fixed C1 graph topology, complete device-input embedding/PLE/GDN/indexed attention, owned capture/replay lifecycle and whole-DAG native fixtures compiled offline | Full-model native logits/state fidelity, actual graph memory and preparation-inclusive latency; eager plans remain distinct |
| Host staging | Event-owned source pool and opt-in HIP PLE consumer, bounded bytes/progress, stale ticket/binding rejection and retained failure ownership | Native event/visibility/lifetime proof and preparation-inclusive benefit; destination lifetime is caller-owned |
| Projection reduction | Fixed-tree shuffle-tail candidate, host sensitivity controls and experimental operator HIP graph replay | Native exact fixtures and end-to-end quality/performance; no measured gain yet |
| Campaign preparation and capture | Private full-kernel/module staging, guarded observations, imported-pool reuse and paired commissioning, eager/graph/integer RAW recording and actual same-plan prefix restoration with full-state/lineage import checks | Reviewed target boot/configuration, actual paired transfers, observed tokenizer/source bindings and independent model references; schedule completion does not prove physical DMA overlap |

The latest completed strict Release integration passed 143 portable CTest entries.
The final source also passed 29 selected ASan/UBSan checks with leak detection and
five HIP host-contract tests. The complete gfx1151 HIP build and the enabled Linux
driver object compiled offline. The complete pinned kernel, modules and private
staging passed modpost, dependency, vermagic and GNU build-ID checks without
installation or boot. The serial graph fixture validated its host
contracts without GPU work. Compiler artifacts and detailed receipts remain private.
This does not establish native GPU execution or DMA. The transport interfaces require
one serialized progress owner. GPU commissioning waits now have separate poll
and monotonic elapsed-time budgets with cooperative pauses. These budgets do not
interrupt a blocked runtime or driver call. The production transport remains C1;
isolated paired commissioning does not enable production concurrency.
Retained lease-registry history has no reclamation mechanism; replacement requires
drained consumers/backends and fresh peer-agreed connection epochs.
Typed backend release now retains the host registry lease on unresolved native
ownership, including failed open and session destruction; no release proof means
no replacement admission through that registry.

The next hardware campaign must distinguish CPU transport, GPU buffer import,
commit visibility, consumer completion, repost/rearm and complete model benefit.
Each stage has its own rejection criteria. No native direct path is enabled by
device-node presence, theoretical precision throughput or an external result.

## Evidence collection entry points

| Evidence you need | Where to begin |
| --- | --- |
| Portable contracts and their implementation boundaries | [Portable foundations](implementation-2026-10-11.md) and [reusable execution components](implementation-2026-10-12-execution-components.md) |
| Native arithmetic, model fidelity and state replay | [Numerical campaign preparation](implementation-2026-10-12-native-numerical-campaign.md) and [numerical lab runbook](../lab/native_campaign/numerical/README.md) |
| Eager, graph or integer-head recording contracts | [RAW baseline recording](implementation-2026-10-12-raw-baseline-recording.md) and [complete-model decode modes](implementation-2026-10-12-decode-execution-modes.md) |
| Precision emission and operator correctness | [Low-precision lab](../lab/low_precision/README.md) and [integer runtime slice](implementation-2026-10-12-integer-runtime.md) |
| Target commissioning and direct transport lifecycle | [Hardware campaign preparation](implementation-2026-10-12-hardware-campaign-preparation.md) and [commissioning runbook](../lab/direct_gpu_stream/COMMISSIONING.md) |
| Complete-workload acceptance and reproducibility | [Benchmarking and qualification](architecture/10-benchmarking-and-qualification.md) |

Retain failures, unknowns and cleanup outcomes as evidence alongside successful
checks. Each result must stay bound to the exact artifacts, configuration and
runtime under assessment; a passing host evaluator cannot supply an unobserved
native execution or an independent model reference. Follow the relevant runbook
for the exact schema, required observations and rejection criteria.

Return to the [documentation guide](README.md) for architecture decisions and
the full implementation-report index.
