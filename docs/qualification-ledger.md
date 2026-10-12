# Current qualification ledger

Updated: 2026-10-12. Earlier slices in `IMPLEMENTATION_STATUS.md` record historical
implementation steps. This table summarizes current gates without promoting
portable fixtures, declarations or offline compilation to native qualification.

| Area | Available implementation and checks | Required target evidence |
| --- | --- | --- |
| Memory/model contracts | Checked physical-pool accounting, GGUF metadata, tensor/runtime bindings and admission | Complete canonical checkpoint ownership and observed physical peak |
| Model execution | Portable semantic oracles, compiled serial runtime seams and experimental HIP implementation | Full-model tensor coverage, logits/tokens/state fidelity on gfx1151 |
| Low precision | Integer/affine/codebook references, bounded packed staging, row/group activation preparation and shared opt-in Wave32 IU8/IU4 launcher with owned resources and epoch-bound completion | Correct native fragment layout, preparation-inclusive projection costs, full-model integration and model-quality gates |
| Instruction emission | Local ROCm 7.2.4 emitted Wave32 IU8/IU4 for gfx1151; artifacts retained privately | Native execution correctness, achieved throughput and complete-workload benefit |
| Distributed boundary transport | Logical/physical identity separation, epoch-bound replay history, explicit consumer completion, RX quarantine and v2 control envelope | Concrete importer/event backend, GPU visibility, consumer drain and fresh-epoch recovery |
| Platform/link discovery | Linux/sysfs discovery and bounded CPU baseline tooling | Two-peer ABI, port/router mapping, negotiated links and physical independence |
| Character-device control/payload | Owned POSIX nonblocking FD, bounded framing, v2 control and dedicated C1 host-copy tensor payload with typed completion, replay and fault regressions | Actual configured two-peer stream/control behavior, throughput and model continuity; GPU DMA remains unavailable |
| Native interface preflight | Exact identities plus bounded driver adapter with import/quiescence ownership, operation-bound visibility/event proofs and same-epoch sequence history | Real driver syscall/GPU provider, native capability query and trusted target evidence; default provider is unavailable |
| Real two-node model | Portable placement and control contracts | State continuity, numerical replay, failure containment and single/two-link measurements |
| Prefix lineage | Actual serial completion history, complete chunk/row provenance, mismatch cache misses and failure taint | Qualified chunked-prefill/verify histories and native fresh/resumed state correspondence |
| Graph eligibility | Current bindings/generations/residency, bounded token/position ABI, experimental embedding gather and device-position norm/RoPE/KV stage with checked memory and failure status | Existing whole-model eager plans remain ineligible; remaining dynamic operators, whole-model capture/replay and actual graph-memory measurement |
| Host staging | Event-owned source pool and opt-in HIP PLE consumer, bounded bytes/progress, stale ticket/binding rejection and retained failure ownership | Native event/visibility/lifetime proof and preparation-inclusive benefit; destination lifetime is caller-owned |
| Projection reduction | Fixed-tree shuffle-tail candidate, host sensitivity controls and experimental operator HIP graph replay | Native exact fixtures and end-to-end quality/performance; no measured gain yet |

The latest completed strict Release integration passed 126 portable CTest entries.
This does not establish native GPU execution or DMA. The transport interfaces require
one serialized progress owner. Completion budgets count polls, not elapsed time.
Retained lease-registry history has no reclamation mechanism; replacement requires
drained consumers/backends and fresh peer-agreed connection epochs.
Typed backend release now retains the host registry lease on unresolved native
ownership, including failed open and session destruction; no release proof means
no replacement admission through that registry.

The next hardware campaign must distinguish CPU transport, GPU buffer import,
commit visibility, consumer completion, repost/rearm and complete model benefit.
Each stage has its own rejection criteria. No native direct path is enabled by
device-node presence, theoretical precision throughput or an external result.

See [the portable implementation slice](implementation-2026-10-11.md),
[reusable execution components](implementation-2026-10-12-execution-components.md),
[low-precision lab](../lab/low_precision/README.md) and the architecture guides.
