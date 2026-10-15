# Current qualification ledger

Updated: 2026-10-12. Earlier slices in `IMPLEMENTATION_STATUS.md` record historical
implementation steps. This table summarizes current gates without promoting
portable fixtures, declarations or offline compilation to native qualification.

| Area | Available implementation and checks | Required target evidence |
| --- | --- | --- |
| Wave router top-k | OFF by default, distinct parallel wave selection and identities; WSL GNU13.3 Release HIP-OFF152/152 tests,6864 independent host insertion/wave cases; typed poisoned eager/captured device fixtures supplied | Actual Core10 HIP compile, softmax/epilogue ISA and FP modes, old-kernel preservation, native device fixtures, full554 arrays/512 IDs and preparation-inclusive paired measurements remain pending |
| Deferred eager MoE status | Opt-in C1 first-error domain/full bits, owned completion stream, prepublication session hook and transactional host histories; shared host/device first-error helper, exact admission and portable middle-layer/fence/host-precedence tests | HIP compilation, native invalid-route/status fixtures, complete-model array equality and preparation-inclusive token latency |
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
| Q8 indexed-attention input prefetch | Default-off per-matrix eligibility, mixed-layer IDs, host planning tests, actual-dispatch receipts, poisoned standalone/full-operator fixtures and zero-node failure checks | Linked legacy ISA stability, candidate ISA/descriptors, execution of the new native fixtures, full-model exactness and measured workload benefit |
| Projection reduction | Fixed-tree shuffle-tail candidate, host sensitivity controls and experimental operator HIP graph replay | Native exact fixtures and end-to-end quality/performance; no measured gain yet |
| Campaign preparation and capture | Private full-kernel/module staging, guarded observations, imported-pool reuse and paired commissioning, eager/graph/integer RAW recording and actual same-plan prefix restoration with full-state/lineage import checks | Reviewed target boot/configuration, actual paired transfers, observed tokenizer/source bindings and independent model references; schedule completion does not prove physical DMA overlap |
| Lane-chain HC output | Experimental opt-in, pure metadata admission and per-step base fallback, exact ordered-chain source and poison/graph fixtures; see [implementation](implementation-2026-10-15-lane-chain-hyper-output.md) | Source review, HIP/ISA, device fixtures, complete-logit identity and paired workload measurements remain open |
| Q8 shared-expert gate/up/down prefetch | Default-off per-tensor/layer selection, distinct IDs, original device definitions preserved, limited host planning tests, standalone and full-operator poison/graph fixtures, actual host submission count | Native HIP compilation, linked legacy ISA and candidate FP/arithmetic/resource descriptors, new device fixture execution, full-model exactness, paired complete-workload benefit |
| Exact HC projection prototype | Diagnostic-only OPW8/16/32 with P1/P2, original payloads, ordered chains and same-input poison/event fixtures | Source review, actual HIP/ISA no-spill/scalar-load checks, device exactness and real captured-input microbench; no decoder integration yet |

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

See [the portable implementation slice](implementation-2026-10-11.md),
[reusable execution components](implementation-2026-10-12-execution-components.md),
[low-precision lab](../lab/low_precision/README.md) and the architecture guides.

Draft integration check (2026-10-15): the combined default-off source passed a
fresh GNU 13.3/WSL Release HIP-OFF build with warnings as errors and 157/157 portable
CTest entries. This includes combined mixed-mode selection, base identity and
deferred graph rejection. Host syntax checks with private HIP declarations passed;
these do not establish native HIP compilation or execution. See
[the draft integration record](implementation-2026-10-15-spec-v1-draft-integration.md).
Combined review, actual native ISA/fixtures, model fidelity and paired timing remain
pending; earlier historical checks above retain their original scope.
