# Current qualification ledger

Updated: 2026-10-16. Earlier slices in `IMPLEMENTATION_STATUS.md` record historical
implementation steps. This table summarizes current gates without promoting
portable fixtures, declarations or offline compilation to native qualification.

| Area | Available implementation and checks | Required target evidence |
| --- | --- | --- |
| Execution inventory and host telemetry | Offline evidence collector, eager launcher counters and explicit GPU-event main/side queue spans with dependency versions; unknown kernel-busy/overlap metrics stay unknown; see [observability](execution-observability.md) | Compilation, trace/owner fault containment, target overhead measurements and native/full-workload checks; queue spans are not kernel execution intervals or a speed gain |
| Scalar greedy output and prefix head omission | Explicit eager generation consumer, persistent GPU argmax/result owner and state-only nonfinal prefix positions; see [output contract](gpu-greedy-output.md) | Final source compilation, full-logit CPU/GPU selection equivalence, complete model/state regression, failure containment and end-to-end timing; execution deferred during implementation |
| Recurrent GDN LDS layout | Test-only stride32, stride33 and transposed candidates with original recurrence retained; see [diagnostic contract](gdn-recurrent-layout-diagnostic.md) | Source/ISA operand and resource checks, native full-state/output equality and operator timing; no decoder adoption |
| Prepared eager submission | Explicit immutable compiled snapshot, cached order/PLE/write counts, strong owner and serial session adapter; see [prepared execution](prepared-eager-execution.md) | Compilation and source fixtures, full-model/state equality, failure containment and measured host/workload improvement; operator tensor resolution remains dynamic and qualification is pending |
| Hierarchical QSA selection | Independent diagnostic top-512 block selection, chronological expansion and sticky finite validation; see [selection contract](qsa-selection-diagnostic.md) | Compilation/native exact selection checks, decoder workspace/status integration, complete attention and long-context measurements; not linked into the decoder |
| Parallel HC projection diagnostic | Fixed split-K FP32 reduction, single-kernel R4/R2 mappings, double-accumulation diagnostic and same-arithmetic global-partial comparator; independent FP64 oracle and guarded experimental integration | Wider captured-input coverage and held-out quality remain required before adoption; experimental modes require explicit selection and do not change the qualified automatic profile |
| Wave router top-k | Low-level OFF in manual selection; content-bound adopted auto v3 profile selects wave with HC output, QSA, shared-expert prefetch and deferred status; historical host evidence: distinct parallel wave selection and identities; WSL GNU13.3 Release HIP-OFF152/152 tests,6864 independent host insertion/wave cases; typed poisoned eager/captured device fixtures supplied | Individual canonical source/native/ISA, full554 equality, generated512 identity and descriptive paired measurements are observed; wider workloads, stronger timing evidence and other model/context combinations remain separate gates |
| Deferred eager MoE status | Low-level/manual OFF; included by the adopted auto v3 profile after admission; C1 first-error domain/full bits, owned completion stream, prepublication session hook and transactional host histories; shared host/device first-error helper, exact admission and portable middle-layer/fence/host-precedence tests | HIP compilation, native invalid-route/status fixtures, complete-model array equality and preparation-inclusive token latency, outside the qualified v3 slice; see the adoption record below |
| Routed activation batching | Explicit eager A8 option submits one 2D SiLU grid across independent existing slot buffers, retaining finite scans and down ordering; see [activation contract](batched-moe-activation.md) | Joint source review, compilation/native equivalence, complete model/state/token fidelity and measured workload benefit; no default adoption |
| Shared-expert multirow weight reuse | BF16 rows R2/R4/R8 with Q8 coefficient reuse; explicit known-input windows and independent C2/C4 serving cohorts now have actual source consumers; see [serving](greedy-serving.md) and [prefill](known-prefix-prefill.md) | Source/ISA review, compilation, native/full-model state and output equality and complete-workload measurements; all newly integrated modes remain unqualified |
| QSA GQA head reuse | Test-only grouping of two/four queries sharing KV with existing per-head arithmetic; see [GQA diagnostic](qsa-gqa-reuse-diagnostic.md) | Compilation/native output and intermediate equality, resource/ISA review and complete attention/model cost; no decoder admission or observed gain |
| Eager shared/routed scheduling | Same retained expanded arena/physical lease supports fixed fork/join and explicit observed whole-token serial/fork/join selection, with independent output-intent controllers; see [policy](moe-token-cost-policy.md) | Final review, compilation, real event/owner failure containment, complete model/state equivalence and paired workload measurements; online observations are not benchmark evidence and defaults remain unchanged |
| UMA retained-owner inventory | Explicit backing lease deduplication, alias interval union, unknown reservations and separate system observations; see [inventory contract](uma-owner-inventory.md) | Compilation/source fixtures and actual allocator/runtime integration; no physical-residency or capacity claim |
| Tiled known-prefix prefill | Explicit T16..2048 step-major source consumer, bounded full-T striped workspace, retained PLE/state/D1 owners, closed row tiles and R1 tails; terminal/state-only output policies; see [prefill](known-prefix-prefill.md) | Final review, compilation, full-model row/state/continuation equivalence and TTFT/setup-peak measurements; no general GEMM or measured physical-fit claim |
| Compact lookup snapshots | Explicit occupied-KV format captures each full-attention layer's occupied prefix and all other components completely; canonical range/hash admission and interior-offset restore; see [lookup](prompt-lookup.md) | Compilation, native/model rollback and different-continuation equality over occupied state, future-row overwrite proof and observed complete-cycle cost; inactive KV tail is indeterminate and serving prefix caches stay full |
| Shared serving prefix cache | Bounded complete seven-component snapshots, content/runtime/history binding, exact/proper prefix consumers and original-stream batch completion | Compilation and native/model cache-hit/miss/fault/continuation checks, observed memory and complete serving latency; no partial-state cache promotion |
| Prompt lookup adaptation | Explicit online depth0/1/2 selector using completed cycle wall time, censored incomplete/final cycles and bounded re-exploration | Compilation, native/model exact commit/replay/bonus checks and complete serving workload benefit; online observations are not paired benchmark evidence |
| PLE row cache | Explicit preallocated content-bound BF16 row cache, strong mapped source owner, success-only insertion and admission before payload allocation | Compilation, cancellation/I/O-failure/cache-order fixtures, native/model correspondence and observed benefit under GPU load |
| Evaluated compilation provenance | Optional manifest binds per-TU evaluated arguments, declared dependencies/artifacts and running executable for offline tuning keys | Actual generated evidence and validated numerical/target keys; not observed compiler execution, driver-resource closure or qualification |
| Individual CTest evidence | Existing-JUnit collector and host CI source retain individual outcomes, inventory, diagnostics and skips; see [evidence](ctest-evidence.md) | Scheduled final execution; a skip, missing case or failed build cannot count as an all-pass receipt; historical runs do not qualify current source |
| Memory/model contracts | Checked physical-pool accounting, GGUF metadata, tensor/runtime bindings and admission | Complete canonical checkpoint ownership and observed physical peak |
| Model execution | Portable semantic oracles, compiled serial runtime seams and experimental HIP implementation | Full-model tensor coverage, logits/tokens/state fidelity on gfx1151 |
| Low precision | Integer/affine/codebook references, bounded packed staging, opt-in Wave32 IU8/IU4 launcher and actual tiled LM-head runtime adapter with exact IEEE widening and atomic full-output publication | Native fragment correctness, preparation-inclusive projection costs and full-model quality; integer LM-head is eager C1 only |
| Instruction emission | Local ROCm 7.2.4 emitted Wave32 IU8/IU4 for gfx1151; artifacts retained privately | Native execution correctness, achieved throughput and complete-workload benefit |
| Distributed boundary transport | Exact identity/replay contracts, concrete Linux importer/event provider, GPU packing/full-payload acquire/unpacking, typed consumer completion and dedicated v2 control | Actual exporter/controller compatibility, external-DMA visibility, peer lifecycle and model continuity; first provider is serialized C1 |
| Platform/link discovery | Linux/sysfs discovery and bounded CPU baseline tooling | Two-peer ABI, port/router mapping, negotiated links and physical independence |
| Character-device control/payload | Owned POSIX FD, bounded framing, v2 control and C1 host-copy payload; separate actual opt-in DMA-BUF kernel extension with strict enabled-object compilation | Configured two-peer behavior, throughput and model continuity; no module load or physical DMA execution yet |
| Native interface preflight | Explicit experimental UAPI, concrete syscall/HIP providers, uncached allocator, guarded imports, operation-bound event proofs and same-epoch sequence history | Independent exact-binding target evidence and real driver/GPU visibility/drain observations; default provider remains unavailable |
| Two-peer input and generation | Actual bounded streaming transaction, existing greedy cursor with dedicated selected-result control, explicit between-layer payload with separately proved position dependencies, selected local weight owner and typed p0 initialization; state/snapshot/scratch still full per peer | Final review/compilation, actual commissioned two-host model/state/fault checks, measured complete cost and standalone commissioning; local state packing and complete capacity qualification remain separate |
| Prefix lineage | Actual serial completion history, complete chunk/row provenance, mismatch cache misses and failure taint | Qualified chunked-prefill/verify histories and native fresh/resumed state correspondence |
| Graph eligibility | Separate fixed C1 graph topology, complete device-input embedding/PLE/GDN/indexed attention, owned capture/replay lifecycle and whole-DAG native fixtures compiled offline | Full-model native logits/state fidelity, actual graph memory and preparation-inclusive latency; eager plans remain distinct |
| Host staging | Event-owned source pool and opt-in HIP PLE consumer, bounded bytes/progress, stale ticket/binding rejection and retained failure ownership | Native event/visibility/lifetime proof and preparation-inclusive benefit; destination lifetime is caller-owned |
| Q8 indexed-attention input prefetch | Low-level/manual OFF; included by the adopted auto v3 profile after admission; per-matrix eligibility, mixed-layer IDs, host planning tests, actual-dispatch receipts, poisoned standalone/full-operator fixtures and zero-node failure checks | Linked legacy ISA stability, candidate ISA/descriptors, execution of the new native fixtures, full-model exactness and measured workload benefit, outside the qualified v3 slice; see the adoption record below |
| Projection reduction | Fixed-tree shuffle-tail candidate, host sensitivity controls and experimental operator HIP graph replay | Native exact fixtures and end-to-end quality/performance; no measured gain yet |
| Campaign preparation and capture | Private full-kernel/module staging, guarded observations, imported-pool reuse and paired commissioning, eager/graph/integer RAW recording and actual same-plan prefix restoration with full-state/lineage import checks | Reviewed target boot/configuration, actual paired transfers, observed tokenizer/source bindings and independent model references; schedule completion does not prove physical DMA overlap |
| Lane-chain HC output | Explicit in manual mode; selected by the content-bound auto policy; historical source increment: pure metadata admission and per-step base fallback, exact ordered-chain source and poison/graph fixtures; see [implementation](implementation-2026-10-15-lane-chain-hyper-output.md) | Isolated canonical source/native/fidelity and paired workload gates passed; combined automatic v3 (wave/QSA/HC/shared/deferred) qualified on the canonical slice; wider workloads remain open |
| Q8 shared-expert gate/up/down prefetch | Low-level/manual OFF; included by the adopted auto v3 profile after admission; per-tensor/layer selection, distinct IDs, original device definitions preserved, limited host planning tests, standalone and full-operator poison/graph fixtures, actual host submission count | Native HIP compilation, linked legacy ISA and candidate FP/arithmetic/resource descriptors, new device fixture execution, full-model exactness, paired complete-workload benefit, outside the qualified v3 slice; see the adoption record below |
| Exact HC projection prototype | Diagnostic-only OPW8/16/32 with P1/P2, original payloads, ordered chains and same-input poison/event fixtures | Source review, actual HIP/ISA no-spill/scalar-load checks, device exactness and real captured-input microbench; no decoder integration yet |

A historical strict Release integration passed 143 portable CTest entries.
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

Historical initial integration check (2026-10-15): the low-level default-off source passed a
fresh GNU 13.3/WSL Release HIP-OFF build with warnings as errors and 157/157 portable
CTest entries. This includes combined mixed-mode selection, base identity and
deferred graph rejection. Host syntax checks with private HIP declarations passed;
these do not establish native HIP compilation or execution. See
[the draft integration record](implementation-2026-10-15-spec-v1-draft-integration.md).
At this historical checkpoint combined review, native ISA/fixtures, model fidelity
and paired timing were pending. The qualified adoption below supersedes that status
for its explicit slice; earlier checks retain their original scope.

Automatic decoder policy follow-up: the application now proposes a content-bound
qualified eager/register profile and a manual A/B override. The actual parser and
resolver are portable-tested for model/content/mode/context/build capability,
explicit overrides and reasoned fallback. The initial policy checkpoint required review
and fresh integrated native qualification; see
[the policy contract](decode-optimization-profile.md). The earlier default-off
integration result records its original revision and scope.

Historical automatic-policy checkpoint: the preceding integrated source completed
225/225 actual native regression tests with no failures/skips, and preserved 225
legacy kernels under the linked restricted ISA comparison. A1 wave and C1 HC output
also have separate full554 array equality, generated512 identity and descriptive
paired workload observations. These preceding observations did not establish joint
automatic profile fidelity or speed; the later direct combination gates are recorded below. Individual gains
are not added.

The host compatibility follow-up retains automatic selection with no optimization
flags, preserves old explicit options unless auto is explicitly requested, validates
roundtrip capacity after profile resolution, and escapes QSA tensor names in JSON.
At that checkpoint changed host source required fresh review/native checks. Those
joint full554, automatic generated512 and paired gates subsequently passed for
the adopted revision below.

## Adopted canonical automatic v3

The exact source `265fd78762884d18eb32aa60294408f080088d52` is now primary.
Its own checks passed: 225 native tests, 14 linked modules/245 preserved kernels,
seven actual CLI cases, all 554 complete reference logit arrays and all 512 immutable
generated IDs in each of eight clean automatic runs. The same-node three-pair
descriptive result was +25.99% against the preserved baseline; an independent
three-pair comparison was +15.55% against the preceding automatic wave+HC profile.
No individual gains were added. See the [profile adoption record](decode-optimization-profile.md#qualified-adoption-2026-10-15)
for exact source/archive, timings, counters and limits. Low-level manual defaults
remain OFF; other models, contexts, graph/speculative modes and projection
prototypes are not admitted by these observations. This records source adoption,
not remote binary deployment.

## Consolidated development baseline

The development branches were consolidated into `main` on 2026-10-15:
qualified automatic v3 (`e40ba98`), exact HC repack diagnostics (`dcbffb0`),
guarded split-K HC experiments (`e747cf1`) and GDN/LM load and eight-wave
projection diagnostics (`0405647`). Subsequent integration work starts from
this shared branch. Historical source pins retain their original scope.

The consolidated source passed a fresh GNU 13.3 Release HIP-OFF build with
warnings as errors and all 162 portable CTest entries. This is host integration
evidence; it does not qualify new HIP binaries, experimental arithmetic,
whole-model quality or performance. The qualified automatic profile is preserved;
experimental split-K selection remains explicit and projection diagnostics remain
outside default decoder selection.

## Externally admitted HC repack research tests

The diagnostic `strix-hip-repacked-hyper-projection-tests` binary is built when
HIP test support is enabled. Its native CTest is explicitly opt-in through the
CMake cache STRING `STRIX_HC_REPACKED_ADMITTED_SCHEDULES`, which defaults to empty.
Only a nonempty, exact ascending comma-separated subset of `31,32,33` registers
the research test. Whitespace, duplicates, descending order, other IDs and list
separators are rejected. CTest sets that exact environment value itself; an
ambient shell variable does not configure or admit the test.

The value must come from an external compiler/ISA qualification receipt bound to
the actual source, executable and eligible schedule subset. CMake validates only
syntax; it does not manufacture or verify that receipt. A successful build,
resource bounds or source review alone do not close a failed pipeline gate. The
existing experimental STOP decisions remain intact, and no schedule is inferred
admitted by default. Direct diagnostic invocation retains its own explicit
schedule checks; it is not a substitute for qualification.

With the default empty value, configuration reports that this research test is
not admitted and omits it from the registered/passed standard-suite count. That
is an explicitly excluded research slice, not a native PASS, skip, numerical
qualification or completion of the HC optimization. Portable HC pack-contract
tests remain registered independently. Their success does not admit GPU
schedules. No kernel, numerical comparison or admission threshold changes here.
