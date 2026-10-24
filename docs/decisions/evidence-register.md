# Evidence register

Parallel HC projection diagnostic: source prototype with a distinct reassociated
FP32 numerical identity; see `docs/research/history/implementation-2026-10-15-parallel-hc-projection.md`.
The portable FP64 oracle passes strict GNU C++20 tests, including all 65536 half
bit patterns and independent indexing/cancellation controls. HIP compilation,
native execution, model quality and workload performance remain pending.

This file is the traceability source for the specification. Allowed states are:
**observed**, **published**, **inferred**, **hypothesis** and **pending**. Rows
marked **withdrawn** keep their ID so that existing references remain stable;
their content is not part of the public record.

| ID | State | Claim | Evidence | Consequence for the spec |
| --- | --- | --- | --- | --- |
| E-001 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-002 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-003 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-004 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-005 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-006 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-007 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-008 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-009 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-010 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-011 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-012 | hypothesis | A lossless internal format could reduce startup cost without duplicating too much memory/disk. | No prototype or measurement yet. | Keep it optional until load time, size and traffic are measured. |
| E-013 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-014 | pending | Best wave32/wave64 policy per family and shape. | Requires compiling and measuring both modes. | The registry must support both during research. |
| E-015 | published | HIP/Clang supports offline compilation for a GFX target via `--offload-arch`. | Official AMD HIP/Clang documentation. | Create a reproducible `gfx1151` lab before the hardware arrives. |
| E-016 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-017 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-018 | observed | `rocm/dev-ubuntu-24.04:7.2.4` compiles the probe offline for `gfx1151`, and the extracted HSACO declares that target, wave32 and zero spills. | Image by digest and hashes in `lab/cross_compile/README.md`. | Basic frontend/codegen is validated; execution and performance remain pending. |
| E-019 | observed | The design review recorded memory, residency, semantics and backend blockers, with a reduced initial scope. | Reproducible record in `docs/research/history/2026-09-28-design-acceptance.md`. | Do not freeze the spec until memory, UMA residency, canonical sources and `gfx1151` libraries are resolved. |
| E-020 | published | Qwen published the official Flash-Next repository, technical report and checkpoint; they describe 125B main parameters, 6B active per token and a 51B extra n-gram table, with 36 GDN layers and 12 QSA layers. | `QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526`, arXiv:2608.30320 and `Qwen/Qwen3.8-Flash-Next@de4b8e4...`. | Use them as the primary sources for contracts. |
| E-021 | published | The official checkpoint uses the Qwen Community License 1.0 and includes additional conditions for certain commercial MaaS/AI Work Assistant uses. | `Qwen/Qwen3.8-Flash-Next/LICENSE`. | Keep the license, distribution and use-case decision open. |
| E-022 | published | On Strix Halo, GTT is a GPUVM mapping limit over shared RAM, not additional VRAM; AMD recommends a small BIOS carve-out and raising TTM/GTT. | Official ROCm 7.2 Strix Halo optimization documentation. | Separate the mapping limit from admission and reserve effective host memory. |
| E-023 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-024 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-025 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-026 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-027 | withdrawn | External comparative record; kept outside the repository. | — | — |
| E-028 | observed + inferred | The pinned official config declares 48 layers, 12 of them full attention, 2 KV heads and `head_dim=256`; it follows that a conventional BF16 KV cache requires 24 KiB/token/session: 6 GiB at 262,144 tokens. | `artifacts/qwen-official-config-de4b8e4.json`, SHA-256 `889658F2508E8C61D409B02E70E0D78D8D4452EC65AAAFBE129805D213D2E74B`; formula in `docs/operations/memory-budget.md`. | C4 at maximum context would need 24 GiB of base KV alone; qualify smaller contexts first. |
| E-029 | pending | The offline `gfx1151` inventory of rocBLAS/hipBLASLt could not be repeated in this session because the Docker Desktop Linux backend did not expose its API, although the client and the backend process exist. | `docker info` against `desktop-linux`, 2026-09-28. | P-011 remains open; do not infer library coverage from the fact that the HIP probe compiles. |
| E-030 | observed | The engine's GGUF reader parses little-endian v3 and rejects invalid bounds, booleans, types, blocks, alignment, ranges and overlaps; ASan+UBSan detect no faults. | CTest in the `build-wsl` and `build-sanitize` builds, A0.2/A0.3 code and pinned GGUF/GGML format reference geometries (see `docs/reference/sources-ledger.md`). | Metadata/directory inspection and physical range verification are now possible without loading weights. |

| E-031 | published + observed | The pinned revision of the public `qwen4_exp` model implementation explicitly defines centered RMSNorm, partial RoPE, the recurrent Gated DeltaNet update and PLE/n-gram hashing; independent oracles and small fixtures with fresh/resumed replay have been created. | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`; `tools/oracles/`. | A0.5b is partially implemented; sampling/MTP and full QSA semantics remain blocked. |


| E-032 | published | Current public GGUF files of Qwen3.8-Flash-Next are approximately: Q8_0 188.26 GB, Q6_K 168.11 GB, Q5_K_M 134.67 GB, Q5_K_S 128.10 GB, Q4_K_M 119.60 GB, Q4_K_S 113.14 GB, Q4_0 100.60 GB and IQ4_XS 97.68 GB. | Current public Qwen3.8-Flash-Next GGUF catalog, captured 2026-09-29. | A single-node 128 GiB profile cannot select a quantization by nominal quality alone; it must leave physical RAM for the reserve, runtime, KV, recurrent state and scratch. |
| E-033 | inferred | With an effective reserve of 16–20 GiB and the rest of the C1 state, an initial target of roughly 100 GiB or less of resident weights leaves reasonable headroom on a 128 GiB machine; Q4_K_M does not meet that headroom and Q4_K_S is very close to the limit before A0.6. | E-028, E-032 and `docs/operations/memory-budget.md`. | Fix a conservative single-node envelope; P-002 remains pending quality and real measurements. |
| E-034 | published | The current public Qwen3.8-Flash-Next config declares `hidden_size=2560`, 24 attention heads, `head_dim=256`, `partial_rotary_factor=0.25`, a vocabulary of 248,320, 512 experts and top-10 per token. | Public Qwen3.8-Flash-Next config, captured 2026-09-29. | Use these geometries as representative shapes in the A1.0/A1.1 microbenchmarks, without treating them as performance results. |

| E-035 | published + observed | The pinned GGML format revision defines IQ4_XS as 256-element, 136-byte blocks: a global FP16 scale, 8 local 6-bit scales and 128 bytes of 4-bit codes indexing a 16-value non-linear table. The core now has a byte contract and an independent fixture covering positive, negative and zero scales. | `ggml-org/llama.cpp@8019dc563b1ecbae6b161a70c3a1359f1b206c1e` (format reference); `include/strix/iq4_xs.hpp`; `src/iq4_xs.cpp`; `tests/iq4_xs_tests.cpp`. | IQ4_XS can move to an experimental A1.1 kernel without closing P-002 yet. |

| E-036 | published + observed | The pinned Transformers revision defines assisted greedy verification as accepting the contiguous candidate prefix equal to the target `argmax`, followed by one bonus token from the target; after EOS/length it recomputes `n_matches = valid_tokens - 1` and trims `candidate_length - n_matches` cache positions. The same revision defines sampled speculation with the `p/q` ratio and the `[p-q]+` fallback, but its exact replay depends on the RNG. | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`, `generation/utils.py`, `generation/candidate_generator.py`; `include/strix/speculation.hpp`; `tests/speculation_tests.cpp`. | Close greedy MTP acceptance; keep sampled mode blocked until the RNG stream/consumption is fixed. |

| E-037 | published + observed | The Qwen MTP head is a one-layer head with an embedding+hidden combiner, an attention+MoE block wrapped in hyper-connections, and a recurrent hidden state carried between draft steps. Source: an external reference implementation (recorded in `docs/reference/sources-ledger.md`); it is not adopted as a semantic oracle. The engine uses it only to derive structural tensor families for auditing and keeps MTP execution blocked. | External reference implementation (see `docs/reference/sources-ledger.md`); `include/strix/mtp_draft_contract.hpp`; `include/strix/mtp_draft_tensor_audit.hpp`. | Keep structure and semantics separate; require an independent contract before implementing the MTP forward pass. |

| E-038 | published + observed | MTP input fusion with hyper-connections: the normalized embedding is projected once and shared across streams; the normalized wide hidden state is interpreted as `hc_count` streams of `hidden_size`, the same `fc_hidden` projection is applied to each stream, and the embedding projection is added. Two external reference implementations (recorded in `docs/reference/sources-ledger.md`) agree on these semantics. An independent CPU oracle was added for this sub-operation. This does NOT resolve attention/QSA or the full MTP forward pass. | External reference implementations (see `docs/reference/sources-ledger.md`); `include/strix/mtp_input_fusion.hpp`; `tests/mtp_input_fusion_tests.cpp`. | Mark only input fusion as semantically resolved; keep construction of full MTP execution blocked. |

| E-039 | published + observed | The state that feeds the next MTP step is the wide hyper-connection residual after the MTP block and before the final output mixer. For the pinned config this is 4 streams × 2560 = 10240 values per row/request. Two external reference implementations (recorded in `docs/reference/sources-ledger.md`) agree on this. A host-side contract was added that rejects using the already-collapsed hidden state as the next-step state. | External reference implementations (see `docs/reference/sources-ledger.md`); `include/strix/mtp_draft_state_contract.hpp`; `tests/mtp_draft_state_contract_tests.cpp`. | Mark draft-state propagation as a resolved sub-contract; keep QSA/output and the full forward pass blocked. |

| E-040 | observed + project fixture | The QSA oracle in `tools/oracles/qwen4_exp_oracle.py` fixes, for tie-free cases: averaging of keys per compressed block, RMSNorm of the pooled key, RoPE applied to the compressed key at the block's starting position, score as the sum over heads of `max(0, q·k)` scaled by `1/sqrt(head_dim)`, top-k block selection and preservation of the partial tail. The oracle itself rejects any tie that could affect the cutoff. | `tools/oracles/qwen4_exp_oracle.py`; `include/strix/qsa_selection_plan.hpp`; `tests/qsa_selection_plan_tests.cpp`. | Mark tie-free QSA selection as resolved; keep the tie-breaking contract open and, therefore, the full MTP attention gate. |

| E-041 | published + observed | The pinned canonical Transformers forward pass confirms the QSA pipeline: indexer Q/K projection, RMSNorm of Q, raw K cache, FP32 averaging over complete groups, RMSNorm of the pooled key, RoPE at the block's starting position, `relu(q·k)` score summed over heads and scaled by `1/sqrt(head_dim)`, top-k block selection and an uncompressed partial tail. `torch.topk` tie-breaking is not an explicit portable contract. | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae:src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`; `tools/oracles/qwen4_exp_oracle.py`. | Keep the no-tie path resolved, keep the cutoff tie open and do not yet enable the full MTP forward pass. |
| E-042 | published + observed | The MTP head applies a final hyper-connection mixer before the logits; the repository has an independent oracle for the gated-residual formula. An MTP head may carry an explicit `shared_head`; when it is absent, the target output head is a possible fallback. This was cross-checked against two external reference implementations (recorded in `docs/reference/sources-ledger.md`). The engine's plan resolves the binding only when the manifest contains exactly one `shared_head`. | External reference implementations (see `docs/reference/sources-ledger.md`); `include/strix/mtp_output_contract.hpp`. | Separate the resolved mixer from the head binding; do not assume a target fallback without an explicit binding. |

| E-043 | published + reproducible decision | The canonical QSA forward pass uses `torch.topk`, and PyTorch does not guarantee index stability for tied elements. Upstream therefore defines no portable order at the cutoff. The engine fixes descending score with the lower block index winning on equal scores, using the same rule for serial and MTP; the `qsa_tie` fixture covers the case. | PyTorch `torch.topk` docs; D-013; `qsa_select_tokens_deterministic`; `tools/oracles/fixtures/qwen4_exp_small.json`. | Close the tie-breaking gate as an explicit engine contract, without attributing that rule to Qwen/PyTorch. |

| E-044 | published | A qualified USB4/NHI transport on Strix Halo can import GPU pools exported as DMA-BUF and transfer payloads without going through the TCP/IP path or a CPU read/write copy. The reference keeps dedicated TX/RX, dedicated framing and two-sided participation of both peers. | External transport reference (recorded in `docs/reference/sources-ledger.md`): architecture documentation and DMA-BUF patches for `kernel/linux-7.2.2/`. | Add an experimental GPU-owned/DMA-BUF transport tier behind the engine's transport abstraction; do not assume availability just because USB4STREAM is detected. |
| E-045 | published | DMA arrival is not enough to reuse RX: the qualified path waits for GPU consumption/validation, completion accounting and frame repost. | External transport reference (recorded in `docs/reference/sources-ledger.md`): architecture documentation. | Extend ownership to consumer-release -> repost-pending -> reusable and measure the completed GPU operation, not only DMA arrival. |
| E-046 | published | The GPU-visible protocol uses terminal metadata with epoch/sequence/payload and a commit published last with release/acquire semantics; it also treats the terminal region as an ownership boundary, so global visibility is not inferred merely from observing the last payload write. | External transport reference (recorded in `docs/reference/sources-ledger.md`): architecture documentation. | Add an engine commit record with epoch/sequence/stripe and require validation before publishing a received tensor. |
| E-047 | published | Direct-stream setup is transactional and depends on a healthy USB4NET control path, reconciliation of both peers, an exclusive lease, primed RX and exact cleanup; a unilaterally prepared endpoint is not equivalent to a ready pair. Ring reuse can also fail because of pending MSI-X state if the stack does not clean up/re-arm correctly. | External transport reference (recorded in `docs/reference/sources-ledger.md`): transport documentation and stale MSI-X patch. | Model lifecycle/lease per physical link and add an explicit stop/cleanup/re-arm/first-completion gate. |
| E-048 | withdrawn | External comparative record; kept outside the repository. | — | — |

| E-049 | published | AMD specifies 2 native 40 Gbps USB4 ports on Ryzen AI Max+ 395/Strix Halo. The product specification does not demonstrate router/upstream independence or a sustained 80 Gbps aggregate. | AMD Ryzen AI Max+ 395 product specification, consulted 2026-10-01. | Design the two physical links as separate resources and require a simultaneous test before enabling striping/aggregation. |

| E-050 | observed (host) | Experimental wave top-k: 6,864 independent insertion/wave bit cases plus slot/admission mutants; WSL GNU 13.3 Release HIP-OFF build and 152/152 CTest cases passed. | Project source `src/hip_moe_route_launcher.hip`, `tests/router_wave_topk_tests.cpp`, implementation document; `cmake --build build -j 8`; `ctest --test-dir build --output-on-failure --no-tests=error -j 8`. | Supports portable order/admission contracts only. Actual revised-source HIP compile, ISA/FP-mode and old-kernel preservation, device fixtures, 554 arrays/512 IDs and paired timing remain independent gates. OFF by default. |
| E-051 | observed | The `--deferred-moe-status` mode has a distinct MoE identity, retains the domain/bits of the first error and finalizes within the session's publication boundary. The host tests inject errors at layer 17 and at the fence, and check that host history stays intact, generations/position/ID do not advance and the session is poisoned. | GNU 13.3, WSL, CMake Release, HIP OFF: 152/152 CTest; `tests/deferred_moe_status_tests.cpp`; `tests/compiled_execution_tests.cpp`; contract in `docs/engine/decode/deferred-moe-status.md`. | Does not imply GPU execution, model array equality or a performance gain. HIP compilation, ISA, native fixtures and model/performance qualification are pending. |

## Rule for new entries

Each new evidence entry must include the artifact identity, the command or test,
the environment conditions and the spec requirement it affects. If a measurement
contradicts an earlier entry, both are kept and the resolution is recorded in
`docs/decisions/decision-register.md`.


## E-052 (2026-10-15): Q8 indexed-attention input prefetch source increment

State: experimental source and host/device fixture implementation; native gates
pending. Artifact: `docs/research/history/implementation-2026-10-15-qsa-input-prefetch.md` and
`tests/qsa_input_prefetch_tests.cpp`, `tests/hip_qsa_input_prefetch_tests.hip`.
Commands: configure Release with `BUILD_TESTING=ON`; host target
`strix-qsa-input-prefetch-tests`, optional gfx1151 HIP target
`strix-hip-qsa-input-prefetch-tests`. The baseline project contains POSIX-only
sources, so a full Windows build is not a portable qualification environment.
Actual Linux HIP compiler/ISA/device/model/performance results are not claimed.

The follow-up source adds exact-message negative assertions, actual mixed catalog
and full-operator ON/OFF eager/graph fixtures, poisoned output/workspace/KV/raw
owners, five-candidate and four-candidate node assertions, and zero-node failure
checks. A host submission counter records only eager enqueue and graph capture;
it does not count graph replay. The host plan test explicitly covers Q8 q/k/v
with BF16 indexer inputs (36 eligible and 24 fallback projections at 12 steps).
These are fixture contracts, not claims that new device tests have executed.


The result-publication follow-up separates the live QSA dispatch count from the
static mode prefix. The raw-token JSON prefix is written before its token loop;
its dynamic count is now read only after that loop. Restore and greedy results
also read the same live owner when publishing their completed results. The
host regression demonstrates a pre-execution zero snapshot and a fresh19944
snapshot after554 synthetic submissions of36 eligible projections; these are
host fixture inputs, not a device execution result. Disabled mode emits no count.
Catalog fixtures additionally inspect each selected launcher's actual user-data
configuration and require retention of the supplied shared stats owner after the
caller's configuration releases it, for both selected and fallback bindings.

Local verification: `strix-qsa-input-prefetch-tests` passes in the existing
Windows portability harness; the catalog test compiles with MSVC C++20/W4/WX.
That catalog check is compilation only. Native execution and full-trajectory
qualification of this publication change remain pending. No kernel arithmetic,
manifest eligibility or expected submission-count policy changes.
## E-053

Experimental source and fixtures: `src/lane_chain_hyper_output.cpp`, `src/hip_hyper_mix_launcher.hip`, host `strix-lane-chain-hyper-output-tests` and HIP `strix-hip-lane-chain-hyper-output-tests` / `strix-hip-lane-chain-hyper-integration-tests`. Reproduce host checks with `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_WARNINGS_AS_ERRORS=ON`, build and `ctest --test-dir build --output-on-failure`. Host execution cannot qualify device arithmetic. HIP gfx1151 compiler/ISA/device, full-model exactness and performance gates remain required; see the lane-chain implementation document. No measured gain is claimed.
E-051 follow-up artifacts: `merge_deferred_moe_first_error` is now the same
host/device helper called by the actual publisher and the portable dispatcher
fixture. Host sequence zero -> A8 bits 5 -> legacy bits 1 -> zero must retain
`{2,5}`. Exact-message compiled admission and host-only launch-scope metadata
fixtures exercise their specific guards. Native scope/alias fixtures require
untouched poison and zero captured nodes where an owned scope exists. A bounded
host callback holds the fixture stream to test asynchronous launcher return;
this fixture has not been executed as part of the portable evidence above.
Actual owner receipts report completed tokens, without synthetic layer counters.


E-051 follow-up observed checks: GNU 13.3/WSL, Release HIP OFF, rebuilt targeted
`strix-deferred-moe-status-tests` and `strix-compiled-execution-tests`: 2/2 PASS.
The actual `hip_launch_context_tests.cpp` metadata fixture also passes as a host
executable linked to the real core with private HIP type/constant declarations;
it invokes no HIP API and provides no runtime qualification. Native fixture C++
API parsing succeeds both with and without experimental A8 capability enabled.
E-051 native/compiler/ISA/model/performance gates: **pending**.
## E-054 (2026-10-15): shared-expert Q8 projection loading source increment

Artifacts: `docs/research/history/implementation-2026-10-15-shared-expert-prefetch.md`,
`tests/shared_expert_prefetch_tests.cpp`,
`tests/hip_shared_expert_prefetch_tests.hip`, and full-operator/catalog fixtures.
Commands: build and run `strix-shared-expert-prefetch-tests`; optional native
`strix-hip-shared-expert-prefetch-tests`, existing shared launcher and catalog
targets. Host planning tests pass in a private external MSVC portability harness
excluding the baseline POSIX stream TU and disabling warnings-as-errors because
of pre-existing unreachable-code warnings. This is a limited host gate. C++ API
syntax checks are not HIP compilation. Actual linked legacy/new ISA, device tests,
complete model arrays/tokens and workload performance remain pending. Requirements:
exact gate/up raw FP32 and down scalar-gate/store behavior, explicit fallback and
identity admission, poisoned eager/graph fixtures and host dispatch counters.


E-054 follow-up source: catalog fixtures now provide a real statistics owner and
check the final native launcher configuration retains the same owner; the catalog
constructor forwards it explicitly. Tensor-name JSON escaping has portable
control/quote/backslash/null fixtures. New device-only payload cases compare
legacy, virtual and candidate outputs for distinct NaNs/infinities and half NaN/
infinite scales, without treating host FMA as a payload oracle. These new native
cases still require actual execution and linked-ISA qualification.

E-054 payload follow-up (pending native execution): a separate finite-scale,
nonzero-code fixture uses feature-dependent quiet NaNs whose distinct payloads
survive BF16 truncation. Host checks distinguish every reduction-stride pair;
finite output poison avoids legitimate NaN payload collisions. The mixed
nonfinite case also tests a distinct NaN scalar gate during graph replays.
## E-055

Diagnostic source: `src/hip_hyper_mix_launcher.hip`, `tests/hip_lane_chain_hyper_projection_tests.hip`, and its implementation document. HIP target `strix-hip-lane-chain-hyper-projection-tests` runs synthetic exactness checks; `--bench` records real-payload event pairs using an externally bound normalized capture. Required environment is actual gfx1151 with the original TU compiler settings; compiler/ISA and device results are pending. Source staging sizes are not measured occupancy or throughput.

## Historical automatic decoder policy preparation (2026-10-15)

The actual application parser and resolver are exercised by
`tests/decode_optimization_profile_tests.cpp` with A8 capability present and absent.
The preceding host-policy checkpoint passed 159/159 strict Release portable CTest
entries. The preceding integrated source also passed 225/225 actual native tests
with no failures/skips; its linked audit preserved 225 legacy kernels. These are
native regression and preservation results, not a joint model/performance gate.

E-050 wave routing and E-053 lane-chain HC output each have separate canonical
full554 array equality,512 generated-ID identity and descriptive paired workload
observations. This limited prompt/context/host evidence does not add together or
qualify the combined automatic profile. No universal checkpoint-quality claim is
made. Detailed receipts remain in the private evidence store.

The current host-only follow-up preserves explicit legacy options unless auto is
explicitly requested, checks prefix capacity after resolving context, and
uses the existing JSON escaping helper for QSA tensor names. Its tests cover
no-flag automatic selection, implicit explicit-option fallback, explicit auto plus
ranked, manual JSON identity, a 200-token prefix accepted under automatic context
8192 versus rejected under manual context 128, and 43-prefix/511-generation requests
accepted under manual, explicit 128 and unknown-checkpoint configurations.
Generation retains the existing `context_capacity` stop. Changed host source needs
review and fresh native validation;
joint automatic full554/512 fidelity and paired timing remain independent gates.

Observed for this host follow-up: GNU 13.3/WSL Release with HIP OFF and warnings
as errors passed the full 159/159 portable CTest entries. Decoder/catalog/catalog
fixture C++20 syntax checks with private HIP declarations passed with the A8
capability both present and absent; this is not a fresh HIP/native result.

Auto-v3 policy preparation: code revision
`dee811a391485b894dbda5ea1d7fd29efc1bd1ec` passed the full 159/159 portable tests
under GNU 13.3/WSL Release, HIP OFF, with warnings as errors. This observation
covers host policy and compatibility only. Documentation-only follow-ups preserve
that tested code; native, full-model fidelity and clean automatic workload gates
remain separate. Manual bundle observations do not by themselves qualify v3 auto.

## E-056 (2026-10-15): canonical automatic v3 qualified and adopted

Exact source `265fd78762884d18eb32aa60294408f080088d52`, archive SHA-256
`f0c87566c37fa5ca2e76caacd02c5af94a0080bb3b08911d377e7058a90db155`,
passed native 225/225 with no failures/skips, 14-module diagnostic linkage,
245-kernel compiled preservation and seven actual CLI cases. The complete 554
FP32 logit arrays match the preserved canonical reference. All eight clean
automatic runs match all 512 immutable generated IDs. These close the preceding
automatic-policy pending gates only for the admitted canonical content, eager
registered execution and context 8192. They do not assert universal checkpoint
quality, all hidden-state equality or other execution-mode qualification.

Two separate same-node campaigns each retained two warmups and three alternating
pairs. Against the immutable baseline, the median paired ratio was 1.2599223
(+25.99%), with wall-rate medians 12.1007 and 15.2520 tokens/s. Against the preceding
automatic wave+HC bundle, both arms used clean automatic selection and the
median ratio was 1.1555118 (+15.55%), with wall-rate medians 13.1692 and 15.2226.
The latter used an explicitly foreign-compiled baseline revalidated on the actual
execution host. Both comparisons matched every 512-token vector, metadata and
actual HIP/HSA mappings; owners/processes were reaped, shared locks released and
input/source/model pins unchanged. All outliers remain included. The prescribed
3/6/10 descriptive screen stopped at 3; no significance, universal performance or
additive-gain claim follows. Host timing instrumentation is explicit and retained
the same14 device modules as each clean application.

The actual full554 receipts report 19,944 QSA and 79,776 shared projection host
submissions plus 554 deferred-owner completions. These are not invented layer or
replay counters. The primary source now equals that exact qualified revision;
this observation does not assert deployment of a remote user binary. Detailed
receipts and original comparative inputs remain outside Git. See the
[profile adoption record](../engine/decode/optimization-profile.md#qualified-adoption-2026-10-15)
for the model identity, fallback policy, measurement scope and remaining gates.
