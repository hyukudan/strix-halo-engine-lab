# Estado de implementación

## Slice A0.1 — núcleo portable

Implementado:

- proyecto CMake C++20 sin dependencias externas;
- contrato de shapes Qwen3.8-Flash-Next fijado desde el config oficial;
- planificador de memoria con aritmética comprobada contra overflow;
- desglose KV base, MTP, indexer provisional, GDN FP32, rollback, modelo y cachés;
- decisión de admisión desde MemTotal, baseline y reserva explícita;
- CLI JSON `strix-memory-plan`, preparada para alimentar la futura WebUI;
- oráculo top-k candidato con empates deterministas y rechazo de NaN;
- pruebas CTest en Linux/WSL con warnings-as-errors.

No implementado o deliberadamente bloqueado:

- semántica QSA canónica, n-gram/PLE, hyper-connections y GDN: esperan contratos
  oficiales/oráculos antes de crear kernels;
- allocator/residency HIP y kernels `gfx1151`;
- MTP sampled;
- servidor y panel administrativo.

## Reproducción

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
./build/strix-memory-plan --tokens 32768 --sessions 1 \
  --resident-gib 68 --baseline-gib 2 --reserve-gib 18
./build/strix-platform-probe
./build/strix-model-manifest MODEL.gguf
```

El indexer es una estimación provisional derivada del config. Se puede excluir
con `--no-indexer`. `--rollback-full` permite cuantificar por qué copiar el
estado GDN completo por borrador no escala; el camino objetivo será snapshot
mínimo más recompute si el contrato lo permite.

## Slice A0.2 — inspección GGUF

Implementado en `include/strix/gguf_reader.hpp` y `src/gguf_reader.cpp`: lector
GGUF v3 little-endian defensivo y CLI `strix-inspect-gguf`. Valida metadata,
arrays, directorio tensorial, geometría de bloques, tamaños, alineación, rangos
y solapamientos sin cargar los datos de pesos.
Las limitaciones y la revisión canónica se documentan en `GGUF_READER.md`.

Validación actual:

- build `RelWithDebInfo` con `-Werror`;
- fixtures válidos y corruptos en CTest;
- build Debug con AddressSanitizer y UndefinedBehaviorSanitizer;
- 100 % de las pruebas actuales superadas en ambos builds.


## Slice A0.4 — model manifest

Implemented:

- logical tensor manifest derived from validated GGUF metadata;
- conservative tensor-role classification;
- per-tensor physical bytes and offsets;
- layer extraction for common block naming;
- explicit residency policy;
- explicit placement with local node zero as the default;
- n-gram/sparse-table storage can be marked as a file-backed candidate without changing admission semantics for ordinary weights;
- JSON CLI `strix-model-manifest`;
- dedicated unit tests.

The manifest is intentionally conservative. Unknown tensors remain visible and resident by default. Complete architecture-specific tensor-name validation remains a later gate once the canonical development checkpoint is fixed.

## Slice A0.5 — platform capability probe

Implemented:

- dependency-free Linux host probe;
- physical RAM from `/proc/meminfo`;
- KFD GPU topology and `gfx_target_version`;
- derivation of `gfx1151` and compute-unit count when KFD exposes the required fields;
- ROCm version-file and `hipcc` presence checks;
- Thunderbolt/USB4 domains and host-router discovery from sysfs;
- USB4STREAM driver and `/dev/tbstream*` discovery;
- JSON CLI `strix-platform-probe`;
- fake-sysfs unit test for deterministic host-independent validation.

Not yet qualified on the target hardware:

- real port-to-router mapping;
- negotiated USB4 link rate;
- simultaneous dual-link independence;
- HIP runtime/device capabilities;
- GTT/TTM effective mapping limits;
- graph and library capability inventory.

Those require native Linux measurements on the Strix Halo hosts.


## Slice A0.5b — semantic and numerical oracle

Partially implemented in `tools/oracles/`:

- dependency-free Python reference code, separate from runtime inference;
- centred and grouped RMSNorm;
- partial rotary application;
- sequential Gated DeltaNet recurrent update in double precision;
- PLE/n-gram hashed row IDs with EOS-aware history;
- deterministic candidate top-k policy;
- tiny frozen fixtures;
- fresh-versus-resumed equivalence tests for recurrent and n-gram state;
- optional CTest integration when Python 3 is available.

Still open:

- canonical sampled RNG/replay semantics;
- MTP acceptance algorithm;
- complete QSA selection and tie fixtures;
- Gated Residual/Hyper-Connection checkpoint fixtures;
- tokenizer/chat-template golden tokens;
- first-layer/logit replay from the canonical development checkpoint.


## Slice A0.7 — transport core

Initial host-independent core implemented:

- explicit producer/transport/consumer buffer ownership state machine;
- abstract asynchronous transport-backend interface with prepare/submit/poll/cancel;
- distinct local-reuse, receive-complete and remote-transport-ack completion kinds; semantic commit remains above transport;
- transfer descriptors bound to epoch, model-plan, request, generation, execution boundary, shape, dtype and byte length;
- bounded transfer descriptors with protocol version and connection epoch;
- qualified-link profiles with latency, payload bandwidth and in-flight accounting;
- deterministic single-link cost estimation;
- flow scheduling that reacts to queued bytes;
- dual-link striping only when both links are explicitly qualified as independent and the estimated complete transfer beats the single-link path;
- unit tests for ownership, stale/oversized descriptors, single-link selection, queue-aware selection and striping;
- standalone TCP ping-pong and stream benchmark harness;
- direct USB4STREAM read/write benchmark harness;
- dual-link parallel-flow runner with configurable payload split and aggregate JSON metrics;
- read-only capture of configured stream index, busy-poll, ring-size, throttling and HopIDs;
- framed payload validation, optional checksums, warmup exclusion and JSON latency/throughput output;
- host-side CI workflow covering the core, oracles, UMA host lab and TCP smoke tests.

Still pending on the target hosts:

- real two-host TCP/network baseline measurements;
- real two-host direct USB4STREAM measurements;
- interrupt/event versus polling progress modes;
- mapped/registered transport-buffer tiers;
- real single-link and dual-link topology qualification.


## Slice A1.0 — kernel registry and basic gfx1151 kernels

Host-side registry implemented:

- explicit operator family and execution phase;
- exact logical-row, request-concurrency and geometry shape classes;
- independent Wave32/Wave64 identity;
- workgroup, LDS/register/scratch metadata fields;
- numerical contract with row-invariance and deterministic flags;
- graph-safety and qualification state;
- compiler identity and measured cost;
- deterministic selector;
- stable plan identity that deliberately excludes measured-time noise;
- qualified/production variants require a non-zero measured cost;
- CTest coverage for validation, selection, wave filtering and plan identity.

Standalone `lab/basic_kernels/` implemented for native ROCm:

- `gfx1151`-only build;
- native Wave32 candidate;
- explicit `-mwavefrontsize64` Wave64 candidate;
- device-side `warpSize` verification;
- FP32 centred RMSNorm;
- FP32 residual add;
- FP32 partial RoPE;
- deterministic argmax;
- deterministic top-k baseline up to k=16;
- solo-versus-batched row-invariance fixture;
- model-representative microbenchmarks for hidden size 2560, 24x256 partial RoPE, vocab 248320 and expert top-10 of 512;
- HIP-event timings and `hipFuncGetAttributes` resource reporting.

Still pending:

- native M5/Strix Halo build and execution;
- Wave32 versus Wave64 measurements;
- profiler-derived VGPR/LDS/spill confirmation;
- graph-capture qualification;
- promotion of any kernel from experimental to target-qualified;
- BF16/mixed-precision variants, which require separate numerical identities.


## Slice A1.1 — packed low-row projection

Host-side planning implemented:

- GGML tensor storage geometry is now a public contract shared by the loader and kernel planning;
- explicit low-row classes 1/2/4/8;
- exact block geometry and packed bytes per output row;
- exact tensor weight bytes;
- activation input/output byte accounting;
- serial repeated-weight traffic versus ideal reused-weight lower bound;
- overflow and quant-block divisibility checks;
- CTest coverage including Q4_K, Q8_0 and IQ4_XS geometry;\n- independent IQ4_XS 136-byte block/dequant contract with nonlinear codebook and 6-bit local-scale extraction;\n- host IQ4_XS fixture compiled with strict warnings and passing.

Standalone `lab/packed_projection/` implemented:

- Q8_0 packed reference kernel for `gfx1151`;\n- IQ4_XS compact packed candidate for `gfx1151`;
- packed block is consumed directly inside compute; no full dequantized weight tensor;
- compile-time C1/2/4/8 logical-row variants;
- each decoded packed weight is reused across active logical rows;
- native Wave32 and explicit Wave64 builds;
- device-side wave-size verification;
- CPU packed-format reference and true row-invariance test;
- model-shape benchmark at 2560x2560;
- HIP-event timings;
- physical packed-weight GiB/s and serial-equivalent weight-traffic metric;
- per-variant register/shared/local-memory attributes.

Still pending:

- native Strix Halo build/profile;
- canonical compact checkpoint format selection;
- native compile/profile of the IQ4_XS compact path and comparison against Q8_0;\n- canonical checkpoint-format decision after quality plus full memory/runtime qualification;
- BF16/mixed-precision activation variants;
- profiler spill/LDS/VALU/WMMA analysis;
- promotion into a target-qualified kernel plan.


## Slice A1.2 — recurrent fast path

Initial implementation:

- host-side recurrent window planner with exact persistent-state byte accounting;
- rollback planner for full candidate snapshots, pre-window replay and sparse checkpoints;
- retained bytes, maximum replay, expected replay and worst-case replay-state-I/O metrics;
- standalone `gfx1151` HIP GDN state-update lab;
- 128-lane key-axis decomposition and 32-column value tiles;
- compile-time retained windows for 1/2/4/8 logical tokens;
- serial one-token-launch baseline versus retained-window path;
- CPU-reference comparison for final state and per-token outputs;
- kernel timing and HIP resource metadata output.

Still pending:

- native Strix Halo performance/resource qualification;
- spill inspection for each token-window size;
- Wave32/Wave64 alternatives where useful;
- surrounding convolution/projection integration;
- long-window fresh/resumed qualification;
- end-to-end speculative replay measurements.


## Slice A1.3 — MoE routing and expert execution

Routing foundation implemented:

- host-side route planner with per-expert assignment histogram;
- distinct-expert union and adjacent-row route-overlap metrics;
- direct-assignment versus ideal grouped packed-weight traffic accounting;
- per-projection expert storage compatibility planner;
- real target geometry pinned at hidden=2560, intermediate=640, experts=512, top-k=10;
- grouped expert batches decomposed into realizable 8/4/2/1 row classes without padding;
- packed IQ4_XS MoE reuse harness for compatible synthetic gate/up and down geometry;
- deterministic CSR-style expert grouping plan;
- route planning now emits a fused touched-expert bitmap plus CSR grouping offsets and assignment row/slot metadata in one validated pass;
- grouped assignments retain original row and top-k slot identity for exact output restoration;
- target-independent direct-versus-grouped execution selector driven by a qualified complete-cost model;
- complete cost includes fixed overhead, per-assignment/per-batch work, actual route-derived packed-weight traffic and grouped row-restoration cost; exact ties stay on the direct path;
- duplicate/out-of-range route validation;
- dependency-free semantic oracle plus portable C++ row selector for FP32 softmax, deterministic lower-expert-index cutoff ties, top-k and selected-weight renormalization;
- Wave32-shaped hierarchical host routing candidate with explicit 32-wide groups, deterministic local/global reduction and exact target-geometry work accounting;
- standalone `gfx1151` HIP routing baseline for 1/2/4/8 logical rows;
- standalone `gfx1151` Wave32 hierarchical HIP candidate with one 512-thread row block, sixteen 32-lane group reductions, deterministic global reduction per top-k slot, cutoff-tie validation and explicit host-plan work counters;
- serial and hierarchical routing binaries share an explicit deterministic cutoff-tie fixture identity and emit comparable iteration/normalization metadata;
- dependency-free native qualification runner interleaves serial/hierarchical C1/C2/C4/C8 repetitions, rejects mismatched device/fixture/`gfx1151` metadata, gates correctness before timing and emits per-row median latency, resource metadata and speedup;
- target-dimension Q8_0 routed-expert candidate using the real 2560-wide hidden input and 640-wide intermediate down input, with direct C1 assignment batches versus deterministic expert-major 8/4/2/1 grouped batches;
- complete-layer MoE policy now incorporates measured shared-expert and combine cost, modeling routed/shared overlap and refusing grouped execution when a routed-only win is hidden by the shared branch;
- a qualified whole-layer schedule materializes routed-expert, shared-expert and combine dependencies from that policy, including explicit parallel or serialized shared-branch execution and a stable runtime-bound schedule identity;
- qualified complete-layer cost policies are sealed to the runtime-compatibility identity and a measurement-campaign identity; foreign runtimes or post-qualification coefficient mutation are rejected before schedule selection;
- fixed-window route-trace aggregation now summarizes real selected-expert traces into distinct-expert histograms, overlap/reuse metrics and realizable direct/grouped packed-weight traffic for C1/C2/C4/C8 qualification;
- packed Q8_0 scalars are decoded once per grouped kernel traversal and reused across active logical rows for gate/up and down projections, with CPU packed-weight reference, exact row restoration, modeled weight traffic and HIP-event timing;
- dependency-free `lab/moe_q8_0/qualify.py` gate for C1/C2/C4/C8 validates fixture/device/geometry parity, modeled byte accounting and numerical error before summarizing direct-versus-grouped medians;
- standalone FP32 expert-execution correctness harness;
- direct assignment and expert-major grouped GPU schedules compared against the same CPU reference;
- grouped schedule preserves original row restoration through explicit assignment metadata;
- routing timing, exact-index agreement, weight error, expert-union and kernel-resource output.

Still pending:

- native Strix Halo build/profile and latency comparison of the hierarchical HIP candidate against the serial-selection baseline;
- native Strix Halo build/profile and C1/C2/C4/C8 qualification of the target-dimension Q8_0 direct/grouped expert candidate;
- shared-expert integration;
- target-native calibration of routed/shared/combine cost coefficients;
- run the new fixed-window route-trace summary on canonical checkpoint traces for C1/C2/C4/C8 and feed the measured distributions into qualification.


MoE format constraint discovered:

- IQ4_XS block geometry (256 elements) is compatible with the 2560-wide routed gate/up input;
- IQ4_XS cannot directly represent the 640-wide real down input under the current pinned GGML row geometry;
- Q8_0 block geometry (32 elements) is compatible with both dimensions;
- the canonical expert checkpoint may therefore require mixed per-projection formats or an explicitly defined lossless repack/padding layout.


## Slice A1.4 — sparse lookup path

Initial host-side implementation:

- sparse lookup planner with requested/unique row accounting;
- deterministic contiguous-run construction;
- bounded coalesced-I/O plan with configurable maximum rows per operation, exact byte offsets/sizes and deterministic inflight-wave accounting;
- portable bounded asynchronous I/O executor that consumes the plan, caps concurrent reads, preserves deterministic result ordering and propagates source failures;
- serial-token PLE prefetch now starts the lookup after deterministic preflight but before the DAG, binds the handle to token/position/backend identity, overlaps I/O with preceding token work, and cancels outstanding lookup on execution failure;
- bounded LRU hit/miss simulation and physical row-read byte model;
- standalone Linux benchmark for mmap, pread, pread+LRU, per-window coalesced pread and bounded asynchronous pread using the production planner/executor contract;
- deterministic PLE-like access pattern with 16 logical head partitions;
- additional random, hot-set and contiguous-run controls;
- materialized-file preparation mode so sparse holes are not mistaken for real storage performance;
- cold-cache hinting and major/minor page-fault telemetry.

Still pending:

- exact production table file/shard layout from the canonical checkpoint;
- real PLE row traces from model execution;
- NVMe and page-cache qualification on the Strix Halo hosts;
- native measurement of the new pre-DAG PLE lookup overlap and CPU gather into registered/staging buffers under simultaneous GPU load;
- direct GPU-visible sparse access experiment;
- execute the mapped-vs-planned qualification/sweep gate on native Strix Halo and select chunk/inflight policy from measured complete-token latency;


## Slice A1.5 — correct serial executor

Implemented through an experimental C1/BF16 HIP path:

- pinned serial-decode dependency DAG and liveness-derived scratch arena;
- exact recurrent/indexed-attention layer pattern, PLE placement and gated-residual boundaries;
- deterministic tensor-requirement/access plan bound into runtime compatibility identity;
- kernel-registry binding plus a complete HIP launcher catalog structural audit;
- independently versioned scratch, persistent session state and model-tensor bindings;
- concrete KV/recurrent/convolution/history state layout and fresh-state initialization;
- file-backed PLE lookup seam plus a planned backend that asynchronously executes bounded/coalesced tensor-relative I/O and restores requested row order into staging;
- mapped model files expose bounded direct reads on their owned descriptor, and the native serial runner can select the planned PLE backend with configurable chunk/inflight limits without changing tensor bindings;
- the serial runner emits per-token/total decode latency plus PLE backend identity, enabling reproducible mapped-vs-planned qualification;
- a portable policy-sweep harness evaluates a grid of chunk/inflight settings against one mapped baseline, rejects incomparable evidence and ranks valid planned policies by complete-token latency;
- eager one-token serial session execution with fail-closed poisoning on partial dispatch failure;
- portable complete-request greedy orchestration with prefix prefill, deterministic lower-token-index argmax ties, EOS termination, new-token budget and context-capacity enforcement;
- the native HIP serial runner exposes `--generate N` (optional `--eos-token`) and drives that complete-request contract directly over the compiled C1 session;
- explicit serial prefix lifecycle metadata binds an exact prefix token sequence to one validated snapshot/runtime identity and only produces a restore plan when prefix, execution, runtime and state-layout identities all still match;
- high-level prefix capture/restore operations now enforce that lifecycle before physical state transfer, preventing foreign or stale prefix metadata from reaching snapshot copy/restore;
- a move-only serial prefix cache entry owns exact prefix metadata plus compact snapshot memory, captures directly from a ready prefilled session and restores only into identity-compatible sessions;
- a HIP snapshot-transfer backend provides the physical device-to-device and host-to-host copy path needed by prefix cache capture/restore on native GPU sessions;
- the HIP runner exposes a prefix-cache round-trip qualification mode that captures a prefilled prefix, mutates the live session through a full greedy continuation, restores the compact snapshot and requires the restored continuation to reproduce the same token sequence/stop result;
- raw-token HIP runner that assembles GGUF manifest, selective residency, runtime memory, tensor access, dispatch and FP32 LM-head logits;
- versioned independent multi-step replay fixture with checkpoint/vocabulary preflight;
- replay qualification rejects non-finite or mis-sized logits, requires exact argmax when declared, enforces the requested logit-error policy and now binds evidence to the exact checkpoint, compiled execution and runtime compatibility identities being qualified;
- dedicated host tests cover replay policy and identity failures.

Still pending:

- native Strix Halo build/run of the complete HIP path;
- execute the complete-request and prefix-cache round-trip gates on native Strix Halo against the canonical checkpoint/tokenizer contract;
- asynchronous production PLE lookup/staging rather than the current file-backed seam;
- promotion from experimental bindings only after native correctness and resource qualification.


## Slice A1.6 — greedy speculative execution

Initial semantic foundation implemented:

- pinned greedy assisted-verification contract;
- contiguous candidate/target matching to first mismatch;
- mandatory target-produced bonus token;
- all-candidates-match bonus behavior;
- EOS-inclusive truncation;
- max-length/token-budget truncation;
- accepted-match recomputation after truncation;
- exact speculative target-state tail crop count;
- zero-match cycle reduces to one ordinary target token;
- qualified depth controller with mandatory serial baseline;
- complete proposal + verify + commit/replay + uncovered communication + control cost accounting;
- expected committed-token normalization;
- hysteresis before switching between serial and speculative depths.

Additional A1.6 runtime work now implemented:

- compiled multi-row target verify plans and eager execution;
- pre-window persistent-state snapshots for positive candidate depth;
- exact restore plus serial replay of the accepted prefix on partial matches;
- all-match cycles retain the already-verified target state without replay;
- verify/replay failures recover from the pre-window snapshot when recovery succeeds;
- snapshot restore can recover a poisoned serial session and republishes only after all state ranges restore successfully;
- compiled greedy backend binds serial and verify executions to the same model content and persistent target-state allocation;
- direct launch dispatch rejects foreign execution identities;
- draft candidates and collected target tokens are rejected when outside the compiled/model vocabulary before they can become visible output;
- bounded draft-proposal backend seam with target-session read-only access;
- depth-zero execution bypasses the drafter and compiled verify DAG entirely;
- over-depth draft results fail before target verification or target-state mutation;
- runtime-owned draft lifecycle metadata with synchronized/active/reconciliation-required/poisoned phases;
- proposal-state identity is bound to target runtime compatibility and a precise synchronization position;
- proposal/reconciliation failures poison only the drafter state while depth zero remains available;
- MTP draft binding selects only manifest tensors classified as `mtp`, accounts their exact bytes and binds both model layout and content identity;
- MTP compatibility identity excludes physical relocation, while a separate physical binding identity includes storage, addresses and allocation generations;
- MTP draft state cannot cross checkpoint-content identity;
- pinned MTP structural contract from the official config: one full-attention predictor layer, shared embeddings, null explicit hidden-source layer, base-model attention geometry and a planning-only 2 KiB/token KV estimate;
- MTP semantic gates are tracked independently: input fusion and next-step draft-state propagation are resolved; QSA scoring/selection is resolved including pooled-key RoPE and an explicit deterministic lower-block-index tie rule; output mixer semantics are resolved and an explicit sidecar `shared_head` resolves the LM-head binding; the complete forward gate remains closed;
- executable MTP construction is explicitly blocked while the complete `forward_semantics_resolved` gate remains false;
- `CompiledMtpDraftPlan` now freezes the structural step order and stable relocation-independent plan identity while surfacing unresolved readiness blockers instead of launching;
- offline `strix-mtp-manifest-audit MTP.gguf` reports required structural tensor families, unresolved MTP tensors and missing families without allocating GPU state;
- host-side MTP input-fusion oracle pins per-stream semantics: normalized embedding is projected once and broadcast, normalized wide hidden is reshaped into `hc_count` streams and projected independently before summation;
- evidence-gated MTP plan qualification separates structural readiness from executable qualification: all structural blockers except the forward gate must be closed, then an independent multi-step replay must satisfy the declared token/hidden/logit policy before the plan is executable;
- replay evidence is bound to proposal identity, compiled plan identity, policy identity and independent-reference identity; evidence from another plan cannot qualify the current plan;
- next-step MTP draft state is pinned as the post-block wide residual before final head collapse, with width `hc_count * hidden_size = 10240` for the target model;
- concrete move-only MTP draft-state memory ownership now allocates and validates a proposal-identity-bound device buffer with explicit row count, element size, alignment and allocation generation;
- a composed MTP forward contract validates cross-stage geometry and checkpoint/tensor/output bindings across input fusion, attention, wide draft state and final output, producing a relocation-stable composition identity while leaving numerical execution gated on replay/kernels;
- the qualified MTP draft wrapper now requires that physical state binding and passes it to the forward/reconciliation backend, preventing model-backed proposal compute from running against absent or foreign draft-state storage;
- memory admission accounts that draft state separately per session; the element size is an explicit planning input (2 bytes by default), not a resolved MTP dtype claim;
- deterministic bounded prompt-lookup drafter using the most recent non-overlapping suffix n-gram match;
- prompt-lookup history has an explicit token-capacity bound and proposal identity;
- draft wrapper preflights target runtime/session before proposal and clamps effective proposal depth by remaining context and `token_budget - 1`;
- speculation controller keys qualified measurements by `(proposal_identity, depth)`, so distinct drafters can compete at the same verify depth;
- hysteresis can preserve or switch a concrete proposal strategy as well as a depth while the depth-zero serial baseline remains proposal-free.

Sampled speculative mathematics is source-identified (acceptance ratio and
residual fallback), but sampled execution remains disabled until RNG stream
identity and rejected-draft consumption are fixed. The draft proposal seam is integrated, but concrete proposal-state
MTP compute semantics/kernels, measured drafter selection and proposal qualification remain
open; concrete model-backed draft-state storage is now available but remains unused until the forward gate opens.


## Slice A1.5b — compiled eager serial runtime

Implemented on the host side:

- canonical public shape mapping shared by registry/binder/tooling;
- full-plan kernel binding with strict no-missing-kernel preflight;
- versioned per-step buffer IO for hyper, hidden, injection, routing and logits;
- buffer generations reject stale reads and skipped writes;
- liveness-derived physical arena with deterministic first-fit reuse;
- alignment-aware arena layout and exact generation-to-offset lookup;
- C1 analytical peak reproduced by the liveness allocator at byte alignment;
- compiled kernel-plan identity plus complete execution identity including arena offsets;
- benchmark-time noise excluded from execution identity;
- eager dispatcher receives token id, absolute position, kernel id, state masks, exact buffer generations and arena layout;
- persistent-state generation accounting with overflow rejected before dispatch;
- serial session wrapper with vocabulary/context guards;
- successful token commit advances position/state metadata only after the complete DAG;
- mid-dispatch failure poisons the session instead of pretending persistent device state rolled back;
- preflight failures remain recoverable and touch no dispatcher.

Additional A1.5 runtime work now implemented:

- deterministic ModelManifest layout identity, explicitly distinct from model-content hash;
- model runtime tensor binding with resident, file-mapped, repacked-cache and remote storage identities;
- local tensor address/allocation-generation validation and remote tensors without fake pointers;
- explicit physical scratch-memory binding;
- explicit serial session-state layout for KV, recurrent matrix, GDN convolution, PLE convolution, provisional sparse-index storage, n-gram history and position history;
- separate host/device session-state pools with allocation generations;
- PLE persistent-state accounting corrected to include the exact 180 KiB short-convolution state plus resume metadata;
- allocated context capacity enforced independently from the model maximum;
- common dispatch resolvers for scratch reads/writes, persistent state and model tensors;
- complete runtime resource bundle;
- canonical Qwen4Exp tensor-requirement generator for the full serial DAG;
- execution-tensor audit CLI with missing/ambiguous/satisfied reporting;
- explicit PLE parameter role separated from the giant file-backed n-gram table;
- public Qwen4Exp GGUF naming support for hyper-connections, SSM, indexer, routed experts, shared experts and PLE;
- tensor-access plan identity already incorporated into runtime compatibility identity;
- compatibility identity binding compiled plan + manifest layout + model content + state layout;
- separate physical binding identity adding addresses and allocation generations;
- model-bound serial sessions reject a different checkpoint before dispatch and cannot bypass content checks through the structural/legacy path.

Additional serial-runtime work now implemented:

- complete experimental C1/BF16 HIP kernel catalog for every serial DAG operator;
- zero-missing-step HIP launcher coverage audit;
- runtime launchers for embedding, hyper expand/inject/combine, fused hyper-mix, PLE, GDN, indexed attention, MoE routing, routed experts, shared expert and LM head;
- explicit indexed-attention KV/raw-index state layout and QSA workspace;
- selective model-weight residency with registered file mappings or copied promoted ranges;
- file-backed-candidate tensors remain ordinary host mappings by default, preserving the PLE lookup path;
- fresh HIP scratch/device-state/host-state initialization;
- `strix-hip-model-residency` hardware probe CLI;
- `strix-hip-decode-token` runner that compiles the full C1 serial DAG from a GGUF, accepts explicit token IDs and materializes FP32 logits/top-5;
- versioned independent serial-replay fixtures can be preflighted against the target vocabulary and checkpoint content identity, and the HIP runner rejects non-finite/mis-sized engine logits before exact-argmax and logit-probe qualification.

Still pending:

- running the canonical tensor audit against the actual selected GGUF and closing any remaining selector deviations;
- native Strix Halo compilation and execution of the complete HIP runtime;
- validating registered read-only GGUF mappings on the M5 ROCm stack and comparing against resident-copy fallback;
- first real token/logit replay against the canonical checkpoint/reference implementation;
- tokenizer/chat-template integration after raw token-id replay is correct.

## Slice A2.0 — distributed placement executor

Initial host-side implementation:

- compiled serial steps are assigned to a node from the exact tensor-access plan and manifest placement;
- tensor-free steps inherit a single dependency node when unambiguous, avoiding artificial boundaries;
- cross-node dependency edges produce stable boundary identities and exact activation payload bytes from arena allocations;
- plans fail closed on mixed-node tensor requirements inside one execution step;
- deterministic execution-mode selection requires all remote nodes or an explicit local-model-copy fallback;
- boundary transfer descriptors bind placement-plan identity, request/generation/sequence, boundary identity, dtype and exact payload bytes;
- control-plane reconciliation requires matching protocol, expected node identity, placement plan, runtime compatibility and exact model content;
- receive-side request state enforces exact generation and monotonically increasing sequence, poisoning the request on stale/foreign boundary delivery.

Still pending:

- remote execution backend that consumes these boundary descriptors;
- one-link data-plane integration with the direct/native transport backend;
- native two-node numerical replay and failure/fallback qualification.


## Slice A2.1 — two-node partition search

Initial host-side implementation:

- candidate evaluation separates capacity and latency objectives;
- capacity feasibility is checked per-node rather than by total parameter balance;
- latency estimates use critical-path compute plus uncovered communication and control cost;
- synchronization count is retained but is not an automatic rejection criterion;
- search requires at least one whole-layer baseline while allowing finer operator-level latency candidates;
- capacity selection minimizes maximum per-node weight footprint with imbalance/token-time tie breakers;
- latency selection requires an explicit minimum improvement over the measured local serial baseline.

Still pending:

- generate candidate costs from measured per-operator/native timing and planned A2.0 boundaries;
- canonical two-Strix memory budgets and transport measurements;
- native validation of the selected capacity and latency candidates.


## Slice A0.7b — direct GPU DMA transport contract

Architecture/core contract implemented from independent public transport evidence:

- transport ownership now distinguishes consumer release, repost pending and reusable;
- receive-DMA completion can no longer skip directly to buffer reuse;
- completion taxonomy includes remote GPU consumer-complete credit and RX repost completion;
- backend capabilities identify progress mode, buffer mode, exclusive lease, explicit repost and GPU commit support;
- 32-byte engine-owned GPU commit record binds protocol version, epoch, sequence, payload bytes and stripe geometry;
- commit validation rejects stale epoch/sequence and invalid stripe metadata;
- direct-link lifecycle records control readiness, peer reconciliation, RX priming, exclusive lease, DMA-BUF capability and completion-rearm qualification;
- direct link is unusable until control/reconciliation plus RX priming and re-arm gates pass;
- normative direct-GPU transport design is in `docs/architecture/16-direct-gpu-transport.md`;
- dual-link direct DMA is defined as two independent physical transports with separate TX/RX pools, epochs and leases;
- move-only direct-pool ownership now materializes each planned TX/RX GPU pool as an independently generated `gpu_transport_pool` binding, validates per-link alignment/size/domain and exposes a physical binding identity for pointer-stability/lease checks;
- a portable exclusive-lease registry now binds one physical link to one pool-binding identity, connection epoch and link index at a time, rejecting duplicate acquisition, stale epochs and mutated lease tokens;
- portable completion bridges now enforce the RX ownership chain (DMA complete → consumer acquire/own/release → repost pending → reusable) and separately track TX local reuse, transport ack and remote consumer-complete credit without conflating those milestones;
- a repeated ring re-arm qualification gate now requires a fresh epoch plus the exact expected first descriptor/completion for every cycle before publishing `completion_rearm_qualified`;
- a native-direct backend contract separates capability proof from implementation and wires pool import, GPU commit publication, receive repost and typed native completion events into the existing lease/completion lifecycle with fail-closed validation.

Still pending on native hosts:

- exact kernel/UAPI capability probe for GPU DMA-BUF export/import;
- dedicated uncached/native GPU transport-pool allocator satisfying the new ownership binding contract;
- imported-pool backend and integration of the portable exclusive-lease token with the native lifecycle/completion path;
- native backend/event-source integration with the portable completion bridges;
- target-qualified system-scope release/acquire commit primitive;
- consumer-credit and RX-repost implementation;
- native execution of the repeated ring stop/re-arm/first-completion gate against the real imported-ring backend;
- 1×USB4 versus 2×USB4 completed-GPU-operation measurements;
- independent-flow, directional, role-specialized and striped dual-link qualification.
