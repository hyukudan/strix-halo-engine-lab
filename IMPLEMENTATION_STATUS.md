# Implementation status

Current cross-cutting summary: [qualification ledger](docs/qualification-ledger.md).
The slices below are retained implementation history, not native qualification.

## DeepSeek V4.1 Flash on two nodes and with a discrete GPU (2026-10-23)

- **Two Strix Halo nodes over USB4:** DeepSeek V4.1 Flash 1017 (2-bit routed experts, c-u6 dense overlay, trellis
  mix-B shared-expert and head sites) runs expert tensor parallel decode, expert parallel device prefill, DSpark
  speculative decode, batched decode and an OpenAI-compatible server (`strix-deepseek41-serve`). Production
  defaults: [docs/ds41-production-defaults.md](docs/ds41-production-defaults.md); decisions D-061 to D-064.
- **Measured** (production defaults, bitwise-gated runs): greedy 28.5 tok/s, DSpark 37.5 / 39.3 / 62.3 tok/s
  (EN / ES / code, 24 prompts), batch N=8 95.9 tok/s, device prefill 734 tok/s at 8K, server C1 / C4 / C8
  38.4 / 59.3 / 66.8 tok/s, 8K time to first token 15.8 s on an idle server.
- **RTX split (opt-in, D-064):** CUDA FRONT layers [0, 22) on an RTX PRO 6000 + pair TAIL: DSpark 61.8 tok/s
  (x1.33 over pair-only), greedy 38.3 tok/s, bitwise without trellis sites, composed identity with them.
- **Gates of this state:** R1 and R6 windows bitwise against the stored references on both ranks, batch tokens
  equal to the campaign, prefill 8K logits FNV, server 64/64 requests equal; Qwen replicas 570/570 equal to one node.

## Current state (2026-10-17, f429bd6)

Speculative MTP decode path. Details: [README.md](README.md) and
[DECISIONS.md](DECISIONS.md).

- **Target:** AMD Strix Halo `gfx1151` (Ryzen AI Max+ 395, 128 GB UMA), ROCm
  core-10.1. Model: Qwen3.8-Flash-Next (MoE, GDN recurrent layers, QSA indexed
  attention, PLE n-gram table, hyper-connections) as a UD-Q4_K_XL GGUF plus a
  Q8_0 MTP sidecar.
- **Draft:** MTP draft chain over a draft vocabulary subset
  (`--mtp-draft-vocab`, `--mtp-draft-context-vocab`). Q4_0 copies of the draft
  projections are on by default (`STRIX_MTP_DRAFT_Q4_ALL=0` disables). The chain
  stops early once the draft probability falls below `--mtp-spec-min-p` (0.85 in
  the measurements below).
- **Verification:** exact target verification in known-prefix windows of 2, 4 or
  8 rows (`--mtp-target-window-rows-set 2,4,8`), chosen per cycle by an online
  cost model (the first window per owner is not taken as a sample; samples are
  capped at 2x the estimate). Longest-accepted-prefix acceptance with deferred
  GDN state commit (no state checkpoint); routed experts are executed as a union
  per window. Output is bit-identical to serial greedy decode.
- **Measured** on one Strix Halo node (Ryzen AI Max+ 395, 128 GB), UD-Q4_K_XL,
  256 generated tokens: Python code prompt 48.4 tok/s stable (49.5 tok/s with a
  smaller draft vocabulary); prose ~34–37 tok/s; output identical across runs.
  These are local workload measurements, not qualification evidence.
- **Recent increments** (all bit-exact; `{2,4,8}` window-set measurements from
  the commit log):
  - any-order launches for kernels independent of the preceding one
    (`STRIX_ANYORDER=0` disables): 45.3 -> 46.0 tok/s;
  - PLE 8-row windows projected as two 4-row passes: 46.7 tok/s;
  - QSA attention output with eight slots of loads issued ahead: 47.3 tok/s;
  - robust window cost model plus Q4_0 draft copies by default: 47.9 tok/s;
  - file-backed PLE n-gram table populated after model registration with
    `MADV_RANDOM` + `MADV_POPULATE_READ` (`STRIX_PLE_WARM=0` skips), so decode
    does not read from disk after another process evicted the pages;
  - GDN rows use a fused FMA (neutral).
- **Several requests** (`--mtp-cohort-request`, CLI benchmark mode, D-039): up to four
  independent requests, each with its own state, MTP runtime and golden replay, verified in
  cohort windows that read the projection, expert, HC and head weights once for all
  requests' rows; each request's
  tokens equal its single-request run. See
  [docs/multi-request-cohort.md](docs/multi-request-cohort.md).
- **Diagnostics:** `STRIX_CYCLE_TIMING=1` (host phase marks),
  `STRIX_CYCLE_LOG=1` (per-cycle window rows and wall time).

## Slice A0.1 — portable core

Implemented:

- dependency-free C++20 CMake project;
- Qwen3.8-Flash-Next shape contract pinned from the official config;
- memory planner with overflow-checked arithmetic;
- breakdown of base KV, MTP, provisional indexer, FP32 GDN, rollback, model and
  caches;
- admission decision from MemTotal, baseline and an explicit reserve;
- JSON CLI `strix-memory-plan`, ready to feed the future WebUI;
- candidate top-k oracle with deterministic ties and NaN rejection;
- CTest tests on Linux/WSL with warnings-as-errors.

Not implemented or deliberately blocked:

- canonical QSA semantics, n-gram/PLE, hyper-connections and GDN: waiting for
  official contracts/oracles before writing kernels;
- HIP allocator/residency and `gfx1151` kernels;
- sampled MTP;
- server and administration panel.

> Note (2026-10-17): the first two items above are superseded. Later slices
> (A1.x) and the commit log through `f429bd6` add HIP residency and `gfx1151`
> kernels for QSA, PLE/n-gram, hyper-connections and GDN, used by the greedy MTP
> path described in "Current state". Sampled MTP decode is not part of that path.

## Reproduction

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
./build/strix-memory-plan --tokens 32768 --sessions 1   --resident-gib 68 --baseline-gib 2 --reserve-gib 18
./build/strix-platform-probe
./build/strix-model-manifest MODEL.gguf
```

The indexer size is a provisional estimate derived from the config. It can be
excluded with `--no-indexer`. `--rollback-full` quantifies why copying the full
GDN state per draft does not scale; the target path will be a minimal snapshot
plus recompute if the contract allows it.

> Note (2026-10-17): the current MTP path uses deferred GDN state commit with no
> state checkpoint (commits `8793b4f`, `1212d3f`).

## Slice A0.2 — GGUF inspection

Implemented in `include/strix/gguf_reader.hpp` and `src/gguf_reader.cpp`: a
defensive little-endian GGUF v3 reader and the `strix-inspect-gguf` CLI. It
validates metadata, arrays, the tensor directory, block geometry, sizes,
alignment, ranges and overlaps without loading weight data.
Limitations and the canonical review are documented in `GGUF_READER.md`.

Current validation:

- `RelWithDebInfo` build with `-Werror`;
- valid and corrupt fixtures in CTest;
- Debug build with AddressSanitizer and UndefinedBehaviorSanitizer;
- 100 % of the current tests pass in both builds.


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


## Slice A2.2 — dual-link USB4 runtime scheduler

Initial host-side implementation:

- directional link profiles allow different outbound/inbound performance on the same physical USB4 link;
- live samples update payload rate, one-way latency and in-flight bytes with bounded EWMA;
- flow-level affinity avoids unnecessary link flapping with explicit switch hysteresis;
- new flows can move between links as directional telemetry changes;
- large non-latency-critical messages reuse the existing complete-cost striping planner and stripe only when two qualified independent links beat the best single-link estimate;
- physical-link health updates invalidate stale affinities, immediately downgrade striped traffic to the surviving link and allow explicit re-qualification/recovery for new flows;
- a completion telemetry bridge now tracks per-handle submit time/link/direction and converts receive-ready, remote-consumer-complete and receive-reposted milestones into directional scheduler observations without inventing native timestamps.
- distributed placement boundaries now map to stable plan-bound flow ids, direction and size-class priority, so A2 traffic can feed the dual-link scheduler without ad-hoc caller identities.

Still pending:

- feed real native USB4 timestamps into the new completion-to-scheduler telemetry bridge;
- qualify 1× versus 2× USB4 on real dual-Strix hosts.


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

- native M5/Strix Halo build and execution (2026-10-17: native `gfx1151`
  kernels now build and run on an M5 node as part of the decode path, see
  "Current state"; the remaining items are not closed by that);
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
- CTest coverage including Q4_K, Q8_0 and IQ4_XS geometry;
- independent IQ4_XS 136-byte block/dequant contract with nonlinear codebook and 6-bit local-scale extraction;
- host IQ4_XS fixture compiled with strict warnings and passing.

Standalone `lab/packed_projection/` implemented:

- Q8_0 packed reference kernel for `gfx1151`;
- IQ4_XS compact packed candidate for `gfx1151`;
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
- native compile/profile of the IQ4_XS compact path and comparison against Q8_0;
- canonical checkpoint-format decision after quality plus full memory/runtime qualification;
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
- a portable MoE layer runtime now consumes only validated runtime-bound schedules, starts routed/shared branches concurrently when qualified, serializes them when required, and joins both before combine through an async-capable backend seam;
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

- native Strix Halo build/run of the complete HIP path (2026-10-17: the HIP
  path now builds and runs natively on a Strix Halo node, see "Current state";
  qualification status is tracked in the qualification ledger);
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
- MTP semantic gates are tracked independently: input fusion and next-step draft-state propagation are resolved; QSA scoring/selection is resolved including pooled-key RoPE and an explicit deterministic lower-block-index tie rule; output mixer semantics are resolved, the exact `hc_head_norm/down/up` triplet is now uniquely bound and geometry-validated, and an explicit sidecar `shared_head` resolves the LM-head binding; the complete forward gate remains closed;
- executable MTP construction is explicitly blocked while the complete `forward_semantics_resolved` gate remains false;
- `CompiledMtpDraftPlan` now freezes the structural step order and stable relocation-independent plan identity while surfacing unresolved readiness blockers instead of launching;
- offline `strix-mtp-manifest-audit MTP.gguf` reports required structural tensor families, unresolved MTP tensors and missing families without allocating GPU state;
- host-side MTP input-fusion oracle pins per-stream semantics: normalized embedding is projected once and broadcast, normalized wide hidden is reshaped into `hc_count` streams and projected independently before summation;
- the same input-fusion oracle now also consumes the checkpoint's canonical single `eh_proj` matrix directly as row-major `[H, 2H]`, with exact `[enorm(token), hnorm(stream)]` concatenation order; tests prove equivalence to the split projection form and detect reversed concat order;
- evidence-gated MTP plan qualification separates structural readiness from executable qualification: all structural blockers except the forward gate must be closed, then an independent multi-step replay must satisfy the declared token/hidden/logit policy before the plan is executable;
- replay evidence is bound to proposal identity, compiled plan identity, policy identity and independent-reference identity; compiled MTP identity is now recomputable and qualification snapshots the exact plan identity, so post-qualification plan/forward mutation cannot reuse stale replay evidence even if a new plan identity is recomputed;
- next-step MTP draft state is pinned as the post-block wide residual before final head collapse, with width `hc_count * hidden_size = 10240` for the target model;
- concrete move-only MTP draft-state memory ownership now allocates and validates a proposal-identity-bound device buffer with explicit row count, element size, alignment and allocation generation;
- a composed MTP forward contract validates cross-stage geometry and checkpoint/tensor/output bindings across input fusion, attention, wide draft state and final output, producing a relocation-stable composition identity while leaving numerical execution gated on replay/kernels;
- the Qwen4Exp NextN graph semantics are now pinned in a versioned semantic contract: per-HC hidden RMS normalization, repeated normalized token embedding, embedding-before-hidden concat into the 2×hidden `eh_proj`, attention combine before FFN combine, publication of the wide next-state only after the FFN combine and before a no-injection head mix, then LM head; this semantic contract contributes to compiled MTP plan identity but does not bypass independent replay qualification;
- a qualified MTP forward runtime adapter now connects the speculative drafter seam to the replay-qualified step runtime: each candidate obtains a validated cycle input (seed token/position/wide hidden), executes the qualified forward, reads a bounded vocabulary token and delegates post-target reconciliation without inventing native numerics;
- a qualified MTP forward runtime walks the compiled 11-step predictor sequence through an explicit backend interface, validates plan/binding/proposal-state/draft-memory before dispatch, and stops immediately on backend failure; this creates the kernel integration seam without bypassing replay qualification;
- the qualified MTP draft wrapper now requires that physical state binding and passes it to the forward/reconciliation backend, preventing model-backed proposal compute from running against absent or foreign draft-state storage;
- qualified MTP forward dispatch now also requires the exact ModelManifest + ModelRuntimeBinding pair, revalidates the draft physical binding identity before any step, and exposes those resources on every step context so native backends can resolve real tensor addresses/generations without bypassing checkpoint identity;
- MTP forward dispatch now preserves the complete candidate cycle input, including wide-hidden physical binding and candidate-specific absolute seed position; the cycle validator permits later speculative positions while rejecting positions before the unchanged target-session base, so multi-candidate drafting no longer collapses onto one target position;
- memory admission accounts that draft state separately per session; the element size is an explicit planning input (2 bytes by default), not a resolved MTP dtype claim;
- deterministic bounded prompt-lookup drafter using the most recent non-overlapping suffix n-gram match;
- prompt-lookup history has an explicit token-capacity bound and proposal identity;
- draft wrapper preflights target runtime/session before proposal and clamps effective proposal depth by remaining context and `token_budget - 1`;
- speculation controller keys qualified measurements by `(proposal_identity, depth)`, so distinct drafters can compete at the same verify depth;
- measured-cycle aggregation now groups repeated observations by campaign + proposal/depth, requires a minimum repetition count and correctness-clean cycles before marking a sample qualified, summarizes component costs by median and expected committed tokens by arithmetic mean, and feeds those samples directly into the existing controller;
- hysteresis can preserve or switch a concrete proposal strategy as well as a depth while the depth-zero serial baseline remains proposal-free;
- sampled speculation now has a stateless counter-based RNG identity contract keyed by seed, request identity, sampling-policy identity, absolute token position, draw class and draw index; draft-acceptance/residual draws therefore cannot advance or perturb the serial target draw assigned to any token position;
- a portable lossless speculative-sampling verifier now implements per-token acceptance `min(1,p/q)`, rejection fallback from normalized `max(0,p-q)`, and ordinary target categorical sampling on the separately addressed serial-target RNG stream, with strict distribution validation.
- deterministic distribution qualification now sweeps many independently keyed requests and checks that the speculative output marginal matches target `p` while observed draft acceptance matches `sum(min(p,q))` within declared statistical tolerances;
- a stateful sampled-cycle planner now maps accepted draft prefixes, first-rejection residuals, all-accepted target bonuses, EOS cropping and token budgets onto the same pre-window snapshot/restore/replay state semantics used by greedy speculation;
- a sampled target executor now performs identity preflight, captures the pre-window snapshot, consumes target distributions, restores/replays only committed accepted candidates after rejection/EOS, and rolls back on verify/finalize/replay failures;

Sampled speculative mathematics, RNG stream identity/consumption and the portable acceptance/residual sampler are fixed and distribution-qualified. Host-side rollback/commit execution is explicit. A compiled sampled backend now joins the already-materialized serial base distribution with D verify-row distributions through a stable FP32 softmax collector, while a HIP logits reader provides the device-to-host materialization seam without leaking HIP into the speculation core; native end-to-end sampled qualification remains pending. The draft proposal seam is integrated, but concrete proposal-state
MTP compute semantics/kernels and native proposal qualification remain
open; host-side measured drafter selection is now wired through repeatable campaign aggregation.

> Note (2026-10-17): superseded for greedy decode. The commit log through
> `f429bd6` adds a native MTP draft head and draft chain (`f8a2d2e` onwards) and
> verifies drafts in known-prefix windows with longest-prefix acceptance and
> deferred GDN state commit, without a pre-window state snapshot (`8793b4f`,
> `1212d3f`). See "Current state" at the top. Sampled speculative decode remains
> as described above.


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
- receive-side request state enforces exact generation and monotonically increasing sequence, poisoning the request on stale/foreign boundary delivery;
- host data-plane packaging now binds each boundary to its stable flow classification, request descriptor and live dual-link transfer plan, while receive-side validation rechecks the same flow/request identities before sequence advancement;
- the native distributed backend consumes boundary packages through an explicit staging/control-plane seam, supports both single-link and exact-cover striped plans across distinct leased sessions, publishes indexed GPU commits per slice, validates all received commits with the logical-message tracker before reassembly, and withholds consumer credit/repost until every stripe has been consumed successfully;
- sequenced remote-step control now binds command/ACK to protocol, placement plan, request/generation, control sequence, step id/op and target node; stale, foreign or unsuccessful ACKs poison the controller, and boundary metadata is revalidated against the same request/epoch before forwarding;
- a bounded little-endian control-wire codec now serializes/deserializes remote-step commands, ACKs and boundary metadata (including striped link slices) behind an abstract byte-stream, with explicit magic/version/type/length checks and configurable frame/slice limits;
- a concrete POSIX TCP control byte-stream now owns connected sockets, supports connect/listen-once/adopt flows, disables Nagle on TCP peers, length-prefixes every framed message, retries interrupted I/O and rejects oversized lengths before allocation;
- a bounded control server pump now inspects validated frame kinds, executes remote-step commands through a receiver handler and emits identity-mirroring success/failure ACKs, while boundary-transfer frames are delivered to the receiver-side metadata handler without ad-hoc decoding;
- the receiver-side control handler now binds one global placement/request identity to an explicit physical receiver node, enforces monotonic remote-command sequence, validates target step ownership/op, queues only plan-owned boundaries addressed to that node, rejects duplicate pending boundary metadata and poisons itself on stale/foreign control input or local execution failure;
- an end-to-end portable control-plane test now drives a sequenced client through the real framed socket stream/server/receiver path, proving remote-step ACK and boundary-metadata delivery compose correctly on one connection.

Still pending:

- native/direct control-channel alternative to TCP and two-host control-plane qualification;
- native two-host qualification of single-link versus striped multi-link boundary submission/reassembly;
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

Additional A2.1 implementation:

- placement-derived candidate generation now consumes exact manifest placement, measured per-step timings for both nodes and measured per-boundary uncovered communication/control costs; resident-only versus all-manifest weight accounting is explicit and missing/duplicate evidence fails closed.

Still pending:

- populate candidate evidence with native per-operator and boundary measurements from the two Strix hosts;
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
- a native-direct backend contract separates capability proof from implementation and wires pool import, GPU commit publication, receive repost and typed native completion events into the existing lease/completion lifecycle with fail-closed validation;
- a move-only native-link session now owns the exclusive lease/import lifetime and the send/receive completion bridges as one unit, revalidating lease/pool/link identity before commits, event polling and repost, and releasing the imported pool before dropping the lease;
- the native backend/session contract now includes fail-closed prepare-send/prepare-receive/submit/cancel seams; session preparation only accepts buffers inside the exact leased TX/RX pool and matching allocation generation, so future kernel/UAPI backends can plug into A2 without bypassing pool ownership;
- receive-side native consumption now requires GPU commit-record readback that validates protocol/epoch/sequence/payload against the transfer descriptor before consumer ownership is granted; the one-link A2 adapter additionally requires an exact one-stripe complete commit;
- native-link sessions can optionally bind the dual-link telemetry bridge: tracked submit records the leased link/direction/bytes, and timestamped native completion events automatically update the live scheduler without bypassing completion ownership.
- remote consumer-complete credit is now an explicit one-shot receive-side operation: only consumer-released buffers may publish it, duplicate publication is rejected before the backend call, and the native-link session binds credit publication to the current lease before RX repost.

Still pending on native hosts:

- exact kernel/UAPI capability probe for GPU DMA-BUF export/import;
- dedicated uncached/native GPU transport-pool allocator satisfying the new ownership binding contract;
- concrete imported-pool backend implementation against the native kernel/UAPI;
- native event-source implementation feeding the already-integrated session completion bridges;
- target-qualified system-scope release/acquire commit primitive;
- native execution of the repeated ring stop/re-arm/first-completion gate against the real imported-ring backend;
- 1×USB4 versus 2×USB4 completed-GPU-operation measurements;
- independent-flow, directional, role-specialized and striped dual-link qualification.

## Canonical automatic v3 source adoption (2026-10-15)

Primary source `265fd78762884d18eb32aa60294408f080088d52` now selects the
content-admitted all-five automatic profile. Native225, 14-module/245-kernel
preservation, seven actual CLI cases, full554 arrays and all 512 generated IDs
passed. Separate descriptive three-pair campaigns measured +25.99% over the
preserved baseline and +15.55% over the preceding automatic profile on the same
execution host. These are limited workload observations, not additive gains or
statistical significance. See E-056 and
[the adoption record](docs/decode-optimization-profile.md#qualified-adoption-2026-10-15).
Earlier status entries retain their historical revision and scope. Source
adoption does not imply remote binary deployment.
