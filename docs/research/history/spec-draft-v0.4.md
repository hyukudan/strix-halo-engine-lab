# Technical specification v0.4 — inference engine for Strix Halo

Status: research draft, revised for technical acceptance and expanded with official
Qwen sources and the GTT policy. P0 gates and on-hardware measurement are still
open. Project name not yet decided.

Traceability: the data supporting this specification is indexed in
[`EVIDENCE.md`](EVIDENCE.md); changes of direction are recorded in
[`DECISIONS.md`](DECISIONS.md). A statement marked as a hypothesis must not be
read as a closed requirement.

## 1. Objective

Build a local, reproducible and auditable inference engine for AMD Strix
Halo `gfx1151`, optimized first for Qwen3.8-Flash-Next. It must combine high
prefill, competitive decode, real concurrency and verifiable quality.

### Initial targets

- Native Linux x86-64, ROCm/HIP, `gfx1151`.
- C++20 + HIP; Python only for tooling and oracles.
- GGUF as the input format; optional, lossless, versioned internal cache.
- First slice: AR greedy/sampled + greedy-only MTP, Chat Completions in a
  separate gateway and an explicit replay contract. Sampled MTP waits for a canonical RNG/replay contract; the probabilistic acceptance algorithm is already fixed.
- C1 as the fast path and C2/C4 as the first concurrent paths.
- Initial context qualified according to the demonstrated UMA budget; 262,144,
  Responses API, prompt lookup and C6/C8 remain later extensions.

### Non-goals of the first version

- portability to other GPUs;
- multiple model families;
- vision before text/MTP is qualified;
- numerical approximations without an explicit quality gate;
- compatibility with every existing quantization.

## 2. Principles

1. **Auditable provenance.** Contracts come from papers, canonical public
   sources and independent oracles.
2. **Finite, justified specialization.** Each variant must correspond to
   observed shapes and pass an end-to-end check.
3. **Correctness before speed.** No change lands by regenerating goldens or
   relaxing tolerances.
4. **Reproducibility.** Compiler, ROCm, Linux kernel, BIOS/IOMMU, power,
   commit, weights, prompt and cache are part of the result.
5. **One winning implementation.** Losing experiments are documented and
   removed; permanent flags do not accumulate.
6. **Primary source before secondary reference.** QSA, GDN, Gated Residual, MoE and
   n-gram are specified from official Qwen sources; secondary observations serve to
   formulate and test hypotheses, never as the sole oracle.

## 3. Architecture

```text
Network/VPN
  -> TLS reverse proxy
    -> inference gateway (Chat API + optional playground)
      -> UDS/loopback -> engine
        -> admission + scheduler
          -> session state (KV, recurrence, RNG, snapshots, rollback)
            -> model executor -> kernel registry -> HIP/hipBLASLt/HIP graphs
    -> admin API (stronger credential, no shell)
      -> typed operations -> supervisor/model manager/benchmark runner

Engine and supervisor -> read-only metrics/logs -> administrative panel
```

### 3.1 Loader and weights

- minimal built-in GGUF reader;
- strict validation of architecture, dimensions, dtype and hash;
- read-only mmap view of the shards for CPU loading; before launching GPU work, every
  range used by the hot path must be registered and resident according to the
  confirmed platform contract;
- lossless planner that decides per tensor: direct read, temporary repack or
  persistent internal cache;
- PLE/n-gram subsystem with a candidate asynchronous direct read (`pread` or
  equivalent), bounded pool, coalescing, row cache and telemetry; demand-paged
  `mmap` will be a control, not the assumed option;
- cache identified by weights hash + layout version + compiler;
- no implicit requantization;
- `inspect` command that reports bytes, bpw, families, layouts and hashes.

### 3.2 UMA memory

- allocator with visible categories: mapped weights, copied weights, KV,
  recurrence, rollback, scratch, libraries and cache;
- parallel, bounded CPU prefault for active ranges, outside the GPU kernel;
- A/B tests of registered mmap versus anonymous copy with huge pages;
- session pool sized in positions, not only in slots;
- snapshots of valid rows only;
- backpressure before committing memory that does not fit; the cap starts from
  measured `MemTotal` minus the engine-less baseline and the reserve, and internal
  accounting takes precedence over a `MemAvailable` that may overestimate pinned pages;
- metrics for GTT, corrected MemAvailable and page faults during preparation;
- pre-load budget: weights and mappings, tables, KV, recurrence,
  rollback, scratch, libraries, OS and recovery margin. A checkpoint that
  nominally fits is not admitted if it violates the margin.
- configure GTT as a mapping limit close to the useful maximum (initial target
  124 GiB on a nominal 128 GB machine), computed from the actual RAM;
- keep a separate initial effective reserve of 16–20 GiB in the admission
  controller: enlarging GTT does not authorize consuming that RAM;
- BIOS with a verified minimal carve-out; `amd_iommu=off` and other
  experimental parameters are never defaults and require a dedicated profile, a warning and
  rollback;
- clean shutdown and a residual-GTT watchdog: after an unclean termination or
  inconsistent usage, reload is blocked and escalated to a controlled reboot.

### 3.3 Executor

- separate paths for prefill, AR decode, MTP verification and catch-up;
- static operator graph per model, with no general runtime in the hot path;
- HIP graph per stable decode shape;
- invariant: no replay observes a reused input; a ring versioned by
  slot/generation is a candidate mechanism, not the contract;
- retained PM4 only as an opt-in experiment for repetitive decode, negotiated
  through capabilities and with fallback to the standard runtime; it does not constrain prefill;
- eager mode for warm-up, tuning and ragged shapes;
- overlap only with demonstrably safe streams and ownership;
- each request keeps independent KV, recurrent state, RNG and rollback.

### 3.4 Kernel registry

Each entry will declare:

- operator, shape/range and phase;
- input and output dtype/layout;
- wave32/wave64, workgroup, VGPR, SGPR, LDS and scratch;
- reduction order and rounding boundaries it must preserve;
- compiler flags, FMA contraction, denormal mode and ABI version;
- correctness evidence;
- measured cost per batch, depth and cache state;
- identity of the toolchain that produced the code.

The selector reads an offline-generated plan. If the identity does not match,
it uses a conservative path and requests retuning; it never times algorithms in
production non-deterministically.

### 3.5 Priority kernel families

1. RMSNorm/residual/RoPE and conversions.
2. Q4/Q5/Q8 projections for C1 and batches 2–8.
3. MoE: routing, expert grouping, fused gate/up and down.
4. Gated DeltaNet: first fix FP32/state/order/rollback; tiled, register-resident
   state and DPP/LDS are candidates after in-house profiling.
5. QSA: canonical sparse oracle pending confirmation from Qwen; dense will be the
   performance and quality baseline, and will not be called exact or equivalent.
   Indexer, top-k with ties, metadata and attention are evaluated as a pipeline.
6. Sampling/argmax and MTP verification.

For each family, wave32 and wave64 will be tested where they make sense; no single
observation will be turned into a universal rule.

## 4. Scheduling and speculation

- FIFO queue with explicit memory and token limits;
- C1 does not pay for batching structures it does not use;
- C2/C4 groups compatible rows in the first slice, keeping state private;
- batching of the full transformer, including predictor and verification;
- MTP with 1–7 drafts, controlled by full-cycle cost;
- timing-based control only in greedy; sampled uses a fixed policy for replay;
- exact rollback for each accepted prefix;
- prompt lookup will be added after the first slice and measured separately;
- cancellation before touching shared state, and consistent snapshots.

## 5. Quality

Three mandatory levels:

1. **Operator oracle:** independent CPU FP64/FP32 formula, including tails
   1/8/9/32/33 and small non-saturated values.
2. **Model replay:** first divergent layer/logit with identical token
   history; AR, MTP, sampling, EOS, continuation and restoration.
3. **Task quality:** PPL/KL/top-1, NIAH and an agents/code corpus.

An optimization must preserve the declared contract. If the contract allows a
tolerance, the tolerance is fixed before measuring performance.

The contract distinguishes C1 replay from concurrent replay. "Deterministic" will never be a
global label: each mode declares whether it requires bitwise identity, token
identity or numerical tolerance, including the reduction order.

## 6. Benchmark

Minimum matrix:

- prefill: 2K, 8K, 32K, 128K;
- decode: depth 0/4K/16K/32K/64K/128K;
- tg128, C1/C2/C4 initially; C6/C8 after C4 is qualified;
- AR, MTP on mixed and repetitive text, prompt lookup;
- cold/warm start, peak memory and time to readiness;
- baseline, candidate and unaffected control in the same environment.
- at least three repetitions, published dispersion, exclusive GPU lock and
  memory guard; one conceptual change per experiment whenever
  possible.

Record individual latencies, aggregate throughput, acceptance, completed
tokens, launches, GPU busy, page faults and memory bytes. A microbenchmark
gain does not count until it survives the full request without a profiler.

## 7. Phases

### Phase A0 — fully offline lab

- public development image pinned by digest and a second control toolchain;
- `gfx1151` compilation, HSACO extraction and ELF/metadata/ISA audit;
- inventory of `gfx1151` code objects in rocBLAS, hipBLASLt and dependencies;
- double compilation with hashes and normalized diff;
- CI gates for VGPR, SGPR, LDS, spills and code size;
- CPU oracles and tiny models/tensors with no dependency on the final hardware.

### Phase A — Strix Halo lab

- hardware/software inventory;
- confirmation that the objects produced in A0 execute;
- bandwidth meter and WMMA ceiling;
- reproducible benchmark harness with fixed prompt, context, power mode and
  completion criteria;
- corpus and weights pinned by hash.

### Phase B — correct reference

- GGUF, tokenizer and minimal CPU/oracle;
- correct AR execution, even if slow;
- state, snapshot and sampling tests.

### Phase C — C1 text

- native projection, MoE, DeltaNet and QSA kernels;
- CPU contracts/oracles for GDN, sparse QSA, n-gram/PLE, hyper-connections and routing;
- after correct AR and in-house profiling: tiled GDN prototypes and direct PLE reader;
- ablation of dense QSA versus full sparse, including quality; partial
  integration is not accepted;
- separate prefill and decode;
- HIP graphs and offline tuning.

### Phase D — MTP and initial concurrency

- predictor, acceptance, rollback and catch-up;
- C2/C4 batching and memory-based admission;
- later expansion to C6/C8 and prompt cache only after C4 is qualified.

### Phase E — server and product

- APIs, health/metrics, inspect/verify/ppl/niah;
- reproducible packaging and documentation;
- hardening of cancellation, driver failures and memory pressure.

The server includes a watchdog, request draining, an explicit policy for driver
hang/reset and a visible degraded state. Every benchmark records
power/TDP, clocks and temperature together with the software environment.

## 8. Decision gates before writing the full engine

1. Choose the public checkpoint and canonical quantization after demonstrating a
   complete budget within 128 GB UMA.
2. Fix the benchmark protocol: same prompt, context, power mode and
   completed-token criterion for every measured run.
3. Confirm Linux/ROCm/kernel of the target machine.
4. Decide whether the internal weights cache is worth its startup time and disk space.
5. Set the license and the MIT reuse policy.
6. Resolve the P0 items in `DESIGN_ACCEPTANCE.md` and submit the consolidated spec to a
   second review.
7. Pin the canonical repo, commit and license for QSA, DeltaNet, MoE routing,
   MTP, tokenizer and chat template.
8. Demonstrate that the chosen ROCm libraries contain `gfx1151` paths or define
   engine-side fallbacks.
9. Approve the use case and distribution against the Qwen Community License 1.0.
10. Confirm the QSA semantics and its oracle; do not replace it with dense attention
    on the basis of a performance observation alone.

## 8.1 Work allowed before receiving the hardware

- compile HIP offline for `gfx1151` with a pinned toolchain;
- inspect ISA, VGPR/SGPR/LDS, spills and code size;
- implement the GGUF loader, hashes and layout planner;
- create CPU oracles and operator/state tests;
- simulate admission, batching, cancellation, snapshots and rollback;
- define the API, metrics, artifact formats and administrative plane;
- build candidate microkernels without declaring winners.

Turning a successful compilation or a static count into a performance claim is
forbidden. Kernel selection, tuning, UMA, page faults, power, temperature and
stability require the real Strix Halo.

## 8.2 Remote administrative plane

The product will have three separate domains: a local engine with no network exposure,
an inference/chat gateway and an administrative plane. Its priority functions
will be health/versions, validated
configuration, service lifecycle, logs, memory/GTT, temperature/power,
models, cache/sessions and benchmark execution/comparison. A chat playground
may live as an optional tab inside the panel, solely for smoke
tests and request reproduction.

The engine's internal port is never published: UDS or restricted loopback is used.
The gateway uses API keys or OIDC, quotas and auditing. The panel uses a local agent
with explicitly allowed typed operations, WebAuthn or mTLS, TLS,
auditing, CSP and CSRF protection. Remote access is preferably restricted to
VPN. Destructive actions or restarts show the target and require
confirmation. The chat tab does not inherit administrative privileges and the UI
does not execute arbitrary shell commands.

The GTT section will show the configured limit versus consumption and effective reserve,
detected carve-out, `MemAvailable` and residual state. It can prepare/apply an
allowlisted GTT/TTM profile through a one-shot boot, preflight,
re-authentication and confirmation. A health check confirms driver/GTT/agent; if
it fails, the next boot reverts to the previous entry. `amd_iommu=off` and
experimental parameters stay out of the WebUI and require a console or SSH.
The playground is served from a different origin and cookies than the administrative panel,
with strictly sanitized model output.

## 9. Specification history

| Version | Date | Change |
| --- | --- | --- |
| 0.4 | 2026-09-28 | Official Qwen sources, UMA/GTT policy and new candidates: tiled GDN, direct PLE reader, UMA ring and QSA/PM4 negative controls. |
| 0.3 | 2026-09-28 | Incorporates design review: P0, reduced scope, numerical contracts, A0 and privilege separation. |
| 0.2 | 2026-09-28 | Pre-hardware cross-compilation and remote administrative plane. |
| 0.1 | 2026-09-28 | Initial scope, architecture, quality, benchmark and phases. |
