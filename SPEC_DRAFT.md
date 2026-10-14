# Technical specification v0.4 — independent Strix Halo engine

Status: research draft revised for technical acceptance and expanded with
Qwen official sources, GTT policy and the external platform study review.
P0 gates and hardware measurement remain outstanding in this draft. The
project name is still undecided.

This document retains the original design history. Consult the
[qualification ledger](docs/qualification-ledger.md) for current implementation
and promotion status and the [architecture guide](docs/README.md) for organized
contracts. The supporting evidence is indexed in [EVIDENCE.md](EVIDENCE.md);
changes of direction are recorded in [DECISIONS.md](DECISIONS.md). A hypothesis
must not be read as a finalized requirement.

## 1. Objective

Build a local, reproducible and auditable inference engine for AMD Strix Halo
`gfx1151`, optimized first for Qwen3.8-Flash-Next. It must combine high prefill
throughput, competitive decode, real concurrency and verifiable quality. It
will not be an external runtime B fork or a reconstruction of external runtime A.

### Initial goals

- Native Linux x86-64, ROCm/HIP, `gfx1151`.
- C++20 + HIP; Python only for tools and oracles.
- GGUF input; optional lossless, versioned internal cache.
- First slice: greedy/sampled AR + greedy-only MTP, Chat Completions through a
  separate gateway and an explicit replay contract. Sampled MTP awaits a
  canonical RNG/replay contract; the probabilistic acceptance algorithm is
  already fixed.
- C1 fast path and C2/C4 as the first concurrent paths.
- Initial qualified context based on the demonstrated UMA budget; 262,144,
  Responses API, prompt lookup and C6/C8 are later extensions.

### Outside the first version's goals

- portability to other GPUs;
- multiple model families;
- vision before text/MTP qualification;
- numerical approximations without an explicit quality gate;
- compatibility with every existing quantization.

## 2. Principles

1. **Independent code and auditable provenance.** Contracts from papers,
   canonical public sources and independent oracles. External runtime A
   contributes only anonymized aggregate observations; no kernel bodies,
   signatures or structures are copied.
2. **Finite, justified specialization.** Every variant must correspond to
   observed shapes and pass an end-to-end control.
3. **Correctness before speed.** Changes are not accepted by refreshing
   goldens or relaxing tolerances.
4. **Reproducibility.** Compiler, ROCm, Linux kernel, BIOS/IOMMU, power,
   commit, weights, prompt and cache are part of each result.
5. **One winning implementation.** Document and remove losing experiments;
   do not accumulate permanent flags.
6. **Primary sources before references.** Specify QSA, GDN, Gated Residual,
   MoE and n-gram from official Qwen sources; external runtime A, external
   runtime B and external platform study formulate and test hypotheses,
   never serving as the sole oracle.

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

- minimal, independently authored GGUF reader;
- strict architecture, dimension, dtype and hash validation;
- read-only mmap shard views for CPU loading; before GPU launch, every hot-path
  range must be registered and resident under the confirmed platform contract;
- lossless per-tensor planner: direct read, temporary repack or persistent
  internal cache;
- candidate PLE/n-gram subsystem with asynchronous direct reads (`pread` or
  equivalent), bounded pool, coalescing, row cache and telemetry; demand-paged
  `mmap` is a control rather than the assumed choice;
- cache identity includes weight hash + layout version + compiler;
- no implicit requantization;
- `inspect` command reports bytes, bpw, families, layouts and hashes.

### 3.2 UMA memory

- allocator with visible categories: mapped weights, copied weights, KV,
  recurrence, rollback, scratch, libraries and cache;
- bounded parallel CPU prefault of active ranges outside GPU kernels;
- A/B tests of registered mmap versus anonymous copy with huge pages;
- session pool dimensioned in positions, not only slots;
- snapshots only of valid rows;
- backpressure before committing memory that does not fit; the cap starts from
  measured `MemTotal` minus the engine-free baseline and reserve, and internal
  accounting takes precedence over `MemAvailable`, which can overestimate
  pinned pages;
- GTT, corrected MemAvailable and preparation-time page-fault metrics;
- pre-load budget covering weights and mappings, tables, KV, recurrence,
  rollback, scratch, libraries, OS and recovery headroom. A nominally fitting
  checkpoint is not admitted if it violates the margin;
- GTT as a mapping limit near the useful maximum (initial target 124 GiB on a
  nominal 128 GB host), calculated from actual RAM;
- separate initial effective admission-controller reserve of 16–20 GiB:
  increasing GTT does not authorize consuming this RAM;
- verified minimum BIOS carve-out; `amd_iommu=off` and other experimental
  parameters are never defaults and require a dedicated profile, warning and
  rollback;
- clean shutdown and residual-GTT watchdog: an unclean termination or
  inconsistent usage blocks reload and escalates to a controlled restart.

### 3.3 Executor

- separate prefill, AR decode, MTP verification and catch-up paths;
- static per-model operator graph without a general runtime in the hot path;
- HIP graphs for stable decode shapes;
- invariant: no replay observes a reused input; a versioned slot/generation
  ring is a candidate mechanism, not the contract;
- retained PM4 only as an opt-in repeated-decode experiment, capability
  negotiated with standard-runtime fallback; it does not constrain prefill;
- eager mode for warm-up, tuning and ragged shapes;
- overlap only with demonstrably safe streams and ownership;
- independent KV, recurrent state, RNG and rollback for every request.

### 3.4 Kernel registry

Each entry declares:

- operator, shape/range and phase;
- input/output dtype and layout;
- wave32/wave64, workgroup, VGPR, SGPR, LDS and scratch;
- required reduction order and rounding boundaries;
- compiler flags, FMA contraction, denormal mode and ABI version;
- correctness evidence;
- measured cost by batch, depth and cache state;
- identity of the toolchain that produced the code.

The selector reads an offline-generated plan. On identity mismatch it uses a
conservative path and requests retuning; it never times algorithms
nondeterministically in production.

### 3.5 Priority kernel families

1. RMSNorm/residual/RoPE and conversions.
2. Q4/Q5/Q8 projections for C1 and batches 2–8.
3. MoE: routing, expert grouping, fused gate/up and down.
4. Gated DeltaNet: fix FP32/state/order/rollback first; tiling, register state
   and DPP/LDS are candidates after our own profiling.
5. QSA: canonical sparse oracle awaiting confirmation from Qwen; dense is a
   performance/quality baseline, not labelled exact or equivalent. Evaluate
   indexer, tied top-k, metadata and attention as a pipeline.
6. Sampling/argmax and MTP verification.

Test wave32 and wave64 where appropriate for each family; external runtime A's
observations do not become a universal rule.

## 4. Scheduling and speculation

- FIFO queue with explicit memory and token limits;
- C1 does not pay for unused batching structures;
- C2/C4 groups compatible rows in the first slice while keeping state private;
- batching of the complete transformer, including predictor and verification;
- MTP with 1–7 drafts, controlled by complete-cycle cost;
- timing control only for greedy; sampled mode uses fixed policy for replay;
- exact rollback for every accepted prefix;
- prompt lookup after the first slice, measured separately;
- cancellation before touching shared state, and consistent snapshots.

## 5. Quality

Three mandatory levels:

1. **Operator oracle:** independent CPU FP64/FP32 formula, including tails
   1/8/9/32/33 and small nonsaturated values.
2. **Model replay:** first divergent layer/logit with identical token history;
   AR, MTP, sampling, EOS, continuation and restoration.
3. **Task quality:** PPL/KL/top-1, NIAH and an agent/code corpus.

An optimization must preserve its declared contract. Where the contract allows
tolerance, fix the tolerance before measuring performance.

Distinguish C1 and concurrent replay. Determinism is never a global label:
each mode declares whether it requires bitwise identity, token identity or
numerical tolerance, including reduction order.

## 6. Benchmark

Minimum matrix:

- prefill: 2K, 8K, 32K, 128K;
- decode: depth 0/4K/16K/32K/64K/128K;
- tg128, initially C1/C2/C4; C6/C8 after C4 qualification;
- AR, MTP on mixed and repetitive text, prompt lookup;
- cold/warm startup, peak memory and readiness time;
- baseline, candidate and unaffected control in the same environment;
- at least three repetitions, reported dispersion, exclusive GPU lock and
  memory guard; one conceptual change per experiment where possible.

Store individual latencies, aggregate throughput, acceptance, completed tokens,
launches, GPU busy, page faults and memory bytes. A microbenchmark gain counts
only after surviving the complete request without a profiler. Benchmark records
and private host inventories stay outside Git under [AGENTS.md](AGENTS.md).

## 7. Phases

### Phase A0 — fully offline lab

- public development image pinned by digest and a second control toolchain;
- `gfx1151` compilation, HSACO extraction and ELF/metadata/ISA audit;
- `gfx1151` code-object inventory in rocBLAS, hipBLASLt and dependencies;
- two builds with hashes and normalized diff;
- CI gates for VGPR, SGPR, LDS, spills and code size;
- CPU oracles and tiny models/tensors independent of final hardware.

### Phase A — Strix Halo lab

- hardware/software inventory;
- confirm execution of A0-produced objects;
- bandwidth and WMMA-ceiling measurements;
- common harness for external runtime A, external runtime B and our binary;
- corpus and weights pinned by hash.

### Phase B — correct reference

- GGUF, tokenizer and minimal CPU/oracle;
- correct AR execution even if slow;
- state, snapshot and sampling tests.

### Phase C — C1 text

- native projection, MoE, DeltaNet and QSA kernels;
- CPU contracts/oracles for GDN, sparse QSA, n-gram/PLE, hyper-connections and routing;
- after correct AR and our own profiling: tiled GDN and direct PLE-reader prototypes;
- dense versus complete sparse QSA ablation including quality; partial
  integration is not accepted;
- separate prefill and decode;
- HIP graphs and offline tuning.

### Phase D — MTP and initial concurrency

- predictor, acceptance, rollback and catch-up;
- C2/C4 batching and memory-based admission;
- later expansion to C6/C8 and prompt cache only after C4 qualification.

### Phase E — server and product

- APIs, health/metrics, inspect/verify/ppl/niah;
- reproducible packaging and documentation;
- harden cancellation, driver faults and memory-pressure handling.

The server includes a watchdog, request draining, explicit driver hang/reset
policy and visible degraded state. Every benchmark records power/TDP, clocks
and temperature alongside its software environment.

## 8. Decision gates before building the complete engine

1. Select a public checkpoint and canonical quantization after demonstrating a
   complete budget within 128 GB UMA.
2. Measure external runtime A and external runtime B with the same prompt,
   context, power and completed-token criterion.
3. Confirm the target machine's Linux/ROCm/kernel.
4. Decide whether the internal weight cache justifies startup time and storage.
5. Fix license and MIT reuse policy.
6. Resolve P0 items in `DESIGN_ACCEPTANCE.md` and submit the consolidated
   specification to a second review.
7. Pin canonical repository, commit and license for QSA, DeltaNet, MoE routing,
   MTP, tokenizer and chat template.
8. Demonstrate `gfx1151` paths in chosen ROCm libraries or define independent
   fallbacks.
9. Approve use case and distribution under Qwen Community License 1.0.
10. Confirm QSA semantics and oracle; do not replace it with dense attention on
    the strength of an external performance observation.

## 8.1 Work permitted before hardware arrives

- offline HIP compilation for `gfx1151` with a pinned toolchain;
- ISA, VGPR/SGPR/LDS, spill and code-size inspection;
- GGUF loader, hashes and layout planner implementation;
- CPU oracles and operator/state tests;
- admission, batching, cancellation, snapshot and rollback simulation;
- API, metric, artifact-format and administrative-plane definition;
- candidate microkernels without declaring winners.

Successful compilation or static counts must not become performance claims.
Kernel selection, tuning, UMA, page faults, power, temperature and stability
require actual Strix Halo hardware.

## 8.2 Remote administrative plane

The product has three separate domains: a local engine with no network
exposure, an inference/chat gateway and an administrative plane. Priority
functions are health/versions, validated configuration, service lifecycle,
logs, memory/GTT, temperature/power, models, cache/sessions and benchmark
execution/comparison. An optional chat playground tab may support smoke tests
and request reproduction.

The engine's internal port is never published: use a UDS or restricted
loopback. The gateway uses API keys or OIDC, quotas and auditing. The panel
uses a local agent with explicitly allowed typed operations, WebAuthn or mTLS,
TLS, auditing, CSP and CSRF protection. Prefer VPN for remote ingress.
Destructive actions or restarts show the target and require confirmation.
The chat tab does not inherit administrative privileges and the UI does not
execute arbitrary shell commands.

The GTT section shows configured limit versus consumption and effective
reserve, detected carve-out, `MemAvailable` and residual state. It may prepare
and apply an allowlisted GTT/TTM profile through one-shot boot, preflight,
reauthentication and confirmation. A health check confirms driver/GTT/agent;
on failure, the next boot returns to the previous entry. `amd_iommu=off` and
experimental parameters remain outside the WebUI and require console or SSH.
The playground uses separate origin and cookies from the administrative panel,
with strictly sanitized model output.

## 9. Specification history

| Version | Date | Change |
| --- | --- | --- |
| 0.4 | 2026-09-28 | Official Qwen sources, UMA/GTT policy and external platform study findings: tiled GDN, direct PLE, UMA ring and QSA/PM4 negative controls. |
| 0.3 | 2026-09-28 | Design review: P0, reduced scope, numerical contracts, A0 and privilege separation. |
| 0.2 | 2026-09-28 | Pre-hardware cross-compilation and remote administrative plane. |
| 0.1 | 2026-09-28 | Initial scope, architecture, quality, benchmarks and phases. |
