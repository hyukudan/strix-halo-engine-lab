# Strix Halo Engine Lab

An experimental inference-engine research repository targeting Linux, ROCm/HIP and AMD Strix Halo-class unified-memory systems, initially **gfx1151**. It develops its own mathematical contracts, independent numerical oracles and explicit hardware qualification gates.

**This is an engineering lab, not a production inference server.** Portable tools and contract tests are available; HIP kernels, graph execution and multi-node transport remain experimental. Compilation, host tests and device discovery do not prove full-model correctness, GPU execution or performance. Consult the [qualification ledger](docs/qualification-ledger.md) for current evidence and blockers.

## Goals and architecture

The engine aims to optimize end-to-end cost per committed token while preserving model semantics, predictable physical-memory use and recoverable state. Priorities are correctness, sustainable unified-memory residency, concurrency-one decode latency, small-concurrency throughput and long-context capacity. Multi-node work must preserve the single-node path.

The architecture separates:

- **Model and memory contracts:** defensive GGUF parsing, exact tensor/layout identities, conservative residency and explicit admission budgets.
- **Execution:** packed operators, a compiled serial DAG, versioned inputs/outputs, physical scratch arenas and persistent session state.
- **State and optimization:** recurrent/KV state, snapshots, prefix lineage, speculative verification and opt-in graph/low-precision paths.
- **Transport:** bounded buffers, epochs, leases, explicit commit, consumer completion and receive reposting across independent links.
- **Qualification:** independent numerical references, exact artifact/runtime bindings and complete-workload gates.

The current model-specific geometry and semantic work focuses on Qwen3.8-Flash-Next; this is not a claim of general GGUF architecture support. C1 means concurrency one. Eager dispatch, graph replay and integer projection are distinct experimental modes with separate acceptance requirements.

See the [architecture guide](docs/README.md), [memory/residency contract](docs/architecture/02-memory-and-residency.md), [execution model](docs/architecture/04-execution-and-graphs.md) and [numerical rules](docs/architecture/06-quantization-and-numerics.md).

## Current implementation

| Area | Available pieces | Remaining qualification boundary |
| --- | --- | --- |
| Host tools | UMA budget planner, defensive little-endian GGUF v3 inspection, tensor manifests and read-only Linux probe | Actual limits, complete checkpoint coverage and observed physical peaks |
| Portable correctness | Semantic oracles, checked accounting, state/ownership tests and serial runtime contracts | Native complete-model logits, tokens and state fidelity |
| HIP execution | Experimental packed projection, GDN, MoE, QSA/PLE, norm/RoPE/KV and serial DAG launchers | Genuine gfx1151 execution, numerical fidelity and resource behavior |
| Graph and low precision | Opt-in C1 device-input capture/replay, integer projection and tiled integer LM-head eager adapter | Preparation-inclusive costs and full-model quality; integer LM-head remains eager C1 only |
| Speculation and state | Greedy/sampled verification contracts, snapshot/replay and prefix lifecycle infrastructure | Qualified proposal/target semantics and native fresh/resumed correspondence |
| Multi-node transport | TCP/direct-stream diagnostics, dual-link scheduling, C1 host-copy boundaries and software-tested DMA-BUF provider/driver contracts | Real two-peer GPU visibility, continuity, recovery and measured independence |
| Campaign tooling | Private commissioning preparation and host-side numerical evidence checks | Independently bound target execution and complete-workload evidence |

The [implementation history](IMPLEMENTATION_STATUS.md) records earlier slices; the [current ledger](docs/qualification-ledger.md) is the cross-cutting summary. An earlier slice's missing-feature list may have been superseded by later experimental code. Neither history nor a successful offline HIP build is native qualification.

## Build and test the host tools

Requirements: **CMake 3.24+**, a **C++20 compiler**, and Ninja for the commands below. The base core has no external library dependencies. Python 3 enables additional oracle/campaign tests when discovered. Linux is the native target; WSL can exercise host-side work but does not establish target GPU support.

From the repository root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Warnings are errors by default (`STRIX_WARNINGS_AS_ERRORS=ON`). HIP is disabled by default (`STRIX_ENABLE_HIP_RUNTIME=OFF`), so this workflow does not require a GPU or model download. Optional Python tests depend on interpreter discovery; inspect the CTest inventory when comparing runs.

### Host-side examples

```bash
./build/strix-memory-plan --tokens 32768 --sessions 1 --host-gib 128 --resident-gib 68 --baseline-gib 2 --reserve-gib 18
./build/strix-platform-probe
./build/strix-inspect-gguf /path/to/model.gguf
./build/strix-model-manifest /path/to/model.gguf
```

| Tool | Output and limits |
| --- | --- |
| `strix-memory-plan` | JSON memory breakdown and admission decision for the pinned model contract; uses explicit budget inputs, not live GPU allocation |
| `strix-platform-probe` | Read-only Linux RAM, KFD, ROCm and USB4/USB4STREAM discovery; presence does not prove runtime readiness |
| `strix-inspect-gguf` | Metadata, tensor geometry, alignment/range checks and storage inspection without loading weight payloads |
| `strix-model-manifest` | Tensor roles, physical bytes, offsets and conservative residency candidates; unknown tensors remain visible |

The memory values above are illustrative. Supply budgets appropriate to your approved workload; a planner admission result is not a reservation or an observed memory peak. `strix-memory-plan` returns 0 for admitted, 2 for rejected and 1 for invalid input. Its `--no-indexer` and `--rollback-full` options support explicit budget exploration; provisional model terms remain subject to their contracts.

See [GGUF_READER.md](GGUF_READER.md) for supported storage formats and defensive limits. Model files and private probe/experiment outputs remain outside Git.

## Optional HIP development build

Use a separately configured Linux environment with the ROCm development toolkit, HIP headers, compiler and `amdhip64` library. The optional target is fixed to gfx1151 in CMake.

```bash
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSTRIX_ENABLE_HIP_RUNTIME=ON -DBUILD_TESTING=ON
cmake --build build-hip
```

`ROCM_PATH` can help CMake locate a non-default toolkit installation. The host build and HIP build have separate prerequisites. A successful build establishes compilation only; review the relevant native campaign requirements before running GPU, model or driver experiments. See the [hardware campaign preparation](docs/implementation-2026-10-12-hardware-campaign-preparation.md) and [native numerical campaign](docs/implementation-2026-10-12-native-numerical-campaign.md).

Raw-token decode tooling and full C1 graph code are experimental execution infrastructure, not a qualified chat interface. Tokenizer/source bindings, complete tensor coverage, numerical replay, memory behavior and lifecycle checks remain independent gates. No stable public serving API or administrative panel is promised.

## USB4 and multi-node work

Every physical link is a separate resource with its own epoch, leases, pools, completion accounting and health. Two cables do not imply twice the usable bandwidth. Directional affinity, independent flows and optional large-message striping depend on qualified paths and complete transfer cost.

The repository contains CPU TCP/direct-stream diagnostics and experimental GPU-owned DMA-BUF transport infrastructure. A working network link or character device does not establish GPU zero-copy transport, external-DMA visibility or distributed model inference. Transport completion, semantic commit, consumer completion and receive reposting must be demonstrated separately.

Read the [multi-node design](docs/architecture/12-multi-node-and-usb4.md) and [direct GPU transport contract](docs/architecture/16-direct-gpu-transport.md). Driver/module staging is an explicit experimental operator workflow; ordinary builds do not install or load a driver. Published benchmark claims require project-owned, reproducible qualification evidence, not negotiated link rates or theoretical kernel throughput.

## Repository and reading guide

| Path | Purpose |
| --- | --- |
| `include/strix/`, `src/` | C++ contracts, host core and optional HIP implementations |
| `apps/` | Inspection, planning, execution-validation and experimental native tools |
| `tests/` | Portable and optional native tests |
| `tools/oracles/` | Independent Python semantic/numerical references |
| `lab/` | Operator, transport and qualification experiments |
| `docs/architecture/` | Normative runtime decisions |
| `docs/qualification-ledger.md` | Current claim/evidence boundaries |

Additional project documents:

- [RESEARCH.md](RESEARCH.md): research scope and evidence criteria.
- [SPEC_DRAFT.md](SPEC_DRAFT.md): initial technical specification draft.
- [EVIDENCE.md](EVIDENCE.md): project claims, evidence and confidence.
- [DECISIONS.md](DECISIONS.md): accepted, pending and rejected decisions.
- [DESIGN_ACCEPTANCE.md](DESIGN_ACCEPTANCE.md): design acceptance and traceability.
- [MEMORY_BUDGET.md](MEMORY_BUDGET.md): proposed UMA/GTT budgeting and admission guards.
- [SOURCES_LEDGER.md](SOURCES_LEDGER.md): canonical sources, licenses and provenance gates.

## Contribution and publication rules

Read [AGENTS.md](AGENTS.md), the architecture guide and the current ledger before changing behavior. New implementations follow public mathematical contracts and independent oracles. Maintain accurate attribution and notices for any actually adopted components; do not copy or reconstruct unrelated runtime code through observational research.

Keep comparative evaluations, external benchmark records, private review correspondence and source-identified inspection artifacts outside Git. Public documentation describes this engine's own contracts and evidence. Numerical proposals remain drafts until compiler, native execution, model fidelity and complete-workload gates pass.

Record project-owned observations with their evidence level; update the specification and decision record when an accepted design changes. Hypotheses, portable fixtures, compilation and native measurements must remain distinguishable. Hardware or model tests need explicitly configured artifacts and target environments.
