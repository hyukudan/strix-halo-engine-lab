# The CUDA backend

All CUDA code lives in one tree with the same layering as the HIP side: model-agnostic host layers (graph, serve,
session, spec, xport, policies, identities) are shared by both backends; each vendor backend implements the same
core interfaces (dense, grouped MoE, attention, norms, weight decode); architecture specialisations sit inside the
backend and are chosen by capability, not by model.

| Path | Holds |
|---|---|
| `src/cuda/core/` | model-agnostic device helpers and kernels |
| `src/cuda/arch/` | capability queries (`capability.hpp`); `arch/sm120/` Blackwell-only paths (none on main yet) |
| `src/cuda/models/ds41`, `qwen`, `glm`, `minimax` | what is still model-specific |
| `include/strix/cuda/` | public headers (`strix/cuda/<name>.hpp`, formerly `strix/cuda_<name>.hpp`) |
| `cmake/StrixCudaBackend.cmake` | `strix_cuda_core` (interface target linked by the CUDA model libraries), core source lists |

Entry points stay in `apps/cuda_*` and tests in `tests/cuda/`; the GLM producer keeps its own CMake project
(`src/cuda/models/glm/producer`) and the MiniMax prefill its build scripts (`src/cuda/models/minimax/*.bat`).
`src/glm/rtxpf/import` (the receiving side of the GLM state import) is host code and stays with GLM.

Session memory is planned by the shared host policy `include/strix/serve/session_reservation.hpp`: each request
reserves prompt + max_tokens + the engine's slack (rounded to a granule, capped at the context), and admission
reuses, builds, re-sizes or releases sessions from the free memory. The CUDA Qwen server binds it through
`apps/cuda_qwen_serve_policy.hpp` (`--session-reserve request|max`, `--session-granule N`). Its serve bench kit is
`scripts/bench/cuda/qwen_serve_arms.sh` (strict arm order, one card) with `qwen_serve_bench.py` (cold and warm
prefill, C1 references, concurrency arms with `same_as_C1`).

## Phase 1 (layout, bitwise)

Files were moved, not rewritten. The SASS gate is `scripts/bench/cuda/sass_compare.py
--base <old build> --new <new build>`. It compares architecture/function pairs, requires nonzero CUDA coverage,
and rejects missing artifacts, disassembly failures and symbol collisions. Artifact moves require explicit
relative-path `--rename OLD=NEW` entries; helper or anonymous-namespace symbol changes require exact
`--symbol-rename OLD=NEW` entries, also applied to call targets. Earlier totals from the lossy symbol gate do
not establish this gate; native builds must be checked again. Identical helpers that several files carried
are defined once:

| Helper | Was in | Now |
|---|---|---|
| `cp_async16`, `cp_async4`, `cp_async_commit`, `cp_async_wait<N>` (no clobber) | ds41 `dense_fast`, `prefill_q8`, `prefill_kernels` | `core/ptx.cuh` `core::ptx` |
| `mma_s8` (m16n8k32 s8, `uint4` A) | ds41 `core_mma`, `dense_fast`, `prefill_q8` | `core::ptx` |
| `smem_addr`, `ldsm_x4` | ds41 `fast_attention`, `fast_index` | `core::ptx` |
| `cp_async16` ("memory" clobber) | ds41 `moe_v2`, `router_v2` | `core::ptx_mem` |
| `cp_async16` zero-fill, `cp_async_commit`, `cp_async_wait_all` ("memory") | ds41 `fast_attention`, `fast_index` | `core::ptx_mem` |
| `butterfly` (warp sum, fixed order) | `ds41_cuda_common.cuh`, `ds41_moe_device.cuh` | `core/reduce.cuh` |
| `better` (top-k order) | ds41 `core_moe`, `prefill_kernels`, `router_v2` | `core/reduce.cuh` |
| `block_exclusive_scan` | ds41 `moe`, `parts_host` | `core/reduce.cuh` |
| `dot_float` (exact int to float) | ds41 `dense_fast`, `prefill_q8` | `core/reduce.cuh` |
| compute capability >= 9 cache (PDL, clusters) | ds41 `core_mma`, `core_norm` | `arch/capability.hpp` |

Same job, not the same text (left in place, to be settled with a SASS check when their core is built): the warp
sums `warp_sum` of ds41 `attention` / `fast_attention` and Qwen `moe`, `butterfly` of ds41 `dspark` (no unroll
pragma, explicit width), the BF16 conversions of Qwen `gdn` / `hc` / `qsa` / `plemtp` / `mtp_draft` and MiniMax
(some with a NaN case, some without), the per-file `check()` of five Qwen files (different message prefixes),
`cp_async_wait_all` of ds41 `moe_v2` (`cp.async.wait_all`, not `wait_group 0`), GLM `h128` (`native.cu` vs
`projection.inl`).

## Phase 2 plan: shared CUDA cores behind the HIP interfaces

Each step extracts one family into `src/cuda/core/<family>/` behind the interface the HIP core already exposes,
keeps the current entry points as thin wrappers (old signatures), and lands only with a byte-for-byte gate on the
operator, its partials and the logits / KV state (1 and 8 rows, 8K and 32K, repeated and disjoint routes, graph
capture and replay). Numerical identity is part of the plan (FMA or not, FTZ, div/sqrt, AMD tables, accumulator
type, top-k ties, split-K, rounding): a core selects only between variants of the same identity, by capability and
shape. Order:

1. **Dense IMMA tile** (`core/dense/`, mirrors `src/core/linear_plan.hpp`, `tiled_core.hpp`, `integer_plan.hpp`):
   `ds41_core_mma.cu` (tiled Q8_0 x A8) first; the DeepSeek and Qwen CMake lists then name the core library.
2. **Common numerics and norms** (`core/norms/`): `ds41_cuda_common.{cuh,cu}` (device, workspace, stream, event
   helpers become `core/runtime` pieces), `ds41_core_norm.cu`, Qwen `hc` with the AMD-identical math
   (`include/strix/cuda/amd_math.cuh` is already in core), MiniMax Gemma norms; one interface, separate arithmetic
   families.
3. **Remaining dense** (`core/dense/`, mirrors `src/core/f16_dense_q8a8.hpp`, `gemv_rows.hpp`,
   `activation_quant.hpp`): ds41 `dense_fast`, `prefill_q8`; Qwen `dense` + `transcode`; GLM `projection.inl` +
   `row_mma.cuh`; MiniMax `fp32_identity.cu` (keeps the `.inc` it shares with HIP).
4. **Grouped MoE** (`core/grouped_moe/`, mirrors `src/moe_core/grouped_core.hpp`, `grouped_policy.hpp`,
   `route.hpp`, `decode.hpp`, `device_plan.hpp`): ds41 `moe`, `moe_device.cuh`, `moe_grouped`, `moe_fast`, `moe_v2`,
   `router_v2`, `core_moe`; Qwen `moe` (already includes the HIP route storage `.inc`); GLM `native.cu` +
   `cutlass_path.inl` + `c1_layout.cuh` (expert order and epilogue kept); MiniMax calls its FP32 variant.
5. **Trellis decode** (`core/trellis/`, mirrors `src/core/trellis_codec.hpp`, `trellis_format.hpp`,
   `src/moe_core/exl3_trellis.hpp`): `core/trellis.cu` split into decode / linear; the GLM decoder and staging
   (`trellis.cuh`, `native.cu`) join with their own descriptor once bit equality is shown.
6. **Attention pieces** (`core/attention/`, mirrors `src/attn_core/flash_core.hpp`, `online_softmax.hpp`,
   `masks.hpp`, `select.hpp`, `index_score_*.hpp`, `sparse_flash_*.hpp`): ds41 `fast_attention` (sparse flash),
   `fast_index` (index score), the selection of `attention`; GLM `dsa`, `kda`; Qwen `qsa`, `gdn`; MiniMax MSA
   primitives. KDA, GDN, MLA and MSA stay distinct kernels under one tile / softmax / mask vocabulary; attention state
   export and replay policies stay model adapters.
7. **Runtime** (`core/runtime/`): buffer, graph and workspace lifetimes out of the ds41 / Qwen prefill drivers and
   the GLM producer runtime; the ds41 expert streamer, hot site and tier site behind one residency contract
   (tierstream kept). Device abstraction through `include/strix/device` (one runtime header, CUDA and HIP
   implementations).
8. **Build**: the GLM producer project and the MiniMax scripts become targets of the common CMake; duplicates are
   removed only when no adapter uses them.

The first Blackwell-only code (`arch/sm120/`, e.g. block-scaled FP4 for routed experts) enters as a policy of the
grouped MoE core of step 4, selected through `arch/capability.hpp`.

## Decode graphs: QSA inside the step graph, draft chain as one replay (Qwen, bitwise)

The 12 QSA attention cores of a decode window used to run as eager launches between 13 graph segments, because their
grids and arguments depend on the positions and the host pooled-key counts. `qwen_qsa::prepare_multi` now builds the
window's `SegmentMap` on the host (validation, pooled ranges, score waves), and `layer_core_device` launches the same
multi-segment kernels reading that map from device memory with a geometry fixed per segment-rows layout (pool and score
kernels walk their blocks and waves grid-stride; score runs past a row's selected count return at once). A window is one
captured graph; each replay first copies the 12 maps from pinned memory. The MTP draft chain of one session is captured
the same way (positions and the pending token from pinned memory, maps 12..18). Kernel bodies and their operation order
are unchanged, so the logits are bit-identical. Switches: `STRIX_QRTX_QSA_GRAPH=0` (cut graphs, eager cores) and
`STRIX_QRTX_DRAFT_GRAPH=0` (eager draft chain), both default on.
The C1 draft chain runs as two graph segments [0, a) and [a, depth): the second only when no draft of the first fell
below min_p (the verify would discard its steps). Kept drafts and verified tokens are unchanged; `a` comes from a
histogram of kept drafts (shared chooser `include/strix/mtp_draft_split.hpp`, expected steps a + P(kept > a) (depth - a
+ 0.25)). `STRIX_QRTX_DRAFT_SPLIT=0` = one segment, `N` = fixed a. Measured (#680, 3 ABAB pairs): prose C1 +7.3 %, json
+1.9 %, edit +-0.2 %, window 16.1 -> 14.5 ms at 1.86 tokens/window.

Served decode by prompt class: `scripts/bench/cuda/qwen_serve_classes.py` (prose / edit / json, C1 / C4 / C8, cold and
warm passes, ids of every request) with `scripts/bench/cuda/qwen_classes_prompts.json`, and strict ABAB server arms with
`scripts/bench/cuda/qwen_serve_abab.sh` (server, model flags, arms and client are parameters; GPU clock samples per arm).
Summarise arms with `scripts/bench/cuda/qwen_serve_summary.py OUT`: C1 means only over prompts run in both passes, C1
requests shorter than 16 tokens (the model ends early: prose slice 0 stops after 1 token) are listed and skipped, cold and
warm side by side per arm. Comparing a 2-prompt warm mean that contains a 1-token request with an 8-prompt cold mean is
what made warm look slower than cold in #680.
