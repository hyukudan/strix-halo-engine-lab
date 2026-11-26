# Backends

The engine is modular by device, in three layers:

1. **Shared, model-agnostic logic**: graph, session, serve, xport, spec, policies, identities, and
   the arithmetic contracts of every operator (host references in `src/ops`, `include/strix/*_formats.hpp`).
2. **One backend per vendor or device kind** implementing the shared core interfaces below.
3. **Architecture specialisations** inside a backend (gfx1151 WMMA, sm_120 tensor cores,
   AVX-512/AMX, NEON/SVE), selected by capability, never by model.

A model binds the shared cores through policies (layout, decode, epilogue) and adds only what is
truly model-specific. There are no per-model kernel copies.

## Where the shared cores stand today

| Core | Location | Backend-neutral part | HIP-only by construction |
|---|---|---|---|
| Norms | `src/ops/norm/norm.hpp`, `reference.hpp` | `Norm<Traits>`: types, `ReductionOrder`, `lanes(n)`, `launch` via traits; host reference | `norm_device.hpp` |
| Dense Q8_0 x A8 | `src/core/f16_dense_q8a8.hpp` | related operand layout (`weight`, `codes`, `deltas`, `out`, strides, `accumulate`, plus `k_bytes` and `tile0`) | FAST FP16-rounded operands and WMMA arithmetic; `_Float16` vectors, `address_space(1)`, `__device__` |
| Dense f16 tall / tiled | `src/core/f16_dense_tall.hpp`, tiled core | `ColumnTile` geometry, `row_layout.hpp` (`RowLayoutPolicy`, 64-byte rows) | WMMA fragments, `hip_runtime.h` |
| Trellis linear | `src/core/backend.hpp` | `DeviceWeights`, `linear_split_count`, `resolve_linear` (plan) | `hipError_t`, `hipStream_t` in every launch signature |
| Integer math | `src/core/integer_math.hpp` | `product`, `affine`, `iq4`: host and device (`STRIX_INT_HD`) | none |
| Grouped MoE | `src/moe_core/grouped_core.hpp` | the policy split (`WeightDecode`, `ExpertLayout`, `InputTransform`, `Epilogue`) and `Item` | `Half16`/`Float8` ext vectors, WMMA accumulators |
| Attention | `src/attn_core/dense_flash_prefill.hpp`, `sparse_flash_keysplit.hpp` | the policy contract (`begin`, `segments`, `key_row`, `value_row`, `query`, `out`), `masks.hpp`, `online_softmax.hpp` | `__device__` policies, `hip_runtime.h`, WMMA tiles |
| Weight decode | `include/strix/ds41_formats.hpp`, `moe_core/exl3_trellis.hpp` | block geometry, host decoders, declared product order (32 lanes + butterfly) | device decode loops |
| Device declaration | `include/strix/device/declaration.hpp` | `Vendor`, `Kind {gpu, npu, cpu}`, cpu `cores`, `byte_cap` | `hip_gate.hpp`, enumerators per vendor |

The rule for new work: the contract (operand view, operation order, policy hooks) lives in a header
with no vendor include; the vendor body lives in the backend tree.

## The backend interface

The operations below describe the backend contract. Each one is defined by its **arithmetic contract**
(operation order, rounding points), so any two backends that claim the same contract give the same
bits, and the host reference is the oracle.

### Memory and stream model

- `DevicePtr`: an address in the backend's memory. GPU: device or host-coherent memory; CPU: a plain
  host pointer (upload is a copy or a mapping of the model file).
- `Stream`: an in-order queue. Operations enqueue; completion is observed by `record`/`wait` events.
  Operations never allocate, transfer or synchronise; the caller owns workspace (256-byte aligned).
- Workspace sizes are queries (`*_workspace_bytes`) resolved at plan time, as in `linear_workspace_bytes`.
- Plans are resolved once (`resolve_*` -> execution record) and launched many times; a plan fixes the
  split count and therefore the numerics.

### Operations

| Group | Operation | Contract |
|---|---|---|
| Activation quant | `quantize_a8(x, width) -> codes, deltas, sums` | groups of 32, `delta = amax/127`, `rint`, clamp +-127, tiny/zero group -> delta 1 |
| Integer decode products | Q8_0, IQ2_XXS, MXFP4 x A8 | per-group `float(int dot) * (sw * sa)` (`integer_math::product`); groups strided over 32 lanes, butterfly; `accumulate` adds once after the product |
| Affine decode product | Q2_K x A8 | separately round `d * float(dot)` and `dmin * float(minimum)`, then subtract and multiply by the activation delta; groups strided over 32 lanes, butterfly |
| Floating / trellis GEMV/GEMM | f16, bf16, trellis (exllamav3 format) | the selected plan declares operand rounding, accumulation and split order; the integer-dot contract does not apply |
| Grouped MoE | `grouped(items, WeightDecode, ExpertLayout, InputTransform, Epilogue)` | the dense contract per expert; split partials summed in plan order |
| Attention prefill | `flash_prefill<Policy>(args, heads, workspace, splits)` | online softmax of `online_softmax.hpp`, masks of `masks.hpp`, key-split merge in split order |
| Attention decode | `sparse_flash_keysplit<Policy>` | same, one query row per head group |
| Norms | `Norm<Traits>::run` | `ReductionOrder::lane_strided_fma_lds_halving` with `Traits::lanes(n)` |
| Weight decode | `decode_<fmt>(block) -> codes, scales` | host decoders of `ds41_formats.hpp` (Q8_0, Q2_K, IQ2_XXS, MXFP4) and the EXL3 trellis codebook |

### Capability and selection

A backend exposes `caps()` and `select(caps) -> Kernels`, a table of entry points. Specialisations
register a table with only the entries they implement; gaps are filled from the backend's
reference. Selection is by capability bits, never by model name, and every specialised entry must
match the reference bit for bit before it registers.
CPU selection returns a complete table by value, so later selections cannot modify a retained
table. It selects scalar, AVX2, AVX-512F+BW or NEON according to the usable feature set. x86
checks CPUID and OS vector state. NEON availability follows the build target. The operator
table currently binds every entry to the portable reference. AMX and SVE have no operator
entries and use the selected lower capability level when available.

## CUDA NVFP4 dense operator

The [shared NVFP4 core](../engine/kernels/cuda-nvfp4.md) provides a vendor-free
operand/plan contract, scalar host oracle, explicit host/device activation
quantizer, and SM120 native FP4 GEMM and small-row decode specialisation.
Its FP32 MMA accumulation is a declared tolerance-qualified arithmetic identity;
it does not register as an exact replacement for an existing dense identity.
Caller-owned workspace and capability selection follow the memory/stream contract
above. Model adapters and complete-workload qualification remain pending.

## Experimental Intel, Vulkan and Metal backends

- **Intel SYCL / Level Zero** provides operator references and host-managed model
  references when built with oneAPI. The dedicated Gemma 4 E4B Q8_0 decode and
  serve executables offer `--backend intel` without a drafter. That path uses
  synchronous transfers for operator inputs and results and a resident cache for
  Q8_0 weights. The Hy4 reference CLI exposes a hybrid Intel projection path.
  Both model references are pending hardware qualification.
- **Vulkan and Metal** provide RMSNorm operator references. Each call stages the
  input and waits for completion. These backends have no model execution plan.
  The Vulkan adapter uses the Linux Vulkan loader.

The shared `strix-serve --backend` interface offers HIP and CUDA.

## CPU backend

The `STRIX_BACKEND_CPU` option (default OFF) builds the shared CPU operator library and its
fixtures. `src/cpu/core` owns the interface, exact portable reference and capability selection.
The Qwen CUDA build also links this library for its
tiered routed-expert executor. That executor uses a worker pool, a separate AVX2 path
and exp lookup tables to reproduce the target device arithmetic. The scalar operator
table and the dedicated model host references have their own dispatch paths.

- **Memory**: host memory is the device memory. Weights are used in place from the mapped model
  file (Q8_0 blocks are 34 bytes, so rows are read unaligned; a repack to 64-byte rows follows
  `row_layout.hpp` when a specialisation needs it).
- **Streams**: the scalar reference runs each call synchronously on the caller's thread. A threaded
  specialisation partitions output rows across `Stream::threads` workers (the `cores` declared for
  the `cpu` unit in the device declaration); the per-output operation order does not change with
  the thread count, so the bits do not either.
- **Numerics**: compiled with contraction off (`-ffp-contract=off`, `/fp:precise`). Integer dots are
  exact in int32, so a SIMD kernel only has to keep the declared float order: per-group product,
  group g into lane g % 32, then the butterfly.
- **Implemented operators** are `quantize_a8`, `gemv_q8_0_a8` (the `Q8a8View` has a related layout to the HIP
  dense view, but omits `k_bytes` and `tile0` and uses a different arithmetic contract) and
  RMSNorm bound to the shared `Norm<Traits>`; tests in `tests/cpu_backend_host_tests.cpp` compare
  them bitwise with `ds41::quantize_a8`, `ds41::product_q8_0` and `ops::norm::reference`.
  A8 width and GEMV `in` must be multiples of 32; every table entry rejects other widths
  with `std::invalid_argument` before writing output. Zero width is allowed. Callers supply
  sufficient buffers and strides, with `x_stride` a multiple of 32. Valid calls allocate no memory.
  Tests cover retained-table selection, width rejection, padded strides, tiny groups, half
  subnormals, cancellation, and norm thresholds/tails/aliasing. Synthetic descriptors exercise
  the full portable table at every selectable capability level. The explicit raw Q8_0 weights
  registry still advertises its scalar arithmetic implementation.
- **Next**: AVX-512 VNNI and NEON SDOT `gemv_q8_0_a8` (same bits); Q2_K / IQ2_XXS / MXFP4 decode
  products from the `ds41_formats` contracts; grouped MoE through the existing policy split;
  attention via the `sparse_flash_keysplit` policy contract with host policies; AMX for prefill GEMM.
- **Role**: host lane and fallback device (`Kind::cpu`), small-batch decode where the memory
  bandwidth suffices, and a second exact implementation for identity checks of the GPU backends.

## Moving a HIP-only core toward the interface

1. Split the contract (view struct, policy hooks, operation order) into a header without vendor
   includes.
2. Keep the HIP body where it is; the CUDA and CPU backends implement the same contract.
3. Add a host test that runs the CPU reference against the existing host reference bitwise.
