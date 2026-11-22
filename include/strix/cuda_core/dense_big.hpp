#pragma once
// Large-M dense GEMM (prefill chunks of up to a few thousand rows) on tensor cores.
// Home: shared (backend core CUDA; arch specialisation: mma.sync m16n8k16 bf16/f16, sm_80+; nvfp4 expansion needs sm_120a).
//
// EXACT CLASS: for every supported weight format the output of dense_big_gemm is BITWISE equal to dense_rows / the nvfp4 decode
// GEMV for the same row (rows 1..32, see dense.hpp / nvfp4.hpp), because every output element keeps the documented reduction
// order of the decode kernels: the same K segments (mma3 bf16: the 8 warp partials of src/cuda/core/dense/dense_mma.cu; q8_0 the
// same with the per-32-block fold tot += d * part; nvfp4: the split-K ranges of nvfp4_split_k(out, in)), the same k permutation
// inside a k16 step, the same instruction shape and the same term order, then the segment partials added in ascending order.
// Only the tiling (rows ride the N dimension, 128 out x 64..128 rows per CTA) and the operand staging differ. The activation
// is split ONCE per layer input into bf16 planes (hi, mid, lo; `terms` 3 = exact, 2/1 = FAST mma2/mma1) or one f16 plane
// (nvfp4), shared by every GEMM that reads the same x. Weights other than bf16 are expanded per call into a bf16/f16 scratch
// (q8_0: int8 values exact in bf16 + FP32 scales; nvfp4: e2m1*e4m3 exact in f16), or once into a resident operand cache (below).
#include "strix/cuda/core/nvfp4_tile.hpp"
#include "strix/cuda_core/operand_cache_plan.hpp"
#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

// Scratch (planes, expanded weights) of one stream context. nullptr in every call = the module default context (single stream).
// Concurrent streams need one context each; a context binds to its first device.
// The implicit context is isolated by host thread/device. Drain its stream before release.
[[nodiscard]] bool dense_big_fast8_supported();
// FAST4 (W4A4, identity nvfp4-tile-mma-f32-v1 of strix/cuda/core/nvfp4_tile.hpp): nvfp4 weights x per-row NVFP4 activations on the
// block-scaled e2m1 mma (sm_120). Bit-changing and quality-gated like FAST8; never selected implicitly.
[[nodiscard]] bool dense_big_fast4_supported();
// Operand class of a prefill (selects the activation split and the resident operand kind).
enum class BigClass : std::uint8_t { exact, fast8, fast4 };
struct BigCtx;
[[nodiscard]] BigCtx* dense_big_ctx_create();
void dense_big_ctx_destroy(BigCtx* ctx);

// Activation planes made by dense_big_split (device memory owned by the module; valid until the next split).
struct BigAct {
    const void* planes = nullptr;  // [terms][rows_pad][in] 16-bit (bf16 hi, mid, lo | f16 permuted)
    std::uint32_t rows = 0, rows_pad = 0, in = 0;
    int terms = 3;
    bool f16 = false;  // f16 plane (nvfp4 weights) instead of bf16 planes
    bool f8 = false;   // FAST/TURBO: e4m3 plane + row scales (nvfp4 weights only; dense_big_gemm then runs the e4m3 x e4m3 mma kernel)
    bool f4 = false;   // FAST4: NVFP4 X tile + row scales (nvfp4 weights only; dense_big_gemm then runs the e2m1 tile GEMM)
};

enum class BigEpi : std::uint8_t {
    none,      // y = v
    residual,  // y = aux + v (aux may alias y)
    silu_mul,  // y = silu(aux) * v   (aux = the gate GEMM output of the same shape; gate/up as two GEMMs)
    round_bf16,  // y = BF16-rounded v widened to FP32 (BF16 weights; three activation terms, or one for a BF16-exact x).
};

// true when the large-M path serves this weight (bf16 / q8_0 with in % 32 == 0, nvfp4 with in % 128 == 0, out % 16 == 0).
[[nodiscard]] bool dense_big_supported(const DenseWeight& w);
// Splits x (FP32 [rows][in]) into the planes weights of format `fmt` need. terms: 1..3 (bf16 and q8_0 only; nvfp4 ignores it).
// rows <= max rows given to dense_big_reserve. `in` % 32 == 0.
cudaError_t dense_big_split(const float* x, std::uint32_t rows, std::uint32_t in, WeightFormat fmt, int terms, BigAct* out,
                            cudaStream_t stream, BigCtx* ctx = nullptr, bool f8 = false);
// FAST4 split: x (FP32 [rows][in], 16-byte aligned, in % 128 == 0) -> NVFP4 X tile + per-row scales (dense_big_fast4_supported()).
cudaError_t dense_big_split4(const float* x, std::uint32_t rows, std::uint32_t in, BigAct* out, cudaStream_t stream, BigCtx* ctx = nullptr);
// The same from BF16 rows (bit patterns, [rows][in], 16-byte aligned): bit for bit dense_big_split4 of their exact FP32 widening.
cudaError_t dense_big_split4(const std::uint16_t* x_bf16, std::uint32_t rows, std::uint32_t in, BigAct* out, cudaStream_t stream,
                             BigCtx* ctx = nullptr);
// FAST4 split of rmsnorm(x, gain, eps) without the f32 normalised rows: the same BigAct, bit for bit, as rmsnorm(x -> xn) followed by
// dense_big_split4(xn). cudaErrorInvalidValue when the width does not fit the fused producer (dense_big_norm_split4_fits); the caller
// then runs rmsnorm + dense_big_split4.
[[nodiscard]] bool dense_big_norm_split4_fits(std::uint32_t in);
cudaError_t dense_big_norm_split4(const float* x, const float* gain, float eps, std::uint32_t rows, std::uint32_t in, BigAct* out,
                                  cudaStream_t stream, BigCtx* ctx = nullptr);
// The same for the GDN value-head gated RMSNorm: bit for bit gdn_chunk_norm(gain, z -> x in place) followed by dense_big_split4(x),
// with x left un-normalised. Same width rule (dense_big_norm_split4_fits); gain = the layer's norm_gain (128 values).
cudaError_t dense_big_gated_norm_split4(const float* x, const float* z, const float* gain, float eps, std::uint32_t rows, std::uint32_t in,
                                        BigAct* out, cudaStream_t stream, BigCtx* ctx = nullptr);
// FAST4 fused gate/up: y[r][c] = silu(gate_r[c]) * up_r[c] in ONE GEMM (gate and up 16-row tiles of the same channels side by side
// in each warp, SwiGLU in registers; same rounding steps as BigEpi::silu_mul on two GEMMs). a must come from dense_big_split4.
cudaError_t dense_big_glu4(const DenseWeight& gate, const DenseWeight& up, const BigAct& a, float* y, cudaStream_t stream,
                           BigCtx* ctx = nullptr);
// The same with BF16 hidden rows (FAST nvfp4-tile-h16-v1): y_bf16[r][c] = bf16_rn(the FP32 value the overload above stores).
cudaError_t dense_big_glu4(const DenseWeight& gate, const DenseWeight& up, const BigAct& a, std::uint16_t* y_bf16, cudaStream_t stream,
                           BigCtx* ctx = nullptr);
// y[r][o] = epilogue(W x_r), r < a.rows. The planes must come from dense_big_split with the same format class as w.
cudaError_t dense_big_gemm(const DenseWeight& w, const BigAct& a, float* y, const float* aux, BigEpi epi, cudaStream_t stream,
                           BigCtx* ctx = nullptr);

// Small-output tile of the same three-term BF16 chain. Supports none and round_bf16.
cudaError_t dense_big_bf16_compact(const DenseWeight& w, const BigAct& a, float* y,
                                  BigEpi epi, cudaStream_t stream, BigCtx* ctx);
// Grows the module's scratch (planes, expanded weights) once, outside any measured region. max_in / max_out_in as elements.
cudaError_t dense_big_reserve(std::uint32_t max_rows, std::uint32_t max_in, std::uint64_t max_weight_elems, int max_terms,
                              BigCtx* ctx = nullptr);
// Release the scratch.
void dense_big_release(BigCtx* ctx = nullptr);
// Diagnostics: bytes currently reserved.
[[nodiscard]] std::uint64_t dense_big_reserved_bytes(const BigCtx* ctx = nullptr);

// RESIDENT EXPANDED OPERANDS (strix/cuda_core/operand_cache_plan.hpp). Without a cache, dense_big_gemm expands an nvfp4 / q8_0
// weight into its MMA operand on every call (per chunk, per slot, per request). With a cache attached, the first call for an
// admitted weight expands it once into a shared arena (same kernel, same bytes) and records an event; every later call on any
// context waits on that event and reads the arena. Outputs are BITWISE identical with and without the cache (identical operand
// bytes into the identical kernel); only the expansion launches and their DRAM traffic go away. Not used while the calling stream
// is capturing a graph (the call then expands into the context scratch as before). Weights must stay alive and unchanged while
// the cache lives (a reused key with another shape is rejected).
struct BigOperandCache;
struct BigCacheReport {
    std::uint64_t arena_bytes = 0, wanted_bytes = 0, budget_bytes = 0;
    std::uint32_t admitted = 0, rejected = 0;
    bool f8 = false;
    bool f4 = false;
};
[[nodiscard]] BigOperandCache* dense_big_cache_create();
void dense_big_cache_destroy(BigOperandCache* cache);  // the caller drains every stream that used it first
// Plans (first fit in the given order under operand_cache::budget(mode, free, total, cap_bytes)) and allocates the arena once on
// the current device. cls selects the operand class: fast8 = e4m3, fast4 = the e2m1 W tile (nvfp4), exact = f16 (nvfp4) / bf16 +
// scales for q8_0; classes the mode does not admit and bf16 weights (no expansion) are skipped. A second call returns the
// first report.
cudaError_t dense_big_cache_reserve(BigOperandCache* cache, const DenseWeight* weights, std::size_t n, BigClass cls,
                                    operand_cache::Mode mode, std::uint64_t cap_bytes, BigCacheReport* report);
// Every dense_big_gemm on ctx consults `cache` (nullptr detaches). Several contexts (streams) may share one cache.
void dense_big_ctx_attach_cache(BigCtx* ctx, BigOperandCache* cache);
// Stage feed of the FAST4 tile GEMMs (dense_big_gemm with a.f4, dense_big_glu4) on ctx. Scheduling only: both feeds produce the
// same bits. A new context (and the module default context) uses nvfp4::tile::kDefaultFeed (lsu); Feed::bulk is opt-in.
void dense_big_ctx_set_tile_feed(BigCtx* ctx, ::strix::cuda::core::nvfp4::tile::Feed feed);

}  // namespace strix::cuda_core
