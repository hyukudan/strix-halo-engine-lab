#pragma once
// Shared FP16 WMMA tile GEMM for short-K strided row batches on v_wmma_f32_16x16x16_f16:
//   y[b][r][n] = sum_k fp16(a[b][r][k]) * w[b][n][k]   (b < batches, r < rows, n < N, k < K)
//
// Class: FAST (not an exact FP32 chain). Arithmetic contract (same as core/f16_dense_tall.hpp): a is rounded to FP16 by
// the _Float16 cast (RNE, finite values beyond the FP16 range become inf, NaN stays NaN); the weights are FP16 as stored;
// per output one WMMA chain from +0 over the 16-wide k units in ascending order (FP32 accumulation in hardware order).
// The bits do not depend on the grid, the CTA count or the row count of a launch.
//
// Why a second geometry next to f16_dense_tall: that core streams A fragments straight from global memory per lane with a
// one-unit lookahead and stages a whole 256-column weight block (36.9 KB LDS, one CTA per CU). It is built for long K
// (in >= 1024). With K = 256 / 512 (GLM MLA absorb / out) it measured 2.2x / 1.4x SLOWER than the exact FP32 VALU GEMM
// (latency-bound: 4 / 8 stages per CTA, no CTA overlap). This core is a classic tile GEMM for that regime:
//   CTA   : 128 rows x 128 columns, 256 threads = 8 waves as 4 (rows, 32 each) x 2 (columns, 64 each);
//           wave tile 32 x 64 = 2 x 4 WMMA tiles (64 accumulator VGPRs).
//   stage : K step 32 (two 16-wide units). A (FP32 -> FP16 at staging, once per element per CTA) and W go to one
//           20 KB LDS stage (rows of 32 halves + 8 pad: b128 fragment reads are bank-conflict free); the next stage's
//           global loads are issued before this stage's WMMAs (register prefetch), so 2-3 CTAs fit one CU and one CTA's
//           loads / epilogue overlap another's WMMAs.
//           A may instead arrive as FP16 bits (Args::a16, e.g. the FP16 image its producer stored): loaded as is,
//           half the A bytes, the same operand bits as the FP32 -> FP16 staging of the values it was made from.
//   W     : Layout::NK = w[n][k] (k contiguous): two b128 loads per thread and stage, stored as is.
//           Layout::KN = w[k][n] (n contiguous): lane = k loads 16 columns (two b128) and writes them transposed with
//           16 ds_store_b16 (a wave writes 64 contiguous bytes per store), so no transposed weight copy is kept.
//   grid  : x = column tile (fastest), y = 128-row block, z = batch: the column CTAs of one row block run back to back
//           and read their A tile from L2; a batch's weights (all its row blocks) stay L2-resident.
// Store policy: where a finished sum goes (default RawStore: y[b][r][n] = sum, FP32; Bf16Store: the bf16 image
// attn_core tile::to_bf16(sum) (RNE, the operand bits a bf16 WMMA consumer makes of the FP32 sum) as u16 at the same
// element strides, half the output bytes; the sums themselves are unchanged).
//   put(b, r, n, v): row r < rows, column n, v = the chain's sum. One call per output.
// Limits (callers check with fits()): K % 32 == 0, N % 128 == 0, rows >= 1, 16-byte aligned a / w / y row starts.
#if defined(__HIPCC__)
#include "attn_core/wmma_types.hpp"
#include "core/dense_row_plan.hpp"
#include "strix/attn/group_indexer.hpp"
#include <hip/hip_runtime.h>
#endif

#include <cstddef>
#include <cstdint>

namespace strix::core::f16_tile_gemm {
inline constexpr unsigned kBM = 128, kBN = 128, kBK = 32, kThreads = 256;
inline constexpr unsigned kPitch = kBK / 2U + 4U;  // dwords per LDS row: 16 data + 4 pad (80 bytes)

enum class Layout { NK, KN };  // weight element (n, k) at w[n * ldw + k] (NK) or w[k * ldw + n] (KN)

// Strides in elements. a: (b, r, k) at a + b * a_batch + r * a_row + k; w: as Layout, + b * w_batch; y: (b, r, n) at
// y + b * y_batch + r * y_row + n.
struct Args {
    const float* a;
    const std::uint16_t* a16 = nullptr;  // when set: A is already FP16 bits (same strides), read instead of `a`. Same
                                         // bits as `a` when the producer stored static_cast<_Float16>(x) (RNE)
    std::size_t a_row, a_batch;
    const std::uint16_t* w;  // FP16 bits
    std::size_t w_ld, w_batch;
    float* y;
    std::size_t y_row, y_batch;
    unsigned rows, K, N, batches;
    // CTA order. 0 = (column tile, row block, batch); 1 = (column tile, batch, row block); 2 = groups of `group`
    // batches: (column tile x batch in group, row block, group). Placement only: the bits never depend on it.
    unsigned order = 0, group = 1;
};

inline bool fits(const Args& p) {
    return p.rows >= 1U && p.K >= kBK && p.K % kBK == 0U && p.N >= kBN && p.N % kBN == 0U && p.batches >= 1U && p.batches <= 65535U &&
           p.rows <= 65535U * kBM && p.order <= 2U && p.group >= 1U && p.batches % p.group == 0U && (p.order != 2U || p.N / kBN <= 65535U / p.group) && p.w_ld % 8U == 0U &&
           (p.a16 != nullptr ? p.a_row % 8U == 0U && p.a_batch % 8U == 0U && reinterpret_cast<std::uintptr_t>(p.a16) % 16U == 0U
                             : p.a_row % 4U == 0U && p.a_batch % 4U == 0U && reinterpret_cast<std::uintptr_t>(p.a) % 16U == 0U) &&
           p.w_batch % 8U == 0U &&
           reinterpret_cast<std::uintptr_t>(p.w) % 16U == 0U;
}

#if defined(__HIPCC__)
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using Float4 = float __attribute__((ext_vector_type(4)));
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));
// A fragment stays 8 dwords (2 x ds_load_b128) and is bit-cast to Half16 only at the WMMA: a <16 x half> value makes
// gfx11 true16 codegen split and re-pack every dword (v_lshrrev_b32 + v_mov_b16, 192 VALU per stage measured).
typedef std::uint32_t Frag __attribute__((ext_vector_type(8), aligned(16)));

struct RawStore {
    float* y;
    std::size_t y_row, y_batch;
    __device__ __forceinline__ void put(unsigned b, unsigned r, unsigned n, float v) const {
        y[b * y_batch + r * y_row + n] = v;
    }
};

struct Bf16Store {
    std::uint16_t* y;
    std::size_t y_row, y_batch;
    __device__ __forceinline__ void put(unsigned b, unsigned r, unsigned n, float v) const {
        y[b * y_batch + r * y_row + n] = strix::attn_core::tile::to_bf16(v);
    }
};

struct alignas(16) Storage {
    std::uint32_t a[kBM][kPitch];  // row r: 32 FP16 values of k (low half first)
    std::uint32_t b[kBN][kPitch];  // column n: 32 FP16 values of k
};
static_assert(sizeof(Storage) == 20480);


namespace detail {
__device__ __forceinline__ Frag fragment(const std::uint32_t* p) { return *reinterpret_cast<const Frag*>(p); }
__device__ __forceinline__ std::uint32_t pack(float lo, float hi) {
    const _Float16 l = static_cast<_Float16>(lo), h = static_cast<_Float16>(hi);
    return static_cast<std::uint32_t>(__builtin_bit_cast(std::uint16_t, l)) |
           (static_cast<std::uint32_t>(__builtin_bit_cast(std::uint16_t, h)) << 16U);
}
}  // namespace detail

// Expand the common loop separately for dense FP16 and packed BF16 so the
// dense compiler input retains its original staging and instruction sequence.
#include "f16_tile_gemm_core.inc"

#define STRIX_F16_TILE_PACKED_OPERANDS
#include "f16_tile_gemm_core.inc"
#undef STRIX_F16_TILE_PACKED_OPERANDS

// A wave group owns 64 columns. Pair panels from the two halves of a
// projection group so that different projections can reuse the same LDS A
// tile. Keep adjacent panels together for a single projection.
inline constexpr unsigned kBf16Panel = kBN / 2U;
template <unsigned Capacity>
struct Bf16ProjectionGroup {
    Bf16RowProjection projections[Capacity];
    unsigned first_panel[Capacity + 1U];
    unsigned count;

    void append(const Bf16RowProjection& projection) {
        projections[count] = projection;
        first_panel[count + 1U] = first_panel[count] + (projection.outputs + kBf16Panel - 1U) / kBf16Panel;
        ++count;
    }
};

#define STRIX_F16_TILE_PLAIN_BF16
#include "f16_tile_gemm_core.inc"
#undef STRIX_F16_TILE_PLAIN_BF16

template <bool kHalfTail, unsigned Capacity>
__global__ __launch_bounds__(kThreads)
void bf16_projection_kernel(const float* a, unsigned rows, unsigned K, Bf16ProjectionGroup<Capacity> group) {
    const Args p{a, nullptr, K, 0U, nullptr, K, 0U, nullptr, 0U, 0U, rows, K, 0U, 1U};
    __shared__ Storage lds;
    core<kHalfTail>(p, lds, group);
}

template <unsigned Capacity>
inline void gemm_bf16_projections(const float* a, unsigned rows, unsigned K,
                                  const Bf16ProjectionGroup<Capacity>& group, hipStream_t stream) {
    const dim3 grid((group.first_panel[group.count] + 1U) / 2U, (rows + kBM - 1U) / kBM);
    if (K % kBK == 0U) STRIX_HIP_LAUNCH(Rdna3Layout, bf16_projection_kernel<false><<<grid, kThreads, 0, stream>>>(a, rows, K, group));
    else STRIX_HIP_LAUNCH(Rdna3Layout, bf16_projection_kernel<true><<<grid, kThreads, 0, stream>>>(a, rows, K, group));
}

template <Layout L, class Store, bool kA16 = false>
__global__ __launch_bounds__(kThreads) void gemm_kernel(Args p, Store store) {
    __shared__ Storage lds;
    core<L, Store, kA16>(p, lds, store);
}

// Packed BF16 batches supply the operand and row strides for each descriptor.
// The binding may decline a descriptor before any LDS barrier is reached.
template <class Binding>
__global__ __launch_bounds__(kThreads) void gemm_kernel(typename Binding::Args descriptors) {
    if (!Binding::active(descriptors, blockIdx.z)) return;
    const auto tile = Binding::bind(descriptors, blockIdx.z);
    __shared__ Storage lds;
    core<Layout::NK, typename Binding::Store, true>(tile.shape, lds, tile.store, tile.weights);
}

template <class Binding>
inline hipError_t gemm_packed_bf16(const typename Binding::Args& descriptors, unsigned outputs, unsigned count,
                                  hipStream_t stream) {
    const dim3 grid(outputs / kBN, 1U, count);
    STRIX_HIP_LAUNCH(Rdna3Layout, gemm_kernel<Binding><<<grid, kThreads, 0, stream>>>(descriptors));
    return hipGetLastError();
}

inline dim3 grid(const Args& p) {
    const unsigned rb = (p.rows + kBM - 1U) / kBM;
    if (p.order == 2U) return dim3(p.N / kBN * p.group, rb, p.batches / p.group);
    return p.order == 1U ? dim3(p.N / kBN, p.batches, rb) : dim3(p.N / kBN, rb, p.batches);
}

template <Layout L, class Store> inline void gemm(const Args& p, const Store& store, hipStream_t s) {
    if (p.a16 != nullptr) STRIX_HIP_LAUNCH(Rdna3Layout, gemm_kernel<L, Store, true><<<grid(p), kThreads, 0, s>>>(p, store));
    else STRIX_HIP_LAUNCH(Rdna3Layout, gemm_kernel<L, Store><<<grid(p), kThreads, 0, s>>>(p, store));
}
template <Layout L> inline void gemm(const Args& p, hipStream_t s) { gemm<L>(p, RawStore{p.y, p.y_row, p.y_batch}, s); }
// Bf16Store into y's memory (the u16 image occupies its first half; same element strides).
template <Layout L> inline void gemm_bf16(const Args& p, hipStream_t s) {
    gemm<L>(p, Bf16Store{reinterpret_cast<std::uint16_t*>(p.y), p.y_row, p.y_batch}, s);
}

#endif  // defined(__HIPCC__): Args and fits also support host admission tests.

}  // namespace strix::core::f16_tile_gemm
