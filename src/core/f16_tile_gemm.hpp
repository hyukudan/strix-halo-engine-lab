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
// Resident Q8_0 projection policy (resident-q8-f16-tile-v1, FAST):
//   a16 = fp16_rne(a), w16 = fp16_rne(float(code) * float(d)), with signed int8 code and stored FP16 scale d.
//   Tight NK rows use Q8Storage blocks (34 bytes / 32 values). Each 128-row CTA stages each live column's block once
//   per K32 step, then uses the same ascending FP16 WMMA chain and FP32 stores. No expanded weight image is stored.
//   Projections sharing A can pair 64-column panels in a CTA. Partial panels mask loads and stores.
//   K must be a multiple of 32. Resident validation requires 16-byte aligned A and weight bases. Outputs are tight FP32.
#if defined(__HIPCC__)
#include "attn_core/wmma_types.hpp"
#include "core/dense_row_plan.hpp"
#include "core/controller_mix.hpp"
#include "core/nvfp4_bf16_table.hpp"
#include "core/q8_f16_staging.hpp"
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

// Schedule of the packed-operand core (placement only: never changes bits). A CTA of 64 x WavesM threads covers
// 16 x RowTiles x WavesM rows x 128 columns: waves are WavesM (rows, 16 x RowTiles each) x 2 (columns, 64 each). One stage
// holds StageK inputs (StageK / 16 K16 units): barrier, publish (A copy, W decode once per CTA and stage), barrier,
// register prefetch of the next stage, WMMAs. Every form runs one
// WMMA chain per output from +0 over the K16 units in ascending order with rows at (row % 16), columns at (column % 16) and
// k at (k % 16) of each fragment, so all forms store the same bits.
template <unsigned RowTiles = 2, unsigned WavesM = 4, unsigned StageK = kBK>
struct TileSchedule {
    static_assert((RowTiles == 2U || RowTiles == 4U) && (WavesM == 4U || WavesM == 8U) &&
                  (StageK == 32U || StageK == 64U));
    static constexpr unsigned kRowTiles = RowTiles, kWavesM = WavesM, kStageK = StageK;
    static constexpr unsigned kRows = 16U * RowTiles * WavesM, kThreads = 64U * WavesM, kUnits = StageK / 16U;
    static constexpr unsigned kPitch = StageK / 2U + 4U;  // dwords per LDS row: data + 4 pad
    static constexpr unsigned kAChunks = kRows * kUnits / kThreads, kWChunks = kBN * kUnits / kThreads;  // per thread
    static_assert(kRows * kUnits % kThreads == 0U && kBN * kUnits % kThreads == 0U && kWChunks >= 1U);
    struct alignas(16) Storage {
        std::uint32_t a[kRows][kPitch];  // row r: StageK FP16 / BF16 values of k (low half first)
        std::uint32_t b[kBN][kPitch];    // column n: StageK values of k
    };
    // The paired epilogue exchanges 2 x 8 rows x 64 columns FP32 per wave through the stage memory.
    static_assert(sizeof(Storage) >= 2U * WavesM * 8U * 64U * sizeof(float) && sizeof(Storage) <= 65536U);
};
using IncumbentSchedule = TileSchedule<>;
using Storage = IncumbentSchedule::Storage;
static_assert(sizeof(Storage) == 20480 && IncumbentSchedule::kPitch == kPitch && IncumbentSchedule::kRows == kBM);



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

// Column pairing of a packed-operand tile (see core): the default keeps 128 consecutive columns per CTA. A pair type
// with kPaired = true gives the CTA logical columns [64 x, 64 x + 64) and the same columns + half, and receives
// put(row, column, sum, paired sum) for every live row and column < half instead of the store.
struct UnpairedColumns {
    static constexpr bool kPaired = false;
    unsigned half = 0U;
    __device__ void put(unsigned, unsigned, float, float) const {}
};

// Packed-operand A rows: row r of the tile is row r of the operand (DirectRows) or row map[r] (MappedRows: a grouped
// GEMM reading token rows in place of a gathered copy; same strides, same operand bits).
struct DirectRows {
    __device__ unsigned operator()(unsigned row) const { return row; }
};
struct MappedRows {
    const std::uint32_t* map;
    __device__ unsigned operator()(unsigned row) const { return map[row]; }
};

#define STRIX_F16_TILE_PACKED_OPERANDS
#include "f16_tile_gemm_core.inc"
#undef STRIX_F16_TILE_PACKED_OPERANDS

// A wave group owns 64 columns. Pair panels from the two halves of a
// projection group so that different projections can reuse the same LDS A tile.
// Keep adjacent panels together for a single projection.
inline constexpr unsigned kProjectionPanel = kBN / 2U;
template <class Projection, unsigned Capacity>
struct ProjectionGroup {
    Projection projections[Capacity];
    unsigned first_panel[Capacity + 1U];
    unsigned count;

    constexpr void append(const Projection& projection) {
        projections[count] = projection;
        const unsigned panels = projection.outputs / kProjectionPanel + (projection.outputs % kProjectionPanel != 0U);
        first_panel[count + 1U] = first_panel[count] + panels;
        ++count;
    }
};
// Keep the contiguous kernels' argument layout independent of host projection strides.
struct Bf16Projection {
    const std::uint16_t* weights;
    unsigned outputs;
    float* output;
    bool round_bf16;
};
static_assert(sizeof(Bf16Projection) == offsetof(Bf16RowProjection, row_bytes));
template <unsigned Capacity> struct Bf16ProjectionGroup : ProjectionGroup<Bf16Projection, Capacity> {};
struct Q8Projection {
    const std::uint8_t* weights;
    unsigned outputs;
    float* output;
};
static_assert(kBK == weight::Q8Storage::elements);
template <unsigned Capacity> using Q8ProjectionGroup = ProjectionGroup<Q8Projection, Capacity>;

// Gated epilogue of a [forget, gain] pair (see core): flat [row][width] FP32 rows.
struct ControllerRows {
    const float* current;
    const float* message;
    float* output;
    unsigned width;
};

#define STRIX_F16_TILE_PLAIN_BF16
#include "f16_tile_gemm_core.inc"
#undef STRIX_F16_TILE_PLAIN_BF16

#define STRIX_F16_TILE_Q8
#include "f16_tile_gemm_core.inc"
#undef STRIX_F16_TILE_Q8

// Only a pitched launch carries a weight leading dimension in elements.
template <bool kHalfTail, unsigned Capacity, class... Pitch>
__global__ __launch_bounds__(kThreads)
void bf16_projection_kernel(const float* a, unsigned rows, unsigned K, Bf16ProjectionGroup<Capacity> group,
                            Pitch... pitch) {
    static_assert(sizeof...(Pitch) <= 1U);
    std::size_t w_ld = K;
    ((w_ld = pitch), ...);
    const Args p{a, nullptr, K, 0U, nullptr, w_ld, 0U, nullptr, 0U, 0U, rows, K, 0U, 1U};
    __shared__ Storage lds;
    core<kHalfTail>(p, lds, group);
}

// [forget, gain] (equal widths, FP32 sums) with the low-rank controller as epilogue: output[r][n] =
// controller_mix(current[r][n], message[r][n], forget sum, gain sum). The grid gives each CTA forget panel x (waves
// wn = 0) and gain panel x (wn = 1): first_panel[1] == grid x.
template <bool kHalfTail, unsigned Capacity>
__global__ __launch_bounds__(kThreads)
void bf16_controller_kernel(const float* a, unsigned rows, unsigned K, Bf16ProjectionGroup<Capacity> group,
                            ControllerRows gate) {
    const Args p{a, nullptr, K, 0U, nullptr, K, 0U, nullptr, 0U, 0U, rows, K, 0U, 1U};
    __shared__ Storage lds;
    core<kHalfTail, Capacity, true>(p, lds, group, gate);
}

template <unsigned Capacity>
inline void gemm_bf16_controller(const float* a, unsigned rows, unsigned K, const Bf16ProjectionGroup<Capacity>& group,
                                 const ControllerRows& gate, hipStream_t stream) {
    const dim3 grid(group.first_panel[1], (rows + kBM - 1U) / kBM);
    if (K % kBK == 0U)
        STRIX_HIP_LAUNCH(Rdna3Layout, bf16_controller_kernel<false><<<grid, kThreads, 0, stream>>>(a, rows, K, group, gate));
    else
        STRIX_HIP_LAUNCH(Rdna3Layout, bf16_controller_kernel<true><<<grid, kThreads, 0, stream>>>(a, rows, K, group, gate));
}

template <unsigned Capacity, class... Pitch>
inline void gemm_bf16_projections(const float* a, unsigned rows, unsigned K,
                                  const Bf16ProjectionGroup<Capacity>& group, hipStream_t stream, Pitch... pitch) {
    const dim3 grid((group.first_panel[group.count] + 1U) / 2U, (rows + kBM - 1U) / kBM);
    if (K % kBK == 0U)
        STRIX_HIP_LAUNCH(Rdna3Layout, bf16_projection_kernel<false><<<grid, kThreads, 0, stream>>>(a, rows, K, group, pitch...));
    else
        STRIX_HIP_LAUNCH(Rdna3Layout, bf16_projection_kernel<true><<<grid, kThreads, 0, stream>>>(a, rows, K, group, pitch...));
}

template <unsigned Capacity>
__global__ __launch_bounds__(kThreads)
void q8_projection_kernel(const float* a, unsigned rows, unsigned K, std::size_t w_row_bytes,
                          Q8ProjectionGroup<Capacity> group) {
    const Args p{a, nullptr, K, 0U, nullptr, w_row_bytes, 0U, nullptr, 0U, 0U, rows, K, 0U, 1U};
    __shared__ Storage lds;
    q8_core(p, lds, group);
}

template <unsigned Capacity>
inline hipError_t gemm_q8_projections(const float* a, unsigned rows, unsigned K,
                                      const Q8ProjectionGroup<Capacity>& group, hipStream_t stream) {
    static_assert(Capacity <= 32U);  // The total panel count fits unsigned.
    const dim3 grid((group.first_panel[group.count] + 1U) / 2U, (rows + kBM - 1U) / kBM);
    const auto w_row_bytes = static_cast<std::size_t>(weight::Q8Storage::row_bytes(K));
    STRIX_HIP_LAUNCH(Rdna3Layout, q8_projection_kernel<Capacity><<<grid, kThreads, 0, stream>>>(a, rows, K, w_row_bytes, group));
    return hipGetLastError();
}

template <Layout L, class Store, bool kA16 = false>
__global__ __launch_bounds__(kThreads) void gemm_kernel(Args p, Store store) {
    __shared__ Storage lds;
    core<L, Store, kA16>(p, lds, store);
}

// Packed BF16 batches supply the operand and row strides for each descriptor.
// The binding may decline a descriptor before any LDS barrier is reached.
// Grid y = row slices of Sched::kRows rows: a descriptor tile taller than the CTA is covered by several CTAs (each a
// 16-aligned row range of the same tile, so the bits do not change); a slice past the tile's rows leaves before any barrier.
template <class Binding, class Sched = IncumbentSchedule>
__global__ __launch_bounds__(Sched::kThreads) void gemm_kernel(typename Binding::Args descriptors) {
    if (!Binding::active(descriptors, blockIdx.z)) return;
    const auto tile = Binding::bind(descriptors, blockIdx.z);
    if (blockIdx.y != 0U && blockIdx.y * Sched::kRows >= tile.shape.rows) return;
    __shared__ typename Sched::Storage lds;
    if constexpr (requires { tile.rows; }) {  // the binding maps its A rows (MappedRows)
        if constexpr (requires { tile.pair; })
            core<Layout::NK, typename Binding::Store, true, decltype(tile.weights), decltype(tile.pair), MappedRows,
                 Sched>(tile.shape, lds, tile.store, tile.weights, tile.pair, tile.rows);
        else
            core<Layout::NK, typename Binding::Store, true, decltype(tile.weights), UnpairedColumns, MappedRows, Sched>(
                tile.shape, lds, tile.store, tile.weights, UnpairedColumns{}, tile.rows);
    } else if constexpr (requires { tile.pair; }) {
        core<Layout::NK, typename Binding::Store, true, decltype(tile.weights), decltype(tile.pair), DirectRows, Sched>(
            tile.shape, lds, tile.store, tile.weights, tile.pair);
    } else {
        core<Layout::NK, typename Binding::Store, true, decltype(tile.weights), UnpairedColumns, DirectRows, Sched>(
            tile.shape, lds, tile.store, tile.weights);
    }
}

// row_slices: grid y (ceil(largest tile rows / Sched::kRows)); 1 when every tile fits one CTA.
template <class Binding, class Sched = IncumbentSchedule>
inline hipError_t gemm_packed_bf16(const typename Binding::Args& descriptors, unsigned outputs, unsigned count,
                                  hipStream_t stream, unsigned row_slices = 1U) {
    const dim3 grid(outputs / kBN, row_slices, count);
    STRIX_HIP_LAUNCH(Rdna3Layout, gemm_kernel<Binding, Sched><<<grid, Sched::kThreads, 0, stream>>>(descriptors));
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
