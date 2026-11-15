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
//   W     : Layout::NK = w[n][k] (k contiguous): two b128 loads per thread and stage, stored as is.
//           Layout::KN = w[k][n] (n contiguous): lane = k loads 16 columns (two b128) and writes them transposed with
//           16 ds_store_b16 (a wave writes 64 contiguous bytes per store), so no transposed weight copy is kept.
//   grid  : x = column tile (fastest), y = 128-row block, z = batch: the column CTAs of one row block run back to back
//           and read their A tile from L2; a batch's weights (all its row blocks) stay L2-resident.
// Store policy: where a finished sum goes (default RawStore: y[b][r][n] = sum, FP32).
//   put(b, r, n, v): row r < rows, column n, v = the chain's sum. One call per output.
// Limits (callers check with fits()): K % 32 == 0, N % 128 == 0, rows >= 1, 16-byte aligned a / w / y row starts.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::core::f16_tile_gemm {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using Float4 = float __attribute__((ext_vector_type(4)));
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));
// A fragment stays 8 dwords (2 x ds_load_b128) and is bit-cast to Half16 only at the WMMA: a <16 x half> value makes
// gfx11 true16 codegen split and re-pack every dword (v_lshrrev_b32 + v_mov_b16, 192 VALU per stage measured).
typedef std::uint32_t Frag __attribute__((ext_vector_type(8), aligned(16)));

inline constexpr unsigned kBM = 128, kBN = 128, kBK = 32, kThreads = 256;
inline constexpr unsigned kPitch = kBK / 2U + 4U;  // dwords per LDS row: 16 data + 4 pad (80 bytes)

enum class Layout { NK, KN };  // weight element (n, k) at w[n * ldw + k] (NK) or w[k * ldw + n] (KN)

// Strides in elements. a: (b, r, k) at a + b * a_batch + r * a_row + k; w: as Layout, + b * w_batch; y: (b, r, n) at
// y + b * y_batch + r * y_row + n.
struct Args {
    const float* a;
    std::size_t a_row, a_batch;
    const std::uint16_t* w;  // FP16 bits
    std::size_t w_ld, w_batch;
    float* y;
    std::size_t y_row, y_batch;
    unsigned rows, K, N, batches;
    unsigned order = 0;  // grid: 0 = (column tile, row block, batch); 1 = (column tile, batch, row block)
};

struct RawStore {
    float* y;
    std::size_t y_row, y_batch;
    __device__ __forceinline__ void put(unsigned b, unsigned r, unsigned n, float v) const {
        y[b * y_batch + r * y_row + n] = v;
    }
};

struct alignas(16) Storage {
    std::uint32_t a[kBM][kPitch];  // row r: 32 FP16 values of k (low half first)
    std::uint32_t b[kBN][kPitch];  // column n: 32 FP16 values of k
};
static_assert(sizeof(Storage) == 20480);

inline bool fits(const Args& p) {
    return p.rows >= 1U && p.K % kBK == 0U && p.N % kBN == 0U && p.batches >= 1U && p.batches <= 65535U &&
           (p.rows + kBM - 1U) / kBM <= 65535U && p.order <= 1U && p.a_row % 4U == 0U && p.a_batch % 4U == 0U && p.w_ld % 8U == 0U &&
           p.w_batch % 8U == 0U && reinterpret_cast<std::uintptr_t>(p.a) % 16U == 0U &&
           reinterpret_cast<std::uintptr_t>(p.w) % 16U == 0U;
}

namespace detail {
__device__ __forceinline__ Frag fragment(const std::uint32_t* p) { return *reinterpret_cast<const Frag*>(p); }
__device__ __forceinline__ std::uint32_t pack(float lo, float hi) {
    const _Float16 l = static_cast<_Float16>(lo), h = static_cast<_Float16>(hi);
    return static_cast<std::uint32_t>(__builtin_bit_cast(std::uint16_t, l)) |
           (static_cast<std::uint32_t>(__builtin_bit_cast(std::uint16_t, h)) << 16U);
}
}  // namespace detail

template <Layout L, class Store>
__device__ __forceinline__ void core(const Args& p, Storage& lds, const Store& store) {
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, wm = wave & 3U, wn = wave >> 2U;
    const unsigned n0 = blockIdx.x * kBN, r0 = (p.order ? blockIdx.z : blockIdx.y) * kBM, b = p.order ? blockIdx.y : blockIdx.z;
    const unsigned c = lane & 15U;
    // A stager: row tid / 2, k half (tid % 2) * 16 of the stage. Rows past `rows` read row 0 (never stored).
    const unsigned ar = tid >> 1U, ak = (tid & 1U) * 16U;
    const float* ap = p.a + b * p.a_batch + static_cast<std::size_t>(r0 + ar < p.rows ? r0 + ar : 0U) * p.a_row + ak;
    // W stager. NK: column n0 + tid / 2, k half (tid % 2) * 16. KN: k = lane, columns n0 + 16 * wave .. + 15.
    const std::uint16_t* wp = p.w + b * p.w_batch +
                              (L == Layout::NK ? static_cast<std::size_t>(n0 + (tid >> 1U)) * p.w_ld + (tid & 1U) * 16U
                                               : static_cast<std::size_t>(lane) * p.w_ld + n0 + 16U * wave);
    Float4 ra[4];
    U4 rb[2];
    const auto load = [&](unsigned k0) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) ra[i] = *reinterpret_cast<const Float4*>(ap + k0 + 4U * i);
        const std::uint16_t* q = wp + (L == Layout::NK ? std::size_t{k0} : std::size_t{k0} * p.w_ld);
        rb[0] = *reinterpret_cast<const U4*>(q);
        rb[1] = *reinterpret_cast<const U4*>(q + 8);
    };
    const auto publish = [&]() __attribute__((always_inline)) {
        U4 h0, h1;
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) {
            const Float4 f = ra[i];
            const std::uint32_t lo = detail::pack(f[0], f[1]), hi = detail::pack(f[2], f[3]);
            if (i < 2U) { h0[2U * i] = lo; h0[2U * i + 1U] = hi; }
            else { h1[2U * (i - 2U)] = lo; h1[2U * (i - 2U) + 1U] = hi; }
        }
        *reinterpret_cast<U4*>(&lds.a[ar][ak / 2U]) = h0;
        *reinterpret_cast<U4*>(&lds.a[ar][ak / 2U + 4U]) = h1;
        if constexpr (L == Layout::NK) {
            *reinterpret_cast<U4*>(&lds.b[tid >> 1U][(tid & 1U) * 8U]) = rb[0];
            *reinterpret_cast<U4*>(&lds.b[tid >> 1U][(tid & 1U) * 8U + 4U]) = rb[1];
        } else {
#pragma unroll
            for (unsigned i = 0; i < 16U; ++i) {
                const std::uint32_t word = rb[i / 8U][(i % 8U) / 2U];
                reinterpret_cast<std::uint16_t*>(&lds.b[16U * wave + i][0])[lane] =
                    static_cast<std::uint16_t>(word >> (16U * (i % 2U)));
            }
        }
    };
    Float8 acc[2][4] = {};
    load(0U);
    for (unsigned k0 = 0; k0 < p.K; k0 += kBK) {
        __syncthreads();  // every wave finished reading the previous stage
        publish();
        __syncthreads();
        if (k0 + kBK < p.K) load(k0 + kBK);  // in flight across this stage's WMMAs
#pragma unroll
        for (unsigned u = 0; u < 2U; ++u) {
            Frag fa[2], fb[4];
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m) fa[m] = detail::fragment(&lds.a[32U * wm + 16U * m + c][8U * u]);
#pragma unroll
            for (unsigned t = 0; t < 4U; ++t) fb[t] = detail::fragment(&lds.b[64U * wn + 16U * t + c][8U * u]);
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t) acc[m][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    __builtin_bit_cast(Half16, fa[m]), __builtin_bit_cast(Half16, fb[t]), acc[m][t]);
        }
    }
    // acc[m][t][j] -> row 32 wm + 16 m + 2 j + lane / 16, column 64 wn + 16 t + lane % 16.
#pragma unroll
    for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned r = r0 + 32U * wm + 16U * m + 2U * j + (lane >> 4U);
            if (r < p.rows) {
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t) store.put(b, r, n0 + 64U * wn + 16U * t + c, acc[m][t][j]);
            }
        }
}

template <Layout L, class Store>
__global__ __launch_bounds__(kThreads) void gemm_kernel(Args p, Store store) {
    __shared__ Storage lds;
    core<L>(p, lds, store);
}

inline dim3 grid(const Args& p) {
    const unsigned rb = (p.rows + kBM - 1U) / kBM;
    return p.order ? dim3(p.N / kBN, p.batches, rb) : dim3(p.N / kBN, rb, p.batches);
}

template <Layout L> inline void gemm(const Args& p, hipStream_t s) {
    gemm_kernel<L, RawStore><<<grid(p), kThreads, 0, s>>>(p, RawStore{p.y, p.y_row, p.y_batch});
}

}  // namespace strix::core::f16_tile_gemm
