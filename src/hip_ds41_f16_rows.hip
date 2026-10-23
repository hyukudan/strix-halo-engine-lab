// F16 projection rows with 16-byte weight loads; arithmetic of f16_rows (see hip_ds41_f16_rows.hpp).
#include "strix/hip_ds41_f16_rows.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::ds41 {
namespace {

constexpr unsigned kWave = 32, kRows = 8;

template <unsigned D> __device__ __forceinline__ float xor_lane(float v) {  // DPP row_xmask: lane ^ D (D < 16)
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v),
                                                                  static_cast<int>(0x160U + D), 0xF, 0xF, false));
}
__device__ __forceinline__ float half_value(std::uint16_t h) {  // exact (the conversion of f16_to_float for every non-NaN)
    return static_cast<float>(__builtin_bit_cast(_Float16, h));
}

// Eight weights e .. e + 7 (e % 8 == 0) of row r as F16 bits: from the F16 rows, or widened from F8 rows (E4M3 code x
// 2^(exponent - 127), exact in F16 for the MX-FP8-valued tensors packed by ds41_packed), so both give the same floats.
// E4M3 code times the block scale 2^k to F16 bits, branch-free: the E4M3 value as F32 by bits (normal 1.m x 2^(e-7)
// -> exponent e + 120, mantissa m << 20; subnormal m x 2^-9 by a select), one exact multiply by the F32 scale, and the
// hardware F32 -> F16 conversion, exact because the packer checked that every value is an F16 value.
__device__ __forceinline__ unsigned f8_half_bits(unsigned code, float scale) {
    const unsigned sign = (code & 0x80U) << 24U, e = (code >> 3U) & 15U, m = code & 7U;
    const float normal = __builtin_bit_cast(float, sign | ((e + 120U) << 23U) | (m << 20U));
    const float sub = __builtin_bit_cast(float, sign | __builtin_bit_cast(unsigned, static_cast<float>(m) * 0x1p-9F));
    const float v = (e != 0U ? normal : sub) * scale;
    return __builtin_bit_cast(unsigned short, static_cast<_Float16>(v));
}
template <bool F8>
__device__ __forceinline__ uint4 load8(const F16RowsView& v, unsigned r, unsigned e) {
    if constexpr (!F8) {
        return *reinterpret_cast<const uint4*>(v.weight + static_cast<std::size_t>(r) * v.width + e);
    } else {
        const std::uint8_t* row = v.weight_f8 + static_cast<std::size_t>(r) * (v.width + v.width / 32U);
        const uint2 c = *reinterpret_cast<const uint2*>(row + e);
        // E8M0 exponent x: 2^(x - 127) as F32 (x >= 1 for the packed tensors; x = 0 would be 2^-127, a subnormal)
        const unsigned x = row[v.width + e / 32U];
        const float scale = x != 0U ? __builtin_bit_cast(float, x << 23U) : 0x1p-127F;
        unsigned h[8];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            h[j] = f8_half_bits((c.x >> (8U * j)) & 0xffU, scale);
            h[4 + j] = f8_half_bits((c.y >> (8U * j)) & 0xffU, scale);
        }
        return uint4{h[0] | (h[1] << 16U), h[2] | (h[3] << 16U), h[4] | (h[5] << 16U), h[6] | (h[7] << 16U)};
    }
}

// ---- one output row per wave (1 row, or 2 rows whose activations fit in LDS) ----
// The wave reads its row 512 contiguous bytes per 256-element chunk (lane l: elements 8 l .. 8 l + 7), kRowDepth
// chunks in flight, passes each chunk through a wave-private LDS tile, and lane c takes its elements c + 32 m
// (m = 0..7) in order: lane c = class c, as in f16_rows. One row per wave also keeps the fewest DRAM streams open
// (measured: 1 row per wave ~195 GB/s, 8 rows per wave 135-150 GB/s of a 243 GB/s linear sweep). Activations are
// staged once per block in LDS (whole rows when they fit, else 1024-element groups).
constexpr unsigned kRowWaves = 8, kRowDepth = 4, kXLds = 8192;
// The body takes its block index and LDS (xs: kXLds floats, tile: kRowWaves x 256) so that a pair launch can run it.
template <unsigned R, bool Whole, bool F8 = false>
__device__ __forceinline__ void f16_rows_row_body(const F16RowsView& v, unsigned block, float* xs, unsigned short (*tile)[256]) {
    const unsigned l = threadIdx.x % kWave, wv = threadIdx.x / kWave, rows = v.rows, chunks = v.width / 256U;
    const unsigned o = block * kRowWaves + wv, oc = o < v.count ? o : v.count - 1U;
    const unsigned span = Whole ? v.width : 1024U;
    const unsigned ri = v.first + oc;  // weight row; lane l reads elements 256 k + 8 l .. + 7 of chunk k
    float acc[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) acc[r] = 0.0F;
    uint4 buf[kRowDepth];
#pragma unroll
    for (unsigned d = 0; d < kRowDepth; ++d) buf[d] = d < chunks ? load8<F8>(v, ri, 256U * d + 8U * l) : uint4{0, 0, 0, 0};
    for (unsigned k0 = 0; k0 < chunks; k0 += kRowDepth) {
#pragma unroll
        for (unsigned d = 0; d < kRowDepth; ++d) {
            const unsigned k = k0 + d;
            if (k < chunks) {
                const unsigned e0 = 256U * k;
                if (e0 % span == 0U) {   // stage activations e0 .. e0 + span - 1 of every row
                    __syncthreads();
                    const unsigned n = v.width - e0 < span ? v.width - e0 : span;
                    for (unsigned i = threadIdx.x; i < rows * n; i += kRowWaves * kWave) {
                        const unsigned r = i / n, q = i % n;
                        xs[r * span + q] = v.x[static_cast<std::size_t>(r) * v.x_stride + e0 + q];
                    }
                    __syncthreads();
                }
                reinterpret_cast<uint4*>(tile[wv])[l] = buf[d];
                if (k + kRowDepth < chunks) buf[d] = load8<F8>(v, ri, 256U * (k + kRowDepth) + 8U * l);
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                const unsigned xo = e0 % span;
#pragma unroll
                for (unsigned m = 0; m < 8; ++m) {
                    const float w = half_value(tile[wv][32U * m + l]);
#pragma unroll
                    for (unsigned r = 0; r < R; ++r)
                        if (r < rows) acc[r] = acc[r] + w * xs[r * span + xo + 32U * m + l];
                }
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        if (r >= rows) break;
        float t = acc[r];
        t = t + __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(unsigned, t), __builtin_bit_cast(unsigned, t),
                                                                       0x76543210U, 0xfedcba98U, false, false));
        t = t + xor_lane<8>(t);
        t = t + xor_lane<4>(t);
        t = t + xor_lane<2>(t);
        t = t + xor_lane<1>(t);
        if (l == 0U && o < v.count) v.out[static_cast<std::size_t>(r) * v.out_stride + o] = t;
    }
}
template <unsigned R, bool Whole, bool F8 = false>
__global__ __launch_bounds__(kRowWaves * kWave) void f16_rows_row_kernel(F16RowsView v) {
    __shared__ float xs[kXLds];
    __shared__ unsigned short tile[kRowWaves][256];
    f16_rows_row_body<R, Whole, F8>(v, blockIdx.x, xs, tile);
}

// ---- four lanes per output (3..8 rows, or widths that are not multiples of 256) ----
// Lane l: output l / 4 of the wave's 8, class group cg = l % 4 = classes 8 cg .. 8 cg + 7. Step n reads the row's
// elements 32 n + 8 cg .. +7 (16 bytes, kAhead steps in flight) and adds w * x to class (8 cg + j)'s accumulator, n
// ascending; the activations are staged in LDS by chunks of kChunk steps. The butterfly's levels 16 and 8 are DPP
// exchanges between the four lanes (lane ^ 2, lane ^ 1) and 4, 2, 1 are adds inside the lane: the same pairs.
constexpr unsigned kChunk = 32, kAhead = 16;
static_assert(kChunk % kAhead == 0);
template <unsigned R, unsigned Waves, bool F8 = false>
__device__ __forceinline__ void f16_rows_lanes_body(const F16RowsView& v, unsigned block, float4 (*xs)[kChunk * 8]) {
    const unsigned l = threadIdx.x % kWave, cg = l % 4U, outs_per_block = Waves * kWave / 4U;
    const unsigned o = block * outs_per_block + threadIdx.x / 4U;
    const unsigned oc = o < v.count ? o : v.count - 1U, steps = v.width / 32U, rows = v.rows;
    const unsigned ri = v.first + oc;  // weight row; step n reads elements 32 n + 8 cg .. + 7
    float acc[R][8];
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) acc[r][j] = 0.0F;
    uint4 w[kAhead];
#pragma unroll
    for (unsigned a = 0; a < kAhead; ++a) w[a] = a < steps ? load8<F8>(v, ri, 32U * a + 8U * cg) : uint4{0, 0, 0, 0};
    for (unsigned c0 = 0; c0 < steps; c0 += kChunk) {
        const unsigned cs = steps - c0 < kChunk ? steps - c0 : kChunk;
        __syncthreads();
        for (unsigned i = threadIdx.x; i < rows * cs * 8U; i += Waves * kWave) {
            const unsigned r = i / (cs * 8U), q = i % (cs * 8U);
            xs[r][q] = reinterpret_cast<const float4*>(v.x + static_cast<std::size_t>(r) * v.x_stride + 32U * c0)[q];
        }
        __syncthreads();
        for (unsigned k0 = 0; k0 < cs; k0 += kAhead) {
#pragma unroll
            for (unsigned a = 0; a < kAhead; ++a) {
                const unsigned k = k0 + a, n = c0 + k;
                if (k < cs) {
                    const uint4 cur = w[a];   // kChunk is a multiple of kAhead, so n % kAhead == a
                    if (n + kAhead < steps) w[a] = load8<F8>(v, ri, 32U * (n + kAhead) + 8U * cg);
                    const std::uint32_t h[4] = {cur.x, cur.y, cur.z, cur.w};
                    float wf[8];
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j) {
                        wf[2 * j] = half_value(static_cast<std::uint16_t>(h[j]));
                        wf[2 * j + 1] = half_value(static_cast<std::uint16_t>(h[j] >> 16U));
                    }
#pragma unroll
                    for (unsigned r = 0; r < R; ++r) {
                        if (r < rows) {
                            const float4 x0 = xs[r][8U * k + 2U * cg], x1 = xs[r][8U * k + 2U * cg + 1U];
                            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                            for (unsigned j = 0; j < 8; ++j) acc[r][j] = acc[r][j] + wf[j] * xv[j];
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        if constexpr (R > kRows) {
            if (r >= rows) continue;
        } else if (r >= rows) break;
        float s[8];
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            float t = acc[r][j];
            t = t + xor_lane<2>(t);   // class ^ 16
            t = t + xor_lane<1>(t);   // class ^ 8
            s[j] = t;
        }
        const float s4[4] = {s[0] + s[4], s[1] + s[5], s[2] + s[6], s[3] + s[7]};  // class ^ 4
        const float s2[2] = {s4[0] + s4[2], s4[1] + s4[3]};                        // class ^ 2
        const float total = s2[0] + s2[1];                                          // class ^ 1
        if (cg == 0U && o < v.count) v.out[static_cast<std::size_t>(r) * v.out_stride + o] = total;
    }
}
template <unsigned R, unsigned Waves, bool F8 = false>
__global__ __launch_bounds__(Waves * kWave) void f16_rows_lanes_kernel(F16RowsView v) {
    __shared__ float4 xs[R][kChunk * 8];
    f16_rows_lanes_body<R, Waves, F8>(v, blockIdx.x, xs);
}

// ---- two projections in one launch (STRIX_DS41_F16_PAIR) ----
// Blocks [0, blocks_a) form the outputs of view a and the rest those of view b, each with the body of its own form, so
// every output is the one of its own launch: kind 0 = row form with whole activation rows, 1 = row form by 1024-element
// groups, 2 = lanes form (Threads / 32 waves; the waves per block only change which outputs share a block).
template <unsigned R, unsigned K>
constexpr std::size_t pair_lds_bytes = K == 2U ? sizeof(float4) * R * kChunk * 8U : sizeof(float) * kXLds + sizeof(unsigned short) * kRowWaves * 256U;
template <unsigned R, unsigned K, unsigned Threads>
__device__ __forceinline__ void pair_body(const F16RowsView& v, unsigned block, unsigned char* lds) {
    if constexpr (K == 2U) f16_rows_lanes_body<R, Threads / kWave>(v, block, reinterpret_cast<float4 (*)[kChunk * 8]>(lds));
    else f16_rows_row_body<R, K == 0U>(v, block, reinterpret_cast<float*>(lds), reinterpret_cast<unsigned short (*)[256]>(lds + sizeof(float) * kXLds));
}
template <unsigned R, unsigned KA, unsigned KB, unsigned Threads>
__global__ __launch_bounds__(Threads) void f16_rows_pair_kernel(F16RowsView a, F16RowsView b, unsigned blocks_a) {
    constexpr std::size_t bytes = pair_lds_bytes<R, KA> > pair_lds_bytes<R, KB> ? pair_lds_bytes<R, KA> : pair_lds_bytes<R, KB>;
    __shared__ alignas(16) unsigned char lds[bytes];
    if (blockIdx.x < blocks_a) pair_body<R, KA, Threads>(a, blockIdx.x, lds);
    else pair_body<R, KB, Threads>(b, blockIdx.x - blocks_a, lds);
}

template <unsigned R, bool F8>
void launch_lanes_as(const F16RowsView& v, hipStream_t s) {
    // Large sites: 8 waves (64 outputs) per block; small ones: 2 waves (16 outputs) so more blocks stream at once.
    if (v.count >= 8192U)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(f16_rows_lanes_kernel<R, 8, F8>), dim3((v.count + 63U) / 64U), dim3(256), 0, s, v);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(f16_rows_lanes_kernel<R, 2, F8>), dim3((v.count + 15U) / 16U), dim3(64), 0, s, v);
}
template <unsigned R>
void launch_lanes(const F16RowsView& v, hipStream_t s) {
    if (v.weight_f8) launch_lanes_as<R, true>(v, s);
    else launch_lanes_as<R, false>(v, s);
}
template <unsigned R, bool F8>
void launch_row_as(const F16RowsView& v, hipStream_t s) {
    const dim3 g((v.count + kRowWaves - 1U) / kRowWaves), b(kRowWaves * kWave);
    if (static_cast<std::size_t>(v.rows) * v.width <= kXLds) hipLaunchKernelGGL(HIP_KERNEL_NAME(f16_rows_row_kernel<R, true, F8>), g, b, 0, s, v);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(f16_rows_row_kernel<R, false, F8>), g, b, 0, s, v);
}
template <unsigned R>
void launch_row(const F16RowsView& v, hipStream_t s) {
    if (v.weight_f8) launch_row_as<R, true>(v, s);
    else launch_row_as<R, false>(v, s);
}

bool row_form(const F16RowsView& v) noexcept {  // the form enqueue_f16_rows_lanes picks
    return v.width % 256U == 0U && (v.rows == 1 || (v.rows == 2 && 2ULL * v.width <= kXLds));
}
unsigned pair_kind(const F16RowsView& v) noexcept {
    return row_form(v) ? (static_cast<std::size_t>(v.rows) * v.width <= kXLds ? 0U : 1U) : 2U;
}
template <unsigned R, unsigned KA, unsigned KB>
void launch_pair_as(const F16RowsView& a, const F16RowsView& b, hipStream_t s) {
    constexpr unsigned threads = KA == 2U && KB == 2U ? 2U * kWave : kRowWaves * kWave;
    const auto blocks = [](const F16RowsView& v, unsigned k) {
        const unsigned outs = k == 2U ? threads / 4U : kRowWaves;
        return (v.count + outs - 1U) / outs;
    };
    const unsigned na = blocks(a, KA), nb = blocks(b, KB);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(f16_rows_pair_kernel<R, KA, KB, threads>), dim3(na + nb), dim3(threads), 0, s, a, b, na);
}
template <unsigned R, unsigned KA>
void launch_pair_b(const F16RowsView& a, const F16RowsView& b, unsigned kb, hipStream_t s) {
    if (kb == 0U) launch_pair_as<R, KA, 0U>(a, b, s);
    else if (kb == 1U) launch_pair_as<R, KA, 1U>(a, b, s);
    else launch_pair_as<R, KA, 2U>(a, b, s);
}
template <unsigned R>
void launch_pair(const F16RowsView& a, const F16RowsView& b, hipStream_t s) {
    if constexpr (R <= 2U) {
        const unsigned ka = pair_kind(a), kb = pair_kind(b);
        if (ka == 0U) launch_pair_b<R, 0U>(a, b, kb, s);
        else if (ka == 1U) launch_pair_b<R, 1U>(a, b, kb, s);
        else launch_pair_b<R, 2U>(a, b, kb, s);
    } else {
        launch_pair_as<R, 2U, 2U>(a, b, s);  // 3..8 rows: the lanes form only
    }
}

}  // namespace

bool f16_rows_lanes_supported(const F16RowsView& v) noexcept {
    const bool weights = v.weight_f8 ? reinterpret_cast<std::uintptr_t>(v.weight_f8) % 8U == 0 && (v.width + v.width / 32U) % 8U == 0
                                     : v.weight && reinterpret_cast<std::uintptr_t>(v.weight) % 16U == 0;
    return weights && v.x && v.out && v.count && v.width && v.width % 32U == 0 && v.rows >= 1 && v.rows <= 16U &&
           (v.rows == 1 || v.x_stride % 4U == 0) && reinterpret_cast<std::uintptr_t>(v.x) % 16U == 0;
}

void enqueue_f16_rows_lanes(const F16RowsView& v, void* stream) {
    if (!f16_rows_lanes_supported(v)) throw std::invalid_argument("F16 rows (lanes) view: width % 32, 16-byte aligned rows, 1..8 rows");
    const auto s = static_cast<hipStream_t>(stream);
    if (row_form(v)) {
        if (v.rows == 1) launch_row<1>(v, s);
        else launch_row<2>(v, s);
    } else if (v.rows == 1) launch_lanes<1>(v, s);
    else if (v.rows == 2) launch_lanes<2>(v, s);
    else if (v.rows <= 4) launch_lanes<4>(v, s);
    else if (v.rows <= kRows) launch_lanes<8>(v, s);
    else launch_lanes<16>(v, s);
    if (hipGetLastError() != hipSuccess) throw std::runtime_error("F16 rows (lanes) launch");
}

bool f16_rows_pair_supported(const F16RowsView& a, const F16RowsView& b) noexcept {
    return !a.weight_f8 && !b.weight_f8 && a.rows == b.rows && f16_rows_lanes_supported(a) && f16_rows_lanes_supported(b);
}

void enqueue_f16_rows_lanes_pair(const F16RowsView& a, const F16RowsView& b, void* stream) {
    if (!f16_rows_pair_supported(a, b)) throw std::invalid_argument("F16 rows (pair) views: F16 weights, the same 1..8 rows, lanes kernel views");
    const auto s = static_cast<hipStream_t>(stream);
    if (a.rows == 1) launch_pair<1>(a, b, s);
    else if (a.rows == 2) launch_pair<2>(a, b, s);
    else if (a.rows <= 4) launch_pair<4>(a, b, s);
    else if (a.rows <= kRows) launch_pair<8>(a, b, s);
    else launch_pair<16>(a, b, s);
    if (hipGetLastError() != hipSuccess) throw std::runtime_error("F16 rows (pair) launch");
}

}  // namespace strix::ds41
