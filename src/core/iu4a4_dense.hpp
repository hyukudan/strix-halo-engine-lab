#pragma once
// Shared A4W4 dense GEMM for prefill row batches on v_wmma_i32_16x16x16_iu4 (gfx11 integer WMMA, IU4 x IU4).
// Class FAST, contract dense-iu4a4-rot-v1 (iu4a4_rot.hpp: the bit-defined rule; the kernels here perform exactly its
// operations, the device fixture checks every output bit against iu4a4_rot::dot).
//
// Why a separate family: the operand class is integer (both operands 4-bit codes, int32 WMMA sums, integer group
// scales); the FP16 tall core (f16_dense_tall.hpp) is bound by the f16 WMMA rate (32 clk per 16x16x16 against 16 for
// iu4). The output side is shared: Cols / Store are the f16_dense_tall policies (ColumnTile, RawStore, Guarded, engine
// epilogues), so an engine binds this class with the epilogue it already has.
//
// Three kernels:
//   quantise: FP32 rows -> rotated (fwht128 per 128 inputs) 4-bit codes in the iu4 fragment tile layout, the integer
//     group scales u (one dword per row per 128 block = 4 groups of 32) and the row base. One wave per row.
//   requantise (load time): the IU4 g64 pack -> rotated per-128 codes (same tile layout) + FP32 steps in a tile order.
//   gemm: y[m][n] = base_m * sum_b fmaf(float(t_b), wstep[n][b], .) with t_b = sum_{g in b} u_g P_g (int32, exact).
//
// Geometry (gemm): the WMMA computes W . x^T: A operand = 16 weight rows n, B operand = 16 activation rows m. The i32
// result puts one m per lane (m = lane % 16) and 8 n per lane (n = 2 e + lane / 16), so the A-group scale u_g is a lane
// scalar shared by all W tiles of the wave, and a W block step is 8 floats per lane per W tile (2 x b128).
// Wave tile kTw W tiles x kTm x tiles; CTA kWavesN x kWavesM waves; grid x = m blocks (fastest: CTAs that run together
// share one weight block in L2), grid y = n blocks. Each A group's integer sum P_g restarts from a zero C (no carried
// sum, no lead term: one v_mad_i32_i24 per output per group, and one convert + one FMA per output per 128 inputs).
// Fragments are loaded straight from global memory (both layouts are fragment-ordered: 8 B per lane, 128 B per tile
// and k unit), one unit ahead.
// Limits (callers check supported()): in % 128 == 0, in <= iu4a4_rot::kMaxInputs, n % 16 == 0; quantise: x_stride % 4 == 0.
#include "f16_dense_tall.hpp"
#include "iu4a4_rot.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strix::core::iu4a4_dense {
namespace rot = iu4a4_rot;
inline constexpr const char* kIdentity = rot::kIdentity;
inline constexpr unsigned kTw = 4U, kTm = 2U;          // wave tile: W tiles x x tiles
inline constexpr unsigned kWavesN = 2U, kWavesM = 4U;  // CTA = 8 waves
inline constexpr unsigned kCtaTilesN = kTw * kWavesN, kCtaTilesM = kTm * kWavesM;
static_assert(rot::kBlock == 128U && rot::kGroupA == 32U, "the kernels walk 8 k units per block, 2 per A group");

inline bool supported(unsigned in, unsigned n) { return in > 0U && in % rot::kBlock == 0U && in <= rot::kMaxInputs && n > 0U && n % 16U == 0U; }
__host__ __device__ inline unsigned padded_rows(unsigned rows) { return (rows + 15U) & ~15U; }
// Activation operand of `rows` rows (rows padded to 16; pad rows: codes 0, u 1, base 0).
inline std::size_t act_code_bytes(unsigned rows, unsigned in) { return std::size_t(padded_rows(rows)) * in / 2U; }
inline std::size_t act_u_bytes(unsigned rows, unsigned in) { return std::size_t(padded_rows(rows)) * (in / rot::kBlock) * 4U; }
inline std::size_t act_base_bytes(unsigned rows) { return std::size_t(padded_rows(rows)) * 4U; }
// Rotated weights: codes in the pack tile layout (n * in / 2 bytes), steps [in/128][n/16][2][8] (n = 16 t + 2 e + h).
inline std::size_t weight_code_bytes(unsigned in, unsigned n) { return std::size_t(n) * in / 2U; }
inline std::size_t weight_step_bytes(unsigned in, unsigned n) { return std::size_t(n) * (in / rot::kBlock) * 4U; }
__host__ __device__ inline std::size_t step_index(unsigned b, unsigned n, unsigned outputs) {
    return ((std::size_t(b) * (outputs / 16U) + n / 16U) * 2U + (n % 2U)) * 8U + (n % 16U) / 2U;
}

struct Act {  // codes [in/16][mp/16][16][8] bytes (pack_byte(k, m, mp)); u [in/128][mp]; base [mp]
    std::uint8_t* codes;
    std::uint32_t* u;
    float* base;
    unsigned rows, in;
};
struct Weights {
    const std::uint8_t* codes;  // rotated pack, tile layout [in/16][n/16][16][8]
    const float* step;          // step_index order
    unsigned in, n;
};
// Cols policy for a plain row-major output: column tile i = outputs 16 i .. 16 i + 15 of y[r * stride + o].
struct OutCols {
    float* y;
    unsigned stride, tiles;
    __device__ __forceinline__ f16_dense_tall::ColumnTile tile(unsigned i) const { return {nullptr, y + 16U * i, stride, i < tiles}; }
};

namespace detail {
using I2 = int __attribute__((ext_vector_type(2)));
using I8 = int __attribute__((ext_vector_type(8)));
using F4 = float __attribute__((ext_vector_type(4)));
__device__ __forceinline__ float fmax_(float a, float o) { return o > a ? o : a; }

// One wave per row (blockDim 256 = 8 rows). Lanes 0-15 take block 2 s, lanes 16-31 block 2 s + 1 of step s; lane
// holds inputs 8 (lane % 16) .. + 7 of its block (the a4_tiles::Emu lane form, bitwise = fwht128).
__global__ __launch_bounds__(256) void quantise_kernel(const float* __restrict__ x, unsigned x_stride, Act a) {
#pragma clang fp contract(off)
    const unsigned lane = threadIdx.x & 31U, row = blockIdx.x * 8U + (threadIdx.x >> 5U);
    const unsigned mp = padded_rows(a.rows), in = a.in, blocks = in / rot::kBlock, steps = (blocks + 1U) / 2U;
    if (row >= mp) return;
    const unsigned half = lane >> 4U, sub = lane & 15U;
    const bool live_row = row < a.rows;
    const float* xr = x + std::size_t(live_row ? row : 0U) * x_stride;
    const auto load = [&](unsigned s, float (&v)[8]) __attribute__((always_inline)) {
        const unsigned blk = 2U * s + half;
        if (live_row && blk < blocks) {
            const F4* p = reinterpret_cast<const F4*>(xr + blk * rot::kBlock + sub * 8U);
            const F4 lo = p[0], hi = p[1];
            for (unsigned j = 0; j < 4U; ++j) v[j] = lo[j], v[4U + j] = hi[j];
        } else {
            for (unsigned j = 0; j < 8U; ++j) v[j] = 0.0f;
        }
        a4_tiles::Emu<32U, true>::fwht(v);
    };
    // Pass 1: row maximum of |rotated| and the first non-finite value in k order.
    float amax_m = 0.0f, bad = 0.0f;
    unsigned bad_k = 0xffffffffU;
    for (unsigned s = 0; s < steps; ++s) {
        float v[8];
        load(s, v);
        for (unsigned j = 0; j < 8U; ++j) {
            const float m = a4_tiles::fabs_(v[j]);
            if (!(m <= 3.402823466e38f)) {
                const unsigned k = (2U * s + half) * rot::kBlock + sub * 8U + j;
                if (k < bad_k) bad_k = k, bad = v[j];
            } else {
                amax_m = fmax_(amax_m, m);
            }
        }
    }
    for (unsigned d = 1; d < 32U; d *= 2U) {
        amax_m = fmax_(amax_m, __shfl_xor(amax_m, int(d), 32));
        const unsigned ok = static_cast<unsigned>(__shfl_xor(static_cast<int>(bad_k), int(d), 32));
        const float ov = __shfl_xor(bad, int(d), 32);
        if (ok < bad_k) bad_k = ok, bad = ov;
    }
    const bool finite = bad_k == 0xffffffffU;
    const float base = finite ? amax_m / (7.0f * static_cast<float>(rot::kUMax)) : bad;
    if (lane == 0U) a.base[row] = base;
    const bool coded = finite && base > 0.0f;
    // Pass 2: the same rotation (same bits), group scales, codes.
    for (unsigned s = 0; s < steps; ++s) {
        const unsigned blk = 2U * s + half;
        float v[8];
        load(s, v);
        float amax = 0.0f;
        for (unsigned j = 0; j < 8U; ++j) amax = fmax_(amax, a4_tiles::fabs_(v[j]));
        amax = fmax_(amax, __shfl_xor(amax, 1, 32));
        amax = fmax_(amax, __shfl_xor(amax, 2, 32));
        int ug = 1;
        if (coded && amax > 0.0f) {
            const float q = amax / (7.0f * base);
            ug = q >= static_cast<float>(rot::kUMax) ? rot::kUMax : static_cast<int>(::ceilf(q));
            ug = ug < 1 ? 1 : ug;
        }
        const float step = base * static_cast<float>(ug);
        std::uint32_t packed = 0U;
        for (unsigned j = 0; j < 8U; ++j) {
            const int code = coded ? integer_math::a4_code_f(v[j], step) : 0;
            packed |= (static_cast<std::uint32_t>(code) & 15U) << (4U * j);
        }
        // Group g = sub / 4 of the block; lanes 4 g of each half hold u_g.
        const unsigned hb = lane & 16U;
        const std::uint32_t uw = static_cast<std::uint32_t>(__shfl(ug, int(hb), 32)) |
                                 static_cast<std::uint32_t>(__shfl(ug, int(hb + 4U), 32)) << 8U |
                                 static_cast<std::uint32_t>(__shfl(ug, int(hb + 8U), 32)) << 16U |
                                 static_cast<std::uint32_t>(__shfl(ug, int(hb + 12U), 32)) << 24U;
        if (blk < blocks) {
            const unsigned k0 = blk * rot::kBlock + sub * 8U;
            *reinterpret_cast<std::uint32_t*>(a.codes + rot::pack_byte(k0, row, mp)) = packed;
            if (sub == 0U) a.u[std::size_t(blk) * mp + row] = uw;
        }
    }
}

// One thread per (column n, block b); load time.
__global__ __launch_bounds__(256) void requantise_kernel(const std::uint8_t* __restrict__ pack, const std::uint16_t* __restrict__ scales,
                                                         unsigned in, unsigned n, std::uint8_t* __restrict__ dst, float* __restrict__ step) {
    const std::size_t id = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned blocks = in / rot::kBlock;
    if (id >= std::size_t(n) * blocks) return;
    const unsigned col = static_cast<unsigned>(id % n), b = static_cast<unsigned>(id / n);
    step[step_index(b, col, n)] = rot::requantise_codes(pack, scales, in, n, col, b, dst);
}

template <class Cols, class Store>
__global__ __launch_bounds__(256) void gemm_kernel(Weights w, Act a, Cols cols, Store store) {
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, l16 = lane & 15U, hi = lane >> 4U;
    const unsigned ntiles = f16_dense_tall::detail::uni(w.n) / 16U, mp = padded_rows(f16_dense_tall::detail::uni(a.rows)), mtiles = mp / 16U;
    const unsigned in = f16_dense_tall::detail::uni(w.in), blocks = in / rot::kBlock;
    const unsigned nt0 = blockIdx.y * kCtaTilesN + (wave % kWavesN) * kTw, mt0 = blockIdx.x * kCtaTilesM + (wave / kWavesN) * kTm;
    if (mt0 >= mtiles) return;  // no live x tile (no barrier in this kernel)
    unsigned nt[kTw], mt[kTm];
#pragma unroll
    for (unsigned i = 0; i < kTw; ++i) nt[i] = nt0 + i < ntiles ? nt0 + i : ntiles - 1U;  // dead tiles read a live one
#pragma unroll
    for (unsigned i = 0; i < kTm; ++i) mt[i] = mt0 + i < mtiles ? mt0 + i : mtiles - 1U;
    const auto* wp = reinterpret_cast<const detail::I2*>(f16_dense_tall::detail::uniform(w.codes));
    const auto* xp = reinterpret_cast<const detail::I2*>(f16_dense_tall::detail::uniform(a.codes));
    const auto* sp = reinterpret_cast<const F4*>(f16_dense_tall::detail::uniform(w.step));
    // Fragment index of (unit ku, tile t): (ku * tiles + t) * 16 + l16; per unit the offsets grow by tiles * 16.
    const std::size_t wstride = std::size_t(ntiles) * 16U, xstride = std::size_t(mtiles) * 16U;
    float acc[kTw][kTm][8];
#pragma unroll
    for (unsigned i = 0; i < kTw; ++i)
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[i][j][e] = 0.0f;
    const I8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    I2 wf[kTw], xf[kTm], wn[kTw], xn[kTm];
    const auto load = [&](unsigned ku, I2 (&wd)[kTw], I2 (&xd)[kTm]) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned i = 0; i < kTw; ++i) wd[i] = wp[ku * wstride + nt[i] * 16U + l16];
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j) xd[j] = xp[ku * xstride + mt[j] * 16U + l16];
    };
    const unsigned last_unit = blocks * 8U - 1U;
    load(0U, wf, xf);
    for (unsigned b = 0; b < blocks; ++b) {
        std::uint32_t ub[kTm];
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j) ub[j] = a.u[std::size_t(b) * mp + mt[j] * 16U + l16];
        I8 t[kTw][kTm];
#pragma unroll
        for (unsigned g = 0; g < rot::kGroupsPerBlock; ++g) {
            I8 c[kTw][kTm];
#pragma unroll
            for (unsigned uu = 0; uu < 2U; ++uu) {
                const unsigned ku = b * 8U + g * 2U + uu;
                load(ku < last_unit ? ku + 1U : ku, wn, xn);  // one unit ahead (the last unit reloads itself)
#pragma unroll
                for (unsigned i = 0; i < kTw; ++i)
#pragma unroll
                    for (unsigned j = 0; j < kTm; ++j) {
                        if (uu == 0U) c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wf[i], true, xf[j], zero, false);
                        else c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wf[i], true, xf[j], c[i][j], false);
                    }
#pragma unroll
                for (unsigned i = 0; i < kTw; ++i) wf[i] = wn[i];
#pragma unroll
                for (unsigned j = 0; j < kTm; ++j) xf[j] = xn[j];
            }
#pragma unroll
            for (unsigned j = 0; j < kTm; ++j) {
                const int ug = static_cast<int>((ub[j] >> (8U * g)) & 255U);
#pragma unroll
                for (unsigned i = 0; i < kTw; ++i)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) t[i][j][e] = g == 0U ? __mul24(ug, c[i][j][e]) : __mul24(ug, c[i][j][e]) + t[i][j][e];
            }
        }
#pragma unroll
        for (unsigned i = 0; i < kTw; ++i) {
            const F4* s = sp + ((std::size_t(b) * ntiles + nt[i]) * 2U + hi) * 2U;
            const F4 s0 = s[0], s1 = s[1];
#pragma unroll
            for (unsigned j = 0; j < kTm; ++j)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e)
                    acc[i][j][e] = __builtin_fmaf(static_cast<float>(t[i][j][e]), e < 4U ? s0[e] : s1[e - 4U], acc[i][j][e]);
        }
    }
    store.run([&](auto& s) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j) {
            const unsigned m = (mt0 + j) * 16U + l16;
            if (mt0 + j >= mtiles) continue;
            const float bm = a.base[mt[j] * 16U + l16];
#pragma unroll
            for (unsigned i = 0; i < kTw; ++i) {
                if (nt0 + i >= ntiles) continue;
                const f16_dense_tall::ColumnTile ct = cols.tile(nt0 + i);
                if (!f16_dense_tall::detail::uni(ct.live ? 1U : 0U)) continue;
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const auto st = s.open(nt0 + i, ct, 2U * e + hi);
                    float v = acc[i][j][e] * bm;
                    asm volatile("" : "+v"(v));  // y = acc * base rounded once (no contraction into the epilogue)
                    if (m < a.rows) s.put(st, m, v);
                }
            }
        }
        s.close();
    });
}
}  // namespace detail

// Host side (stream-ordered; callers check supported() and hipGetLastError()).
inline void enqueue_quantise(const float* x, unsigned x_stride, const Act& a, hipStream_t s) {
    hipLaunchKernelGGL(detail::quantise_kernel, dim3((padded_rows(a.rows) + 7U) / 8U), dim3(256), 0, s, x, x_stride, a);
}
inline void enqueue_requantise(const std::uint8_t* pack, const std::uint16_t* scales, unsigned in, unsigned n, std::uint8_t* dst, float* step,
                               hipStream_t s) {
    const std::size_t threads = std::size_t(n) * (in / rot::kBlock);
    hipLaunchKernelGGL(detail::requantise_kernel, dim3(static_cast<unsigned>((threads + 255U) / 256U)), dim3(256), 0, s, pack, scales, in, n, dst, step);
}
template <class Cols, class Store = f16_dense_tall::RawStore>
inline void enqueue_gemm(const Weights& w, const Act& a, Cols cols, hipStream_t s, Store store = {}) {
    const unsigned mtiles = padded_rows(a.rows) / 16U, ntiles = w.n / 16U;
    const dim3 grid((mtiles + kCtaTilesM - 1U) / kCtaTilesM, (ntiles + kCtaTilesN - 1U) / kCtaTilesN);
    hipLaunchKernelGGL((detail::gemm_kernel<Cols, Store>), grid, dim3(32U * kWavesN * kWavesM), 0, s, w, a, cols, store);
}
}  // namespace strix::core::iu4a4_dense
