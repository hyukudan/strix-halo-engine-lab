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
//   requantise (load time): a weight source (the IU4 g64 pack; microscaled expert matrices) -> rotated per-128 codes
//     (same tile layout) + FP32 steps in a tile order.
//   gemm: y[m][n] = base_m * sum_b fmaf(float(t_b), wstep[n][b], .) with t_b = sum_{g in b} u_g P_g (int32, exact).
//     Traits: dense (grid over the m x n blocks of one matrix) and grouped (GroupCta: one CTA per group tile of <= 128
//     gathered rows with its own weight image, e.g. routed expert tiles; plain or paired gate / up epilogue).
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
// The GEMM copies codes, steps and u in 16 B pieces: Act / Weights buffers start 16 B aligned (as hipMalloc returns).
#include "f16_dense_tall.hpp"
#include "iu4a4_rot.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

// Grouped trait (expert tiles; same contract per output): one CTA = 1..kCtaTilesM * 16 rows of one group tile that
// share one weight image; grid x = tiles, grid y = blocks of kCtaTilesN W tiles, grid z = row slices of taller tiles. Src::active(t) refuses tiles past a
// device count (uniform, before any barrier); Src::bind(t) gives the CTA's operands. Activation row r of the tile is
// row map ? map[first + r] : first + r of the Act image (dense layout: e.g. a token image read through a routing map,
// or rows already in tile order). The CTA stores rows r < rows only.
struct GroupCta {
    Weights w;                      // the group's rotated image (in = Act::in, n = the launch's n)
    const std::uint32_t* map;       // optional image row map (an entry >= Act::rows reads the last row and faults)
    unsigned first, rows;
    float* y;                       // output row r at y + r * y_row
    std::size_t y_row;
    unsigned* fault;                // optional: kBoundsFault is OR-ed in on a map entry past the image
};
inline constexpr unsigned kBoundsFault = 1U << 30U;
// Epilogues: PlainOut y[r][o] = acc * base (the dense rounding). A paired epilogue (kPaired = true, float operator()(gate,
// up)) reads W tiles 2 c and 2 c + 1 as the two halves of output columns 16 c .. 16 c + 15 and stores h[r][16 c + j] =
// epi(gate, up) (the weight image is built in that column order: iu4a4_rot::paired_column), n / 2 outputs per row.
struct PlainOut {
    static constexpr bool kPaired = false;
};

namespace detail {
using I2 = int __attribute__((ext_vector_type(2)));
using I8 = int __attribute__((ext_vector_type(8)));
using F4 = float __attribute__((ext_vector_type(4)));
__device__ __forceinline__ float fmax_(float a, float o) { return o > a ? o : a; }
using I4 = int __attribute__((ext_vector_type(4)));
typedef const __attribute__((address_space(1))) I4 GI4;
typedef const __attribute__((address_space(1))) F4 GF4;
typedef const __attribute__((address_space(1))) std::uint32_t GU32;

// One wave per row (blockDim 256 = 8 rows). Lanes 0-15 take block 2 s, lanes 16-31 block 2 s + 1 of step s; lane
// holds inputs 8 (lane % 16) .. + 7 of its block (the a4_tiles::Emu lane form, bitwise = fwht128).
// Output layout of the quantisers: Rows = false the fragment tile layout (codes pack_byte(k, m, mp), u [blk][mp]), Rows =
// true the row-major image the grouped launch gathers (codes [m][in / 2], u [m][in / 128]); same values.
template <bool Rows>
__device__ __forceinline__ std::size_t code_offset(unsigned k0, unsigned row, unsigned mp, unsigned in) {
    return Rows ? std::size_t(row) * (in / 2U) + k0 / 2U : rot::pack_byte(k0, row, mp);
}
template <bool Rows>
__device__ __forceinline__ std::size_t u_offset(unsigned blk, unsigned row, unsigned mp, unsigned in) {
    return Rows ? std::size_t(row) * (in / rot::kBlock) + blk : std::size_t(blk) * mp + row;
}
template <bool Rows>
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
            *reinterpret_cast<std::uint32_t*>(a.codes + code_offset<Rows>(k0, row, mp, in)) = packed;
            if (sub == 0U) a.u[u_offset<Rows>(blk, row, mp, in)] = uw;
        }
    }
}

// CTA form of the quantiser (one 8-wave CTA per row, one read of x): wave w takes steps w, w + 8, ... (a step = 2
// blocks, as quantise_kernel) and keeps its rotated values in registers; the row maximum and the first non-finite
// value meet in LDS. Same operations, same bits as quantise_kernel. F::apply(a, b) maps two inputs to the value that
// is quantised (an engine's gated unit, e.g. SiLU(g) * u, so the FP32 product is never stored); Ident takes a.
// Rows up to kCtaInputs inputs; quantise_kernel takes longer rows.
inline constexpr unsigned kCtaSteps = 12U, kCtaInputs = 8U * kCtaSteps * 2U * 128U;  // 24,576
struct Ident {
    __device__ __forceinline__ static float apply(float a, float) { return a; }
};
template <class F, bool Rows>
__device__ __forceinline__ void quantise_cta_row(const float* __restrict__ x0, const float* __restrict__ x1, unsigned x_stride, Act a) {
#pragma clang fp contract(off)
    __shared__ float s_amax[8], s_bad[8];
    __shared__ unsigned s_badk[8];
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, row = blockIdx.x;
    const unsigned mp = padded_rows(a.rows), in = a.in, blocks = in / rot::kBlock, steps = (blocks + 1U) / 2U;
    const unsigned half = lane >> 4U, sub = lane & 15U;
    const bool live_row = row < a.rows;
    const std::size_t ro = std::size_t(live_row ? row : 0U) * x_stride;
    float v[kCtaSteps][8];
    float amax_m = 0.0f, bad = 0.0f;
    unsigned bad_k = 0xffffffffU;
#pragma unroll
    for (unsigned q = 0; q < kCtaSteps; ++q) {
        const unsigned blk = 2U * (wave + 8U * q) + half;
        if (live_row && blk < blocks) {
            const std::size_t at = ro + blk * rot::kBlock + sub * 8U;
            const F4 lo = *reinterpret_cast<const F4*>(x0 + at), hi = *reinterpret_cast<const F4*>(x0 + at + 4U);
            F4 lo1 = lo, hi1 = hi;
            if (x1) lo1 = *reinterpret_cast<const F4*>(x1 + at), hi1 = *reinterpret_cast<const F4*>(x1 + at + 4U);
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                float t0 = F::apply(lo[j], lo1[j]), t1 = F::apply(hi[j], hi1[j]);
                asm volatile("" : "+v"(t0), "+v"(t1));  // the value rounded once, as a stored FP32 input
                v[q][j] = t0, v[q][4U + j] = t1;
            }
        } else {
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) v[q][j] = 0.0f;
        }
        if (wave + 8U * q < steps) {  // wave-uniform: every lane of the wave takes part in the shuffles
            a4_tiles::Emu<32U, true>::fwht(v[q]);
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const float m = a4_tiles::fabs_(v[q][j]);
                if (!(m <= 3.402823466e38f)) {
                    const unsigned k = blk * rot::kBlock + sub * 8U + j;
                    if (k < bad_k) bad_k = k, bad = v[q][j];
                } else {
                    amax_m = fmax_(amax_m, m);
                }
            }
        }
    }
    for (unsigned d = 1; d < 32U; d *= 2U) {
        amax_m = fmax_(amax_m, __shfl_xor(amax_m, int(d), 32));
        const unsigned ok = static_cast<unsigned>(__shfl_xor(static_cast<int>(bad_k), int(d), 32));
        const float ov = __shfl_xor(bad, int(d), 32);
        if (ok < bad_k) bad_k = ok, bad = ov;
    }
    if (lane == 0U) s_amax[wave] = amax_m, s_bad[wave] = bad, s_badk[wave] = bad_k;
    __syncthreads();
    for (unsigned o = 0; o < 8U; ++o) {
        amax_m = fmax_(amax_m, s_amax[o]);
        if (s_badk[o] < bad_k) bad_k = s_badk[o], bad = s_bad[o];
    }
    const bool finite = bad_k == 0xffffffffU;
    const float base = finite ? amax_m / (7.0f * static_cast<float>(rot::kUMax)) : bad;
    if (threadIdx.x == 0U) a.base[row] = base;
    const bool coded = finite && base > 0.0f;
#pragma unroll
    for (unsigned q = 0; q < kCtaSteps; ++q) {
        if (wave + 8U * q >= steps) continue;
        const unsigned blk = 2U * (wave + 8U * q) + half;
        float amax = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) amax = fmax_(amax, a4_tiles::fabs_(v[q][j]));
        amax = fmax_(amax, __shfl_xor(amax, 1, 32));
        amax = fmax_(amax, __shfl_xor(amax, 2, 32));
        int ug = 1;
        if (coded && amax > 0.0f) {
            const float qq = amax / (7.0f * base);
            ug = qq >= static_cast<float>(rot::kUMax) ? rot::kUMax : static_cast<int>(::ceilf(qq));
            ug = ug < 1 ? 1 : ug;
        }
        const float step = base * static_cast<float>(ug);
        std::uint32_t packed = 0U;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const int code = coded ? integer_math::a4_code_f(v[q][j], step) : 0;
            packed |= (static_cast<std::uint32_t>(code) & 15U) << (4U * j);
        }
        const unsigned hb = lane & 16U;
        const std::uint32_t uw = static_cast<std::uint32_t>(__shfl(ug, int(hb), 32)) |
                                 static_cast<std::uint32_t>(__shfl(ug, int(hb + 4U), 32)) << 8U |
                                 static_cast<std::uint32_t>(__shfl(ug, int(hb + 8U), 32)) << 16U |
                                 static_cast<std::uint32_t>(__shfl(ug, int(hb + 12U), 32)) << 24U;
        if (blk < blocks) {
            *reinterpret_cast<std::uint32_t*>(a.codes + code_offset<Rows>(blk * rot::kBlock + sub * 8U, row, mp, a.in)) = packed;
            if (sub == 0U) a.u[u_offset<Rows>(blk, row, mp, a.in)] = uw;
        }
    }
}
template <class F, bool Rows>
__global__ __launch_bounds__(256) void quantise_cta_kernel(const float* __restrict__ x0, const float* __restrict__ x1, unsigned x_stride, Act a) {
    quantise_cta_row<F, Rows>(x0, x1, x_stride, a);
}
// Verdict-gated form: the launch stores nothing when *verdict != 0. The word must not change during the launch (every
// wave reads the same value before the first barrier).
template <class F, bool Rows>
__global__ __launch_bounds__(256) void quantise_cta_gated_kernel(const unsigned* __restrict__ verdict, const float* __restrict__ x0,
                                                                 const float* __restrict__ x1, unsigned x_stride, Act a) {
    if (*verdict != 0U) return;
    quantise_cta_row<F, Rows>(x0, x1, x_stride, a);
}

// One thread per (column n, block b); load time.
// Load-time requantisation (one thread per column and 128 block) from a weight source: Src::column(n, b) gives the value
// functor of iu4a4_rot::requantise_values (the IU4 g64 pack: G64Source; expert images: moe_core/grouped_iu4a4.hpp).
template <class Src>
__global__ __launch_bounds__(256) void requantise_kernel(Src src, unsigned in, unsigned n, std::uint8_t* __restrict__ dst, float* __restrict__ step) {
    const std::size_t id = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned blocks = in / rot::kBlock;
    if (id >= std::size_t(n) * blocks) return;
    const unsigned col = static_cast<unsigned>(id % n), b = static_cast<unsigned>(id / n);
    step[step_index(b, col, n)] = src.requantise(col, b, n, dst);
}
struct G64Source {
    const std::uint8_t* pack;
    const std::uint16_t* scales;
    unsigned in;
    rot::WeightRule rule = rot::WeightRule::amax;
    __device__ __forceinline__ float requantise(unsigned col, unsigned b, unsigned n, std::uint8_t* dst) const {
        return rot::requantise_codes(pack, scales, in, n, col, b, dst, rule);
    }
};

// GEMM main loop. The CTA stages each 128-input block of its 8 W tiles and 8 x tiles in LDS (2 KB per K16 unit:
// W tiles then x tiles, 128 B each = the global fragment layout, so a piece is one 16 B copy), double buffered: the
// block b + 1 copy is loaded at the start of block b and stored after it, one barrier per block. The block's side data
// rides in the same copy (side area: the 8 W tiles' steps, 64 B each, then the 8 x tiles' u, 64 B each), so no global
// latency sits inside the barrier-synchronised block (measured: global side loads there cost ~25 %). Waves read their
// fragments with ds_load_b64 from two base registers + immediate offsets (no per-fragment address arithmetic), one
// unit ahead. The arithmetic per group / block is the contract's (iu4a4_rot::dot), unchanged.
inline constexpr unsigned kUnitBytes = 2048U, kCodeBytes = 8U * kUnitBytes, kSideBytes = 1024U, kBlockBytes = kCodeBytes + kSideBytes;
static_assert(kCtaTilesN == 8U && kCtaTilesM == 8U, "a staged unit holds 8 W tiles and 8 x tiles (64 x 16 B per side)");
// The contract arithmetic of one staged 128-input block `buf` (codes, then side data) for a wave's kTw x kTm tiles,
// added to acc, for the grouped trait: the integer accumulator c holds one x tile at a time (kTw x 8 registers instead
// of the dense launch's kTw x kTm x 8; with the dense form the grouped launch spilled 38-40 VGPRs, its copy staging, in
// the K loop). Every t is the same exact integer sum in the same group order as the dense launch, so acc is the same
// bits. The group fold (t (+)= u_g * c) and the block scale (acc += float(t) * step) are written in both bodies on
// purpose: the grouped launch sits at 256 VGPRs, and moving either step into a shared inline function (whole block or
// one W tile) changes its schedule to 396 B of scratch (12 / 16 B inline), while the fold alone reorders both kernels;
// this body inside the dense launch loses its one-unit-ahead fragment loads (256 VGPRs, 7 spills). Keep the two
// bodies' arithmetic statement for statement equal (the fixtures compare both against iu4a4_rot::dot bit for bit).
__device__ __forceinline__ void block_dot(const std::uint8_t* buf, unsigned wn, unsigned wm, unsigned l16, unsigned hi, float (&acc)[kTw][kTm][8]) {
    const I2* wl = reinterpret_cast<const I2*>(buf + wn * (kTw * 128U) + l16 * 8U);
    const I2* xl = reinterpret_cast<const I2*>(buf + 1024U + wm * (kTm * 128U) + l16 * 8U);
    const std::uint8_t* sd = buf + kCodeBytes;
    std::uint32_t ub[kTm];
#pragma unroll
    for (unsigned j = 0; j < kTm; ++j) ub[j] = reinterpret_cast<const std::uint32_t*>(sd + 512U + (wm * kTm + j) * 64U)[l16];
    I8 t[kTw][kTm];
    const I8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (unsigned g = 0; g < rot::kGroupsPerBlock; ++g) {
        I2 w0[kTw], w1[kTw];
#pragma unroll
        for (unsigned i = 0; i < kTw; ++i) {
            w0[i] = wl[((2U * g) * kUnitBytes + i * 128U) / 8U];
            w1[i] = wl[((2U * g + 1U) * kUnitBytes + i * 128U) / 8U];
        }
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j) {
            const I2 x0 = xl[((2U * g) * kUnitBytes + j * 128U) / 8U], x1 = xl[((2U * g + 1U) * kUnitBytes + j * 128U) / 8U];
            I8 c[kTw];
#pragma unroll
            for (unsigned i = 0; i < kTw; ++i) c[i] = STRIX_HIP_WMMA_IU4(true, w0[i], true, x0, zero, false);
#pragma unroll
            for (unsigned i = 0; i < kTw; ++i) c[i] = STRIX_HIP_WMMA_IU4(true, w1[i], true, x1, c[i], false);
            const int ug = static_cast<int>((ub[j] >> (8U * g)) & 255U);
#pragma unroll
            for (unsigned i = 0; i < kTw; ++i)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) t[i][j][e] = g == 0U ? __mul24(ug, c[i][e]) : __mul24(ug, c[i][e]) + t[i][j][e];
        }
    }
#pragma unroll
    for (unsigned i = 0; i < kTw; ++i) {
        const F4* sb = reinterpret_cast<const F4*>(sd + (wn * kTw + i) * 64U + hi * 32U);
        const F4 s0 = sb[0], s1 = sb[1];
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e)
                acc[i][j][e] = __builtin_fmaf(static_cast<float>(t[i][j][e]), e < 4U ? s0[e] : s1[e - 4U], acc[i][j][e]);
    }
}
// Epilogue staging: a wave's 32 rows x 32 columns at a row stride of 36 floats (16 B aligned rows; 36 l16 mod 64 + hi
// hits 32 distinct banks per store). kStagedLds = the operand buffers or the 8 waves' regions, whichever is larger.
inline constexpr unsigned kStageStride = 36U;
inline constexpr unsigned kStagedLds = 2U * kBlockBytes > kWavesN * kWavesM * kTm * 16U * kStageStride * 4U
                                           ? 2U * kBlockBytes
                                           : kWavesN * kWavesM * kTm * 16U * kStageStride * 4U;
// Staged stores of the dense trait: RawStore, or a store with put_chunk(ct, r, c, v) = the 4 values of row r, columns
// c .. c + 3 of column tile ct (c % 4 == 0), each exactly as put() would store it.
template <class S> concept ChunkStore = requires(const S& s, const f16_dense_tall::ColumnTile& ct, F4 v) { s.put_chunk(ct, 0U, 0U, v); };
template <class S> inline constexpr bool kStagedStore = ChunkStore<S> || std::is_same_v<S, f16_dense_tall::RawStore>;
__device__ __forceinline__ void store_chunk(const f16_dense_tall::ColumnTile& ct, unsigned r, unsigned c, F4 v) {
    float* const y = ct.y + std::size_t(r) * ct.stride + c;
    if ((reinterpret_cast<std::uintptr_t>(y) & 15U) == 0U) {
        *reinterpret_cast<F4*>(y) = v;
    } else {
#pragma unroll
        for (unsigned q = 0; q < 4U; ++q) y[q] = v[q];
    }
}
template <class S>
__device__ __forceinline__ void put_chunk(const S& s, const f16_dense_tall::ColumnTile& ct, unsigned r, unsigned c, F4 v) {
    if constexpr (ChunkStore<S>) s.put_chunk(ct, r, c, v);
    else store_chunk(ct, r, c, v);
}
template <class Cols, class Store>
__global__ __launch_bounds__(256) void gemm_kernel(Weights w, Act a, Cols cols, Store store) {
    __shared__ __attribute__((aligned(16))) std::uint8_t lds[kStagedStore<Store> ? kStagedLds : 2U * kBlockBytes];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, l16 = lane & 15U, hi = lane >> 4U;
    const unsigned ntiles = f16_dense_tall::detail::uni(w.n) / 16U, mp = padded_rows(f16_dense_tall::detail::uni(a.rows)), mtiles = mp / 16U;
    const unsigned in = f16_dense_tall::detail::uni(w.in), blocks = in / rot::kBlock;
    const unsigned wn = wave % kWavesN, wm = wave / kWavesN;
    // Column blocks run fastest (grid x): CTAs in flight share the same x rows and write neighbouring 512 B windows of
    // those rows. Rows fastest made every CTA in flight write the same window of different rows, which with a row pitch
    // of a multiple of 4 KiB lands on the same memory channels (2x slower stores).
    const unsigned cta_n = blockIdx.x, cta_m = blockIdx.y;
    const unsigned nt0 = cta_n * kCtaTilesN + wn * kTw, mt0 = cta_m * kCtaTilesM + wm * kTm;
    const bool live = mt0 < mtiles;  // a wave without a live x tile still copies and meets the barriers
    unsigned nt[kTw], mt[kTm];
#pragma unroll
    for (unsigned i = 0; i < kTw; ++i) nt[i] = nt0 + i < ntiles ? nt0 + i : ntiles - 1U;  // dead tiles read a live one
#pragma unroll
    for (unsigned i = 0; i < kTm; ++i) mt[i] = mt0 + i < mtiles ? mt0 + i : mtiles - 1U;
    // Copy role: piece q of a block = unit 2 q + (tid >> 7); waves 0-1 / 4-5 copy W, 2-3 / 6-7 copy x; 16 B at tid % 64.
    const unsigned side = (tid >> 6U) & 1U, piece = tid & 63U, cu = tid >> 7U;
    const unsigned stiles = side == 0U ? ntiles : mtiles, cbase = side == 0U ? cta_n * kCtaTilesN : cta_m * kCtaTilesM;
    const unsigned ctile = cbase + (piece >> 3U) < stiles ? cbase + (piece >> 3U) : stiles - 1U;
    const std::size_t ustride = std::size_t(stiles) * 8U;  // one K16 unit of this side, in 16 B pieces
    const GI4* src = (side == 0U ? (const GI4*)w.codes : (const GI4*)a.codes) + std::size_t(cu) * ustride + std::size_t(ctile) * 8U + (piece & 7U);
    const unsigned dst = cu * kUnitBytes + side * 1024U + piece * 16U;
    // Side copy (waves 0-1): piece p < 32: steps of W tile p / 4 (16 floats = 4 pieces); p >= 32: u of x tile (p - 32) / 4.
    const bool scopy = tid < 64U;
    const unsigned sq = tid & 31U, sstile = (tid < 32U ? cta_n * kCtaTilesN : cta_m * kCtaTilesM) + (sq >> 2U);
    const unsigned slim = tid < 32U ? ntiles : mtiles, stc = sstile < slim ? sstile : slim - 1U;
    const std::size_t sstride = tid < 32U ? std::size_t(ntiles) * 4U : std::size_t(mp) / 4U;  // one block, in 16 B
    const GI4* ssrc = (tid < 32U ? (const GI4*)w.step : (const GI4*)a.u) + std::size_t(stc) * 4U + (sq & 3U);
    I4 stg[4], sstg = {0, 0, 0, 0};
    const auto fetch = [&](unsigned b) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned q = 0; q < 4U; ++q) stg[q] = src[(std::size_t(b) * 8U + 2U * q) * ustride];
        if (scopy) sstg = ssrc[std::size_t(b) * sstride];
    };
    const auto put = [&](unsigned buf) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned q = 0; q < 4U; ++q) *reinterpret_cast<I4*>(lds + buf * kBlockBytes + dst + q * 2U * kUnitBytes) = stg[q];
        if (scopy) *reinterpret_cast<I4*>(lds + buf * kBlockBytes + kCodeBytes + tid * 16U) = sstg;
    };
    float acc[kTw][kTm][8];
#pragma unroll
    for (unsigned i = 0; i < kTw; ++i)
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[i][j][e] = 0.0F;
    const I8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    I2 wa[kTw], xa[kTm], wb[kTw], xb[kTm];  // fragment sets in ping-pong (named: no ring copies)
    const auto mma = [&](const I2 (&wd)[kTw], const I2 (&xd)[kTm], I8 (&c)[kTw][kTm], bool first) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned i = 0; i < kTw; ++i)
#pragma unroll
            for (unsigned j = 0; j < kTm; ++j)
                c[i][j] = STRIX_HIP_WMMA_IU4(true, wd[i], true, xd[j], first ? zero : c[i][j], false);
    };
    fetch(0U);
    put(0U);
    __syncthreads();
    for (unsigned b = 0; b < blocks; ++b) {
        const bool more = b + 1U < blocks;
        if (more) fetch(b + 1U);
        if (live) {
            const I2* wl = reinterpret_cast<const I2*>(lds + (b & 1U) * kBlockBytes + wn * (kTw * 128U) + l16 * 8U);
            const I2* xl = reinterpret_cast<const I2*>(lds + (b & 1U) * kBlockBytes + 1024U + wm * (kTm * 128U) + l16 * 8U);
            const auto load = [&](unsigned k, I2 (&wd)[kTw], I2 (&xd)[kTm]) __attribute__((always_inline)) {
#pragma unroll
                for (unsigned i = 0; i < kTw; ++i) wd[i] = wl[(k * kUnitBytes + i * 128U) / 8U];
#pragma unroll
                for (unsigned j = 0; j < kTm; ++j) xd[j] = xl[(k * kUnitBytes + j * 128U) / 8U];
            };
            const std::uint8_t* sd = lds + (b & 1U) * kBlockBytes + kCodeBytes;
            std::uint32_t ub[kTm];
#pragma unroll
            for (unsigned j = 0; j < kTm; ++j) ub[j] = reinterpret_cast<const std::uint32_t*>(sd + 512U + (wm * kTm + j) * 64U)[l16];
            I8 t[kTw][kTm];
            load(0U, wa, xa);
#pragma unroll
            for (unsigned g = 0; g < rot::kGroupsPerBlock; ++g) {
                I8 c[kTw][kTm];
                load(2U * g + 1U, wb, xb);
                mma(wa, xa, c, true);
                if (g + 1U < rot::kGroupsPerBlock) load(2U * g + 2U, wa, xa);
                mma(wb, xb, c, false);
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
                const F4* sb = reinterpret_cast<const F4*>(sd + (wn * kTw + i) * 64U + hi * 32U);
                const F4 s0 = sb[0], s1 = sb[1];
#pragma unroll
                for (unsigned j = 0; j < kTm; ++j)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e)
                        acc[i][j][e] = __builtin_fmaf(static_cast<float>(t[i][j][e]), e < 4U ? s0[e] : s1[e - 4U], acc[i][j][e]);
            }
        }
        if (more) put((b + 1U) & 1U);
        __syncthreads();
    }
    if (!live) return;  // after the last barrier
    if constexpr (kStagedStore<Store>) {
        // Epilogue through LDS (the operand buffers are dead after the last barrier). Direct stores put 16 rows x 8 B per
        // instruction; at a row pitch of a multiple of 4 KiB those 16 lines fall on one channel (8 % slower at a pitch
        // of 3,072 floats, 8K rows, with column blocks first). Staged: each wave writes its 32 rows x 32
        // columns into a private region, then stores whole 16 B chunks, 8 lanes per row (128 B contiguous). Same values,
        // same rounding (one product per output, no contraction); only the store order changes.
        float* const st = reinterpret_cast<float*>(lds) + wave * (kTm * 16U * kStageStride);
        float bm[kTm];
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j) bm[j] = a.base[mt[j] * 16U + l16];
        store.run([&](auto& s) __attribute__((always_inline)) {
#pragma unroll
            for (unsigned p = 0; p < kTw / 2U; ++p) {
#pragma unroll
                for (unsigned j = 0; j < kTm; ++j)
#pragma unroll
                    for (unsigned k = 0; k < 2U; ++k)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) {
                            float v = acc[2U * p + k][j][e] * bm[j];
                            asm volatile("" : "+v"(v));  // y = acc * base rounded once (no contraction into the epilogue)
                            st[(j * 16U + l16) * kStageStride + k * 16U + 2U * e + hi] = v;
                        }
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
                for (unsigned it = 0; it < kTm * 16U / 4U; ++it) {
                    const unsigned row = it * 4U + lane / 8U, ch = lane % 8U, m = mt0 * 16U + row, tile = nt0 + 2U * p + ch / 4U;
                    if (m >= a.rows || tile >= ntiles) continue;
                    const f16_dense_tall::ColumnTile ct = cols.tile(tile);
                    if (!ct.live) continue;
                    put_chunk(s, ct, m, (ch % 4U) * 4U, *reinterpret_cast<const F4*>(st + row * kStageStride + ch * 4U));
                }
                if (p + 1U < kTw / 2U) {
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");  // the next phase overwrites the region
                    __builtin_amdgcn_wave_barrier();
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                }
            }
            s.close();
        });
        return;
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

// Grouped trait of the GEMM (see GroupCta): the dense W copy, block_dot and output rounding. The x side reads a ROW-MAJOR
// A4 image (enqueue_quantise_rows: a row's codes contiguous, u [row][blocks]): each x-side thread copies the 64 B of one
// gathered row per block (4 x 16 B, two K16 units each) into the same LDS fragment layout, so a gathered row costs whole
// lines. Grid: x = blocks of kCtaTilesN W tiles (fastest: the CTAs of one tile slice share its rows in L2), y = tile *
// slices + slice (the slices and tiles of one expert run together and share its weight block).
// Epilogue staging of the grouped trait: kStageStride / kGroupedLds (with the dense trait's staged store, above).
inline constexpr unsigned kGroupedLds = kStagedLds;
template <class Src, class Epi>
__global__ __launch_bounds__(256) void grouped_gemm_kernel(Src src, Act a, unsigned n, unsigned slices, Epi epi) {
    __shared__ __attribute__((aligned(16))) std::uint8_t lds[kGroupedLds];
    const unsigned tile = blockIdx.y / slices, slice = blockIdx.y % slices;
    if (!src.active(tile)) return;  // uniform, before any barrier
    GroupCta g = src.bind(tile);
    // Row slice of the tile: rows [128 slice, 128 slice + 128) (a plan tile taller than one CTA).
    const unsigned skip = slice * kCtaTilesM * 16U, all = f16_dense_tall::detail::uni(g.rows);
    if (all <= skip) return;
    const unsigned rows = all - skip < kCtaTilesM * 16U ? all - skip : kCtaTilesM * 16U;
    g.first += skip;
    g.y += std::size_t(skip) * g.y_row;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, l16 = lane & 15U, hi = lane >> 4U;
    const unsigned ntiles = f16_dense_tall::detail::uni(n) / 16U, in = f16_dense_tall::detail::uni(a.in);
    const unsigned mtiles = (rows + 15U) / 16U, blocks = in / rot::kBlock;
    const unsigned wn = wave % kWavesN, wm = wave / kWavesN;
    const unsigned nt0 = blockIdx.x * kCtaTilesN + wn * kTw, mt0 = wm * kTm;
    const bool live = mt0 < mtiles;  // a wave without a live x tile still copies and meets the barriers
    const auto image_row = [&](unsigned r) __attribute__((always_inline)) {
        const unsigned c = g.first + (r < rows ? r : rows - 1U);  // rows past the tile re-read its last row (never stored)
        if (!g.map) return c;
        const unsigned m = g.map[c];
        if (m < a.rows) return m;
        if (g.fault) atomicOr(g.fault, kBoundsFault);  // a map entry past the image: read its last row, report
        return a.rows - 1U;
    };
    // W copy (waves 0-1 / 4-5, as the dense launch): piece q of a block = unit 2 q + (tid >> 7).
    const unsigned side = (tid >> 6U) & 1U, piece = tid & 63U, cu = tid >> 7U;
    const unsigned wdst = cu * kUnitBytes + piece * 16U;
    const std::size_t wstride = std::size_t(ntiles) * 8U;  // one K16 unit of W, in 16 B pieces
    const unsigned wtile = blockIdx.x * kCtaTilesN + (piece >> 3U), wct = wtile < ntiles ? wtile : ntiles - 1U;
    const GI4* wsrc = (const GI4*)g.w.codes + std::size_t(cu) * wstride + std::size_t(wct) * 8U + (piece & 7U);
    // x copy (waves 2-3 / 6-7): thread = CTA row xi (tile xi / 16, row xi % 16); piece q = units 2 q, 2 q + 1.
    const unsigned xi = piece + 64U * cu, xdst = 1024U + (xi / 16U) * 128U + (xi % 16U) * 8U;
    const GI4* xsrc = (const GI4*)(a.codes + std::size_t(image_row(xi)) * (in / 2U));
    // Side copy: tid < 32 the steps of W tile tid / 4 (as the dense launch); 32 <= tid < 64 the u of 4 rows of an x tile.
    const unsigned sq = tid & 31U;
    const bool wside = tid < 32U, xside = tid >= 32U && tid < 64U;
    const unsigned stile = blockIdx.x * kCtaTilesN + (sq >> 2U), stc = stile < ntiles ? stile : ntiles - 1U;
    const GI4* ssrc = (const GI4*)g.w.step + std::size_t(stc) * 4U + (sq & 3U);
    const std::size_t sstride = std::size_t(ntiles) * 4U;
    const unsigned ut = (sq >> 2U) < mtiles ? (sq >> 2U) : mtiles - 1U;
    std::size_t ur[4];
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) ur[i] = std::size_t(image_row(ut * 16U + (sq & 3U) * 4U + i)) * blocks;
    const GU32* ug = (const GU32*)a.u;
    I4 stg[4], sstg = {0, 0, 0, 0};
    const auto fetch = [&](unsigned b) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned q = 0; q < 4U; ++q) stg[q] = side == 0U ? wsrc[(std::size_t(b) * 8U + 2U * q) * wstride] : xsrc[std::size_t(b) * 4U + q];
        if (wside) {
            sstg = ssrc[std::size_t(b) * sstride];
        } else if (xside) {
            const I4 words = {static_cast<int>(ug[ur[0] + b]), static_cast<int>(ug[ur[1] + b]), static_cast<int>(ug[ur[2] + b]),
                              static_cast<int>(ug[ur[3] + b])};
            sstg = words;
        }
    };
    const bool scopy = tid < 64U;
    const auto put = [&](unsigned buf) __attribute__((always_inline)) {
        std::uint8_t* base = lds + buf * kBlockBytes;
#pragma unroll
        for (unsigned q = 0; q < 4U; ++q) {
            if (side == 0U) {
                *reinterpret_cast<I4*>(base + wdst + q * 2U * kUnitBytes) = stg[q];
            } else {
                const I2 lo = {stg[q][0], stg[q][1]}, hv = {stg[q][2], stg[q][3]};
                *reinterpret_cast<I2*>(base + xdst + 2U * q * kUnitBytes) = lo;
                *reinterpret_cast<I2*>(base + xdst + (2U * q + 1U) * kUnitBytes) = hv;
            }
        }
        if (scopy) *reinterpret_cast<I4*>(base + kCodeBytes + tid * 16U) = sstg;
    };
    float acc[kTw][kTm][8];
#pragma unroll
    for (unsigned i = 0; i < kTw; ++i)
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[i][j][e] = 0.0F;
    fetch(0U);
    put(0U);
    __syncthreads();
    for (unsigned b = 0; b < blocks; ++b) {
        const bool more = b + 1U < blocks;
        if (more) fetch(b + 1U);
        if (live) block_dot(lds + (b & 1U) * kBlockBytes, wn, wm, l16, hi, acc);
        if (more) put((b + 1U) & 1U);
        __syncthreads();
    }
    if (!live) return;  // after the last barrier
    // Epilogue through LDS (the operand buffers are dead after the last barrier). The WMMA layout gives a lane 2 adjacent
    // columns of 16 rows, so direct stores cost 16 lines of 8 B per instruction (measured: ~6.6 ms fixed per 8K launch).
    // Each wave stages its values in a private 32-row x 32-column region (row stride kStageStride: conflict-free writes),
    // then stores whole 16 B chunks, 8 lanes per row (128 B contiguous). The stored values and their rounding are the
    // direct form's, bit for bit; only the store order changes. Plain: 2 phases of 2 W tiles; paired: 1 phase of 2 pairs.
    constexpr unsigned kPhases = Epi::kPaired ? 1U : kTw / 2U;
    float* const st = reinterpret_cast<float*>(lds) + wave * (kTm * 16U * kStageStride);
    const unsigned c0 = Epi::kPaired ? nt0 * 8U : nt0 * 16U;  // first output column of this wave
    const bool vec = ((reinterpret_cast<std::uintptr_t>(g.y) | (g.y_row * sizeof(float))) & 15U) == 0U;  // uniform
    float bm[kTm];
#pragma unroll
    for (unsigned j = 0; j < kTm; ++j) bm[j] = a.base[image_row((mt0 + j) * 16U + l16)];  // clamped row: always readable
#pragma unroll
    for (unsigned p = 0; p < kPhases; ++p) {
#pragma unroll
        for (unsigned j = 0; j < kTm; ++j)
#pragma unroll
            for (unsigned k = 0; k < 2U; ++k)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    float v;
                    if constexpr (!Epi::kPaired) {
                        v = acc[2U * p + k][j][e] * bm[j];
                        asm volatile("" : "+v"(v));  // y = acc * base rounded once, as the dense launch
                    } else {
                        float gate = acc[2U * k][j][e] * bm[j], up = acc[2U * k + 1U][j][e] * bm[j];
                        asm volatile("" : "+v"(gate), "+v"(up));
                        v = epi(gate, up);
                    }
                    st[(j * 16U + l16) * kStageStride + k * 16U + 2U * e + hi] = v;
                }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
        for (unsigned it = 0; it < kTm * 16U / 4U; ++it) {
            const unsigned row = it * 4U + lane / 8U, ch = lane % 8U, r = mt0 * 16U + row;
            // Column tile of the chunk: plain W tile nt0 + 2 p + ch / 4; paired pair (nt0 + 2 (ch / 4), + 1).
            const bool col_live = Epi::kPaired ? nt0 + 2U * (ch / 4U) + 1U < ntiles : nt0 + 2U * p + ch / 4U < ntiles;
            if (r >= rows || !col_live) continue;
            const F4 v = *reinterpret_cast<const F4*>(st + row * kStageStride + ch * 4U);
            float* yp = g.y + std::size_t(r) * g.y_row + c0 + p * 32U + ch * 4U;
            if (vec) {
                *reinterpret_cast<F4*>(yp) = v;
            } else {
#pragma unroll
                for (unsigned q = 0; q < 4U; ++q) yp[q] = v[q];
            }
        }
        if (p + 1U < kPhases) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");  // the next phase overwrites the region
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        }
    }
}
}  // namespace detail

// Host side (stream-ordered; callers check supported() and hipGetLastError()).
// x: rows of a.in values at x_stride floats. quantise_map2: the quantised value is F::apply(a[k], b[k]) (same stride);
// with a second input it needs map2_supported(in).
inline bool map2_supported(unsigned in) { return in <= detail::kCtaInputs; }
template <class F = detail::Ident, bool Rows = false>
inline void enqueue_quantise_map2(const float* x0, const float* x1, unsigned x_stride, const Act& a, hipStream_t s) {
    if (a.in <= detail::kCtaInputs) STRIX_HIP_LAUNCH_GGL(Rdna3Layout, (detail::quantise_cta_kernel<F, Rows>), dim3(padded_rows(a.rows)), dim3(256), 0, s, x0, x1, x_stride, a);
    else STRIX_HIP_LAUNCH_GGL(Rdna3Layout, detail::quantise_kernel<Rows>, dim3((padded_rows(a.rows) + 7U) / 8U), dim3(256), 0, s, x0, x_stride, a);
}
inline void enqueue_quantise(const float* x, unsigned x_stride, const Act& a, hipStream_t s) {
    enqueue_quantise_map2<detail::Ident>(x, nullptr, x_stride, a, s);
}
// The row-major image of the grouped launch (same values as enqueue_quantise; byte sizes act_*_bytes).
inline void enqueue_quantise_rows(const float* x, unsigned x_stride, const Act& a, hipStream_t s) {
    enqueue_quantise_map2<detail::Ident, true>(x, nullptr, x_stride, a, s);
}
// Same image, verdict-gated (nothing stored when *verdict != 0; the word must not change during the launch). Needs
// map2_supported(a.in).
inline void enqueue_quantise_rows_gated(const unsigned* verdict, const float* x, unsigned x_stride, const Act& a, hipStream_t s) {
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, (detail::quantise_cta_gated_kernel<detail::Ident, true>), dim3(padded_rows(a.rows)), dim3(256), 0, s,
                         verdict, x, nullptr, x_stride, a);
}
// Requantisation from a weight source (Src::requantise(col, b, n, dst) -> step, see detail::requantise_kernel).
template <class Src>
inline void enqueue_requantise_source(const Src& src, unsigned in, unsigned n, std::uint8_t* dst, float* step, hipStream_t s) {
    const std::size_t threads = std::size_t(n) * (in / rot::kBlock);
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, (detail::requantise_kernel<Src>), dim3(static_cast<unsigned>((threads + 255U) / 256U)), dim3(256), 0, s, src, in, n,
                         dst, step);
}
// Grouped launch (GroupCta) over a ROW-MAJOR activation image (enqueue_quantise_rows). Limits: supported(a.in, n), a
// paired epilogue n % 32 == 0, 0 < tiles * grouped_slices(max_tile_rows) <= 65535 (grid y); every image row a map names
// is < a.rows.
inline unsigned grouped_slices(unsigned max_tile_rows) {
    const unsigned s = (max_tile_rows + kCtaTilesM * 16U - 1U) / (kCtaTilesM * 16U);
    return s > 0U ? s : 1U;
}
inline bool grouped_supported(unsigned in, unsigned n, bool paired, unsigned tiles, unsigned max_tile_rows) {
    return supported(in, n) && (!paired || n % 32U == 0U) && tiles > 0U && std::size_t(tiles) * grouped_slices(max_tile_rows) <= 65535U;
}
// max_tile_rows: the plan's tallest tile (grid z = row slices of kCtaTilesM * 16 rows; a slice past a tile leaves).
template <class Src, class Epi = PlainOut>
inline void enqueue_grouped_gemm(const Src& src, unsigned tiles, unsigned max_tile_rows, unsigned n, const Act& a, hipStream_t s, Epi epi = {}) {
    const unsigned slices = grouped_slices(max_tile_rows);
    const dim3 grid((n / 16U + kCtaTilesN - 1U) / kCtaTilesN, tiles * slices);
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, (detail::grouped_gemm_kernel<Src, Epi>), grid, dim3(32U * kWavesN * kWavesM), 0, s, src, a, n, slices, epi);
}
inline void enqueue_requantise(const std::uint8_t* pack, const std::uint16_t* scales, unsigned in, unsigned n, std::uint8_t* dst, float* step,
                               hipStream_t s, rot::WeightRule rule = rot::WeightRule::amax) {
    enqueue_requantise_source(detail::G64Source{pack, scales, in, rule}, in, n, dst, step, s);
}
template <class Cols, class Store = f16_dense_tall::RawStore>
inline void enqueue_gemm(const Weights& w, const Act& a, Cols cols, hipStream_t s, Store store = {}) {
    const unsigned mtiles = padded_rows(a.rows) / 16U, ntiles = w.n / 16U;
    const dim3 grid((ntiles + kCtaTilesN - 1U) / kCtaTilesN, (mtiles + kCtaTilesM - 1U) / kCtaTilesM);
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, (detail::gemm_kernel<Cols, Store>), grid, dim3(32U * kWavesN * kWavesM), 0, s, w, a, cols, store);
}
}  // namespace strix::core::iu4a4_dense
