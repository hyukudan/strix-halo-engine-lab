#pragma once
// TURBO arithmetic gsq-iq-i8mma-turbo-cuda-v1 (contract and oracle geometry: turbo_i8.hpp). GSQ experts on int8 tensor
// cores: int8 activation codes (the shared A8 quantizers) x int8 weight codes, one exact int32 IMMA per 32-value group and
// one FP32 FMA per group and output. Selected only by the TURBO profile or the explicit expert arithmetic
// (qwen_moe.cu enqueue_routed); the exact, FAST and A8 kernels of moe_a8.cuh are not touched.
// Included after moe_a8.cuh (at(), cp16 and the cp.async group helpers). Binding: hidden, ffn, projection_nonfinite, Rows,
// Tables and activation_tile, as for the A8 kernels.
//
// Work mapping: one block = one CSR tile (<= 64 assignments of one expert = 4 m16 blocks), 8 warps.
//  - A warp owns 8 weight rows per matrix (project: gate and up) or 16 (down: two n8 blocks) and decodes exactly those rows'
//    integer codes straight into B registers once per tile; the 4 m16 blocks reuse every decoded fragment, so no code is
//    decoded twice per tile and there is no shared-memory weight stage.
//  - int8 activation codes and their per-group deltas are staged by cp.async, 128 values per stage, double buffered, one
//    barrier per stage, and read by all 8 warps.
//  - The IMMA accumulator input is the constant magic (turbo_i8::kMagicBits): the group dot becomes FP32 with one FADD.
//  - project ends in the shared A8 SwiGLU + down quantizer (Binding::activation_tile), so h codes are produced by exactly
//    the code the A8 class uses; down reads those int8 codes and writes rawdots with the native layout (merge / commit
//    unchanged).
#include "turbo_i8.hpp"

namespace strix::cuda::core::moe_turbo {
namespace ti = turbo_i8;
using moe_a8::at;
using moe_a8::cp16;
using moe_a8::cp_commit;
using moe_a8::cp_wait_all;

constexpr unsigned kTurboThreads = ti::kThreads;
constexpr unsigned kTurboProjectOutputs = ti::kProjectOutputs;  // grid (tiles, ffn / 64)
constexpr unsigned kTurboDownOutputs = ti::kDownOutputs;        // grid (tiles, hidden / 128)
constexpr unsigned kStageA = ti::kTileRows * ti::kPitch;        // int8 codes of one stage
constexpr unsigned kStageBytes = kStageA + ti::kTileRows * 16U; // + 4 FP32 deltas per row
static_assert(ti::kStageK == 128U, "4 groups (16 B of deltas) per staged row");

// D = A(16x32 s8) * B(32x8 s8) + C, C = (c, c, c, c).
__device__ __forceinline__ void imma_k32(unsigned (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1, unsigned c) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}
// D = A(16x16 s8) * B(16x8 s8) + C, C = (c, c, c, c).
__device__ __forceinline__ void imma_k16(unsigned (&d)[4], unsigned a0, unsigned a1, unsigned b0, unsigned c) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(c));
}

// B words of one weight row (column g of this lane) for one group, and the scales of this lane's two C columns.
template <class Decode>
__device__ __forceinline__ void turbo_weights(const std::uint8_t* row, unsigned group, unsigned lane, unsigned (&b)[2],
                                              float (&scale)[2][2]) {
#pragma unroll
    for (unsigned r = 0; r < 2U; ++r)
        b[r] = Decode::word(row, group, Decode::halves ? ti::k16_word(lane, r) : ti::k32_word(lane, r));
    float first = 0.0F, second = 0.0F;
    Decode::scales(row, group, first, second);
    scale[0][0] = __shfl_sync(0xFFFFFFFFU, first, ti::scale_lane(lane, 0U));
    scale[0][1] = __shfl_sync(0xFFFFFFFFU, first, ti::scale_lane(lane, 1U));
    if constexpr (Decode::halves) {
        scale[1][0] = __shfl_sync(0xFFFFFFFFU, second, ti::scale_lane(lane, 0U));
        scale[1][1] = __shfl_sync(0xFFFFFFFFU, second, ti::scale_lane(lane, 1U));
    } else {
        scale[1][0] = scale[0][0]; scale[1][1] = scale[0][1];
    }
}

// One group of one m16 block against one decoded weight fragment: IMMA(s) + the contract's FP32 step per C element.
// A: staged codes (pitch ti::kPitch), rows 16 m + g and + 8, group j of the stage; e: deltas of this lane's two rows.
template <bool Halves>
__device__ __forceinline__ void turbo_group(float (&acc)[4], const unsigned char* A, unsigned row0, unsigned j, unsigned lane,
                                            const unsigned (&b)[2], const float (&scale)[2][2], const float (&e)[2]) {
    unsigned d[4];
    if constexpr (Halves) {
#pragma unroll
        for (unsigned h = 0; h < 2U; ++h) {
            const unsigned k = 32U * j + ti::k16_a_byte(lane, h);
            const unsigned a0 = *reinterpret_cast<const unsigned*>(A + row0 * ti::kPitch + k);
            const unsigned a1 = *reinterpret_cast<const unsigned*>(A + (row0 + 8U) * ti::kPitch + k);
            imma_k16(d, a0, a1, b[h], ti::kMagicBits);
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) acc[i] = ti::step(acc[i], ti::dot_float(d[i]), e[i >> 1U], scale[h][i & 1U]);
        }
    } else {
        const unsigned k = 32U * j + ti::k32_a_byte(lane);
        const uint2 r0 = *reinterpret_cast<const uint2*>(A + row0 * ti::kPitch + k);
        const uint2 r1 = *reinterpret_cast<const uint2*>(A + (row0 + 8U) * ti::kPitch + k);
        unsigned a[4];
        ti::a_regs(r0.x, r0.y, r1.x, r1.y, a);
        imma_k32(d, a, b[0], b[1], ti::kMagicBits);
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) acc[i] = ti::step(acc[i], ti::dot_float(d[i]), e[i >> 1U], scale[0][i & 1U]);
    }
}

// Stages `count` rows of 128 codes + 4 deltas (rows >= count are not written: an MMA row reaches only its own C row, and
// those C rows are never stored).
__device__ __forceinline__ void turbo_stage(unsigned char* st, unsigned count, unsigned t, const std::int8_t* const* code_rows,
                                            const float* const* delta_rows, unsigned s) {
    for (unsigned i = t; i < count * 8U; i += kTurboThreads) {
        const unsigned r = i >> 3, c = i & 7U;
        cp16(st + r * ti::kPitch + 16U * c, code_rows[r] + s * ti::kStageK + 16U * c);
    }
    for (unsigned r = t; r < count; r += kTurboThreads) cp16(st + kStageA + 16U * r, delta_rows[r] + 4U * s);
    cp_commit();
}

// grid (tiles, ffn / 64), 256 threads. Gate and up of one output are the same C element of two accumulators; the FP32
// g / u tile then goes through Binding::activation_tile (the A8 SwiGLU + per-32 down quantizer, fail-closed flags).
template <unsigned T, class Binding>
__global__ __launch_bounds__(kTurboThreads, 2) void turbo_project_kernel(const std::uint8_t* gate, const std::uint8_t* up,
                                                                         typename Binding::Rows v, typename Binding::Tables tables) {
    using Decode = weight::GsqDecode<T>;
    constexpr unsigned hidden = Binding::hidden, ffn = Binding::ffn, projection_nonfinite = Binding::projection_nonfinite;
    constexpr unsigned stages = hidden / ti::kStageK, groups_per_stage = ti::kStageK / 32U;
    constexpr std::uint64_t row_bytes = std::uint64_t(hidden / Decode::be) * Decode::bb;
    static_assert(hidden % ti::kStageK == 0U && hidden % Decode::be == 0U && ffn % kTurboProjectOutputs == 0U,
                  "TURBO project covers whole stages, storage blocks and output blocks");
    constexpr unsigned kTileBytes = 2U * ti::kTileRows * kTurboProjectOutputs * 4U;  // g / u FP32 tile after the K loop
    constexpr unsigned kSmem = kTileBytes > 2U * kStageBytes ? kTileBytes : 2U * kStageBytes;
    const auto& csr = v.csr;
    if ((csr.summary[0] & ~projection_nonfinite) || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], count = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[blockIdx.x];
    __shared__ alignas(16) unsigned char smem[kSmem];
    __shared__ unsigned row_of[ti::kTileRows], slot_of[ti::kTileRows];
    __shared__ const std::int8_t* code_rows[ti::kTileRows];
    __shared__ const float* delta_rows[ti::kTileRows];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, g = lane >> 2;
    const unsigned out_base = ti::project_output(blockIdx.y, warp, 0U);
    if (t < ti::kTileRows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[tile_begin + a]; slot_of[t] = csr.slots[tile_begin + a];
        code_rows[t] = at(v.codes, row_of[t]);
        delta_rows[t] = at(v.deltas, row_of[t]);
    }
    __syncthreads();
    const std::uint64_t wr = std::uint64_t(expert) * ffn + out_base + g;
    const std::uint8_t* wrow[2] = {gate + wr * row_bytes, up + wr * row_bytes};
    const unsigned mblocks = ti::m_blocks(count);  // block-uniform
    float acc[4][2][4];
#pragma unroll
    for (unsigned m = 0; m < 4U; ++m)
#pragma unroll
        for (unsigned mat = 0; mat < 2U; ++mat)
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) acc[m][mat][i] = 0.0F;
    turbo_stage(smem, count, t, code_rows, delta_rows, 0U);
#pragma unroll 1
    for (unsigned s = 0; s < stages; ++s) {
        cp_wait_all();
        __syncthreads();  // stage s landed; every warp finished stage s - 1, so its buffer is free
        if (s + 1U < stages) turbo_stage(smem + ((s + 1U) & 1U) * kStageBytes, count, t, code_rows, delta_rows, s + 1U);
        const unsigned char* A = smem + (s & 1U) * kStageBytes;
        const float* E = reinterpret_cast<const float*>(A + kStageA);  // [row][4]
#pragma unroll
        for (unsigned j = 0; j < groups_per_stage; ++j) {
            const unsigned group = s * groups_per_stage + j;
            unsigned b[2][2];
            float scale[2][2][2];
            turbo_weights<Decode>(wrow[0], group, lane, b[0], scale[0]);
            turbo_weights<Decode>(wrow[1], group, lane, b[1], scale[1]);
#pragma unroll
            for (unsigned m = 0; m < 4U; ++m) {
                if (m >= mblocks) break;
                const unsigned row0 = 16U * m + g;
                const float e[2] = {E[4U * row0 + j], E[4U * (row0 + 8U) + j]};
#pragma unroll
                for (unsigned mat = 0; mat < 2U; ++mat)
                    turbo_group<Decode::halves>(acc[m][mat], A, row0, j, lane, b[mat], scale[mat], e);
            }
        }
    }
    __syncthreads();  // the stage buffers become the g / u tile
    float* Gt = reinterpret_cast<float*>(smem);  // [64 rows][64 outputs]
    float* Ut = Gt + ti::kTileRows * kTurboProjectOutputs;
#pragma unroll
    for (unsigned m = 0; m < 4U; ++m)
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) {
            const unsigned row = 16U * m + ti::c_row(lane, i), o = 8U * warp + ti::c_col(lane, i);
            if (row < count) { Gt[row * kTurboProjectOutputs + o] = acc[m][0][i]; Ut[row * kTurboProjectOutputs + o] = acc[m][1][i]; }
        }
    __syncthreads();
    Binding::template activation_tile<kTurboProjectOutputs, kTurboThreads>(Gt, Ut, v, tables, row_of, slot_of, count,
                                                                         blockIdx.y * kTurboProjectOutputs);
}

// grid (tiles, hidden / 128), 256 threads. Reads the A8 down codes / deltas of each (row, slot) and writes FP32
// rawdots[row][slot * hidden + o] (the native route's storage; merge / commit unchanged).
template <unsigned T, class Binding>
__global__ __launch_bounds__(kTurboThreads, 2) void turbo_down_kernel(const std::uint8_t* down, typename Binding::Rows v) {
    using Decode = weight::GsqDecode<T>;
    constexpr unsigned hidden = Binding::hidden, ffn = Binding::ffn;
    constexpr unsigned stages = ffn / ti::kStageK, groups_per_stage = ti::kStageK / 32U;
    constexpr std::uint64_t row_bytes = std::uint64_t(ffn / Decode::be) * Decode::bb;
    static_assert(ffn % ti::kStageK == 0U && ffn % Decode::be == 0U && hidden % kTurboDownOutputs == 0U,
                  "TURBO down covers whole stages, storage blocks and output blocks");
    const auto& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], count = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[blockIdx.x];
    __shared__ alignas(16) unsigned char smem[2U * kStageBytes];
    __shared__ unsigned row_of[ti::kTileRows], slot_of[ti::kTileRows];
    __shared__ const std::int8_t* code_rows[ti::kTileRows];
    __shared__ const float* delta_rows[ti::kTileRows];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, g = lane >> 2;
    const unsigned out_base = ti::down_output(blockIdx.y, warp, 0U, 0U);
    if (t < ti::kTileRows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[tile_begin + a]; slot_of[t] = csr.slots[tile_begin + a];
        code_rows[t] = at(v.down_codes, row_of[t]) + slot_of[t] * ffn;
        delta_rows[t] = at(v.down_deltas, row_of[t]) + slot_of[t] * (ffn / 32U);
    }
    __syncthreads();
    const std::uint8_t* wrow[2];
#pragma unroll
    for (unsigned n = 0; n < 2U; ++n) wrow[n] = down + (std::uint64_t(expert) * hidden + out_base + 8U * n + g) * row_bytes;
    const unsigned mblocks = ti::m_blocks(count);  // block-uniform
    float acc[4][2][4];
#pragma unroll
    for (unsigned m = 0; m < 4U; ++m)
#pragma unroll
        for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) acc[m][n][i] = 0.0F;
    turbo_stage(smem, count, t, code_rows, delta_rows, 0U);
#pragma unroll 1
    for (unsigned s = 0; s < stages; ++s) {
        cp_wait_all();
        __syncthreads();
        if (s + 1U < stages) turbo_stage(smem + ((s + 1U) & 1U) * kStageBytes, count, t, code_rows, delta_rows, s + 1U);
        const unsigned char* A = smem + (s & 1U) * kStageBytes;
        const float* E = reinterpret_cast<const float*>(A + kStageA);
#pragma unroll
        for (unsigned j = 0; j < groups_per_stage; ++j) {
            const unsigned group = s * groups_per_stage + j;
            unsigned b[2][2];
            float scale[2][2][2];
            turbo_weights<Decode>(wrow[0], group, lane, b[0], scale[0]);
            turbo_weights<Decode>(wrow[1], group, lane, b[1], scale[1]);
#pragma unroll
            for (unsigned m = 0; m < 4U; ++m) {
                if (m >= mblocks) break;
                const unsigned row0 = 16U * m + g;
                const float e[2] = {E[4U * row0 + j], E[4U * (row0 + 8U) + j]};
#pragma unroll
                for (unsigned n = 0; n < 2U; ++n) turbo_group<Decode::halves>(acc[m][n], A, row0, j, lane, b[n], scale[n], e);
            }
        }
    }
#pragma unroll
    for (unsigned m = 0; m < 4U; ++m)
#pragma unroll
        for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
            for (unsigned i = 0; i < 4U; i += 2U) {
                const unsigned row = 16U * m + ti::c_row(lane, i);
                if (row < count) {
                    float* dst = at(v.rawdots, row_of[row]) + slot_of[row] * hidden + out_base + 8U * n + ti::c_col(lane, i);
                    *reinterpret_cast<float2*>(dst) = make_float2(acc[m][n][i], acc[m][n][i + 1U]);
                }
            }
}

}  // namespace strix::cuda::core::moe_turbo
