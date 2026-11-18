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
//  - Raster: tile-major (native_mma::tile_major): consecutive blocks are the output slices of ONE tile, so its staged
//    activation rows are read from DRAM once and the expert's weights stay in L2 for the next tile of the same expert
//    (tile-fastest re-streamed every tile's activations once per slice). Each block's outputs are a pure function of its
//    (tile, slice), so the raster changes the order of blocks only: bytes equal.
//  - Limbs = 1: gsq-iq-i8mma-turbo-cuda-v1. Limbs = 2: gsq-iq-i8mma-turbo-cuda-v2 (turbo_i8.hpp): a second int8 activation
//    limb (Rows::lo_codes / down_lo_codes) staged beside the A8 codes and multiplied by the SAME decoded B fragment
//    (one more IMMA, two more FMA per group and output; decode unchanged). The input limb comes from turbo_input_lo_kernel,
//    the down limb from the project epilogue (from the block's own FP32 h, A8 code and delta).
#include "native_mma.hpp"  // tile_major: the shared tile-major block raster of the expert kernels
#include "turbo_i8.hpp"
#include <type_traits>
// Weights: the shared word-level GSQ decoder (weight/gsq_decode.cuh, bit-identical to the one-shot word/scales), fields of
// group g + 1 loaded before group g's IMMAs, codebook copied to shared memory per block when it fits (turbo_grid).

namespace strix::cuda::core::moe_turbo {
namespace ti = turbo_i8;
using moe_a8::at;
using moe_a8::cp16;
using moe_a8::cp_commit;
using moe_a8::cp_wait_all;

constexpr unsigned kTurboThreads = ti::kThreads;
constexpr unsigned kTurboProjectOutputs = ti::kProjectOutputs;  // grid (tiles, ffn / 64)
constexpr unsigned kTurboDownOutputs = ti::kDownOutputs;        // grid (tiles, hidden / 128)
constexpr unsigned kStageA = ti::kTileRows * ti::kPitch;        // int8 codes of one stage (one limb)
template <unsigned Limbs> inline constexpr unsigned kStageBytesOf = Limbs * kStageA + ti::kTileRows * 16U;  // + 4 deltas/row
static_assert(2U * kStageBytesOf<2> + 2U * 1024U <= 48U * 1024U, "two v2 stages + row tables fit static shared memory");
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

// The two code words a lane owns per group (B reg r): k32 placement 2 q + r, two-scale halves q + 4 r.
template <class Decode> __device__ __forceinline__ unsigned turbo_word_index(unsigned lane, unsigned r) {
    return Decode::halves ? ti::k16_word(lane, r) : ti::k32_word(lane, r);
}
// Field bytes of one weight row (column g of this lane) for one group: loads only (shared GsqDecode::load), so the loops
// issue them one group ahead of the decode and IMMAs.
template <class Decode>
__device__ __forceinline__ typename Decode::Raw turbo_raw(const std::uint8_t* row, unsigned group, unsigned lane) {
    return Decode::load(row, group, turbo_word_index<Decode>(lane, 0U), turbo_word_index<Decode>(lane, 1U));
}
// Codebook view of a TURBO kernel: the block's shared-memory copy when it fits beside the kernel's static shared memory
// (48 KB per block keeps two blocks per SM), else the constant tables. Types without a codebook (IQ4_NL/IQ4_XS/Q2_0)
// decode from registers and never read it. The view changes where a codebook word is read, never its value.
template <unsigned T, unsigned KernelSmem>
inline constexpr bool kTurboSmemGrid =
    weight::GsqDecode<T>::grid_words != 0U && KernelSmem + 4U * weight::GsqDecode<T>::grid_words <= 48U * 1024U;
template <unsigned T, unsigned KernelSmem>
using TurboGrid = std::conditional_t<kTurboSmemGrid<T, KernelSmem>, weight::GsqSmemGrid, weight::GsqConstGrid<T>>;
template <unsigned T, unsigned KernelSmem>
__device__ __forceinline__ TurboGrid<T, KernelSmem> turbo_grid(const std::uint32_t* words) {
    if constexpr (kTurboSmemGrid<T, KernelSmem>) return weight::GsqSmemGrid{words};
    else return weight::GsqConstGrid<T>{};
}
// Row-table bytes of both TURBO kernels (row_of, slot_of, code_rows, lo_rows, delta_rows) for the smem budget above.
inline constexpr unsigned kTurboRowTables = ti::kTileRows * (2U * 4U + 3U * 8U);

// B words of one weight row (column g of this lane) for one group from its loaded fields (shared word-level decode, bit
// for bit the one-shot GsqDecode::word / scales), and the scales of this lane's two C columns.
template <class Decode, class Grid>
__device__ __forceinline__ void turbo_weights(const typename Decode::Raw& raw, unsigned group, const Grid& grid,
                                              unsigned lane, unsigned (&b)[2], float (&scale)[2][2]) {
#pragma unroll
    for (unsigned r = 0; r < 2U; ++r) b[r] = Decode::word(raw, r, turbo_word_index<Decode>(lane, r), grid);
    float first = 0.0F, second = 0.0F;
    Decode::scales(raw, group, first, second);
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
// Limbs = 2: the lo limb plane follows the A8 plane at A + kStageA (same pitch and placement).
template <bool Halves, unsigned Limbs>
__device__ __forceinline__ void turbo_group(float (&acc)[4], const unsigned char* A, unsigned row0, unsigned j, unsigned lane,
                                            const unsigned (&b)[2], const float (&scale)[2][2], const float (&e)[2]) {
    static_assert(Limbs == 1U || Limbs == 2U, "one or two activation limbs");
    unsigned d[4];
    if constexpr (Halves) {
#pragma unroll
        for (unsigned h = 0; h < 2U; ++h) {
            const unsigned k = 32U * j + ti::k16_a_byte(lane, h);
            const unsigned a0 = *reinterpret_cast<const unsigned*>(A + row0 * ti::kPitch + k);
            const unsigned a1 = *reinterpret_cast<const unsigned*>(A + (row0 + 8U) * ti::kPitch + k);
            imma_k16(d, a0, a1, b[h], ti::kMagicBits);
            if constexpr (Limbs == 1U) {
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i) acc[i] = ti::step(acc[i], ti::dot_float(d[i]), e[i >> 1U], scale[h][i & 1U]);
            } else {
                const unsigned char* L = A + kStageA;
                unsigned dl[4];
                imma_k16(dl, *reinterpret_cast<const unsigned*>(L + row0 * ti::kPitch + k),
                         *reinterpret_cast<const unsigned*>(L + (row0 + 8U) * ti::kPitch + k), b[h], ti::kMagicBits);
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i)
                    acc[i] = ti::step2(acc[i], ti::dot_float(d[i]), ti::lo_dot_float(dl[i]), e[i >> 1U], scale[h][i & 1U]);
            }
        }
    } else {
        const unsigned k = 32U * j + ti::k32_a_byte(lane);
        const uint2 r0 = *reinterpret_cast<const uint2*>(A + row0 * ti::kPitch + k);
        const uint2 r1 = *reinterpret_cast<const uint2*>(A + (row0 + 8U) * ti::kPitch + k);
        unsigned a[4];
        ti::a_regs(r0.x, r0.y, r1.x, r1.y, a);
        imma_k32(d, a, b[0], b[1], ti::kMagicBits);
        if constexpr (Limbs == 1U) {
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) acc[i] = ti::step(acc[i], ti::dot_float(d[i]), e[i >> 1U], scale[0][i & 1U]);
        } else {
            const unsigned char* L = A + kStageA;
            const uint2 l0 = *reinterpret_cast<const uint2*>(L + row0 * ti::kPitch + k);
            const uint2 l1 = *reinterpret_cast<const uint2*>(L + (row0 + 8U) * ti::kPitch + k);
            unsigned al[4], dl[4];
            ti::a_regs(l0.x, l0.y, l1.x, l1.y, al);
            imma_k32(dl, al, b[0], b[1], ti::kMagicBits);
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i)
                acc[i] = ti::step2(acc[i], ti::dot_float(d[i]), ti::lo_dot_float(dl[i]), e[i >> 1U], scale[0][i & 1U]);
        }
    }
}

// Stages `count` rows of 128 codes + 4 deltas (rows >= count are not written: an MMA row reaches only its own C row, and
// those C rows are never stored).
// Limbs = 2 also stages the lo limb rows into the second plane.
template <unsigned Limbs>
__device__ __forceinline__ void turbo_stage(unsigned char* st, unsigned count, unsigned t, const std::int8_t* const* code_rows,
                                            const std::int8_t* const* lo_rows, const float* const* delta_rows, unsigned s) {
#pragma unroll
    for (unsigned limb = 0; limb < Limbs; ++limb) {
        const std::int8_t* const* rows = limb == 0U ? code_rows : lo_rows;
        for (unsigned i = t; i < count * 8U; i += kTurboThreads) {
            const unsigned r = i >> 3, c = i & 7U;
            cp16(st + limb * kStageA + r * ti::kPitch + 16U * c, rows[r] + s * ti::kStageK + 16U * c);
        }
    }
    for (unsigned r = t; r < count; r += kTurboThreads) cp16(st + Limbs * kStageA + 16U * r, delta_rows[r] + 4U * s);
    cp_commit();
}

// grid (tiles, ffn / 64), tile-major raster, 256 threads. Gate and up of one output are the same C element of two
// accumulators; the FP32 g / u tile then goes through Binding::activation_tile (the A8 SwiGLU + per-32 down quantizer,
// fail-closed flags). Limbs = 2 then writes the down lo limb of the block's outputs (needs Rows::intermediates for h).
template <unsigned T, unsigned Limbs, class Binding>
__global__ __launch_bounds__(kTurboThreads, 2) void turbo_project_kernel(const std::uint8_t* gate, const std::uint8_t* up,
                                                                         typename Binding::Rows v, typename Binding::Tables tables) {
    using Decode = weight::GsqDecode<T>;
    constexpr unsigned hidden = Binding::hidden, ffn = Binding::ffn, projection_nonfinite = Binding::projection_nonfinite;
    constexpr unsigned stages = hidden / ti::kStageK, groups_per_stage = ti::kStageK / 32U;
    constexpr std::uint64_t row_bytes = std::uint64_t(hidden / Decode::be) * Decode::bb;
    static_assert(hidden % ti::kStageK == 0U && hidden % Decode::be == 0U && ffn % kTurboProjectOutputs == 0U,
                  "TURBO project covers whole stages, storage blocks and output blocks");
    constexpr unsigned kTileBytes = 2U * ti::kTileRows * kTurboProjectOutputs * 4U;  // g / u FP32 tile after the K loop
    constexpr unsigned kStage = kStageBytesOf<Limbs>;
    constexpr unsigned kSmem = kTileBytes > 2U * kStage ? kTileBytes : 2U * kStage;
    const auto& csr = v.csr;
    const native_mma::TileSlice ts = native_mma::tile_major(blockIdx.x, blockIdx.y, gridDim.x, ffn / kTurboProjectOutputs);
    if ((csr.summary[0] & ~projection_nonfinite) || ts.tile >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[ts.tile], count = csr.tile_begins[ts.tile + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[ts.tile];
    __shared__ alignas(16) unsigned char smem[kSmem];
    __shared__ unsigned row_of[ti::kTileRows], slot_of[ti::kTileRows];
    __shared__ const std::int8_t* code_rows[ti::kTileRows];
    __shared__ const std::int8_t* lo_rows[ti::kTileRows];
    __shared__ const float* delta_rows[ti::kTileRows];
    constexpr unsigned kOwnSmem = kSmem + kTurboRowTables;
    __shared__ std::uint32_t grid_words[kTurboSmemGrid<T, kOwnSmem> ? Decode::grid_words : 1U];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, g = lane >> 2;
    const unsigned out_base = ti::project_output(ts.slice, warp, 0U);
    if (t < ti::kTileRows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[tile_begin + a]; slot_of[t] = csr.slots[tile_begin + a];
        code_rows[t] = at(v.codes, row_of[t]);
        lo_rows[t] = Limbs == 2U ? at(v.lo_codes, row_of[t]) : nullptr;
        delta_rows[t] = at(v.deltas, row_of[t]);
    }
    if constexpr (kTurboSmemGrid<T, kOwnSmem>) weight::gsq_grid_fill<T>(grid_words, t, kTurboThreads);
    __syncthreads();
    const auto grid = turbo_grid<T, kOwnSmem>(grid_words);
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
    turbo_stage<Limbs>(smem, count, t, code_rows, lo_rows, delta_rows, 0U);
    // Weight fields of the next group are loaded before this group's decode and IMMAs (and across the stage barrier): the
    // global round trip overlaps one group of work. The last group re-loads itself (no branch, no read past the row).
    constexpr unsigned groups = stages * groups_per_stage;
    typename Decode::Raw ahead[2] = {turbo_raw<Decode>(wrow[0], 0U, lane), turbo_raw<Decode>(wrow[1], 0U, lane)};
#pragma unroll 1
    for (unsigned s = 0; s < stages; ++s) {
        cp_wait_all();
        __syncthreads();  // stage s landed; every warp finished stage s - 1, so its buffer is free
        if (s + 1U < stages) turbo_stage<Limbs>(smem + ((s + 1U) & 1U) * kStage, count, t, code_rows, lo_rows, delta_rows, s + 1U);
        const unsigned char* A = smem + (s & 1U) * kStage;
        const float* E = reinterpret_cast<const float*>(A + Limbs * kStageA);  // [row][4]
#pragma unroll
        for (unsigned j = 0; j < groups_per_stage; ++j) {
            const unsigned group = s * groups_per_stage + j, next = group + 1U < groups ? group + 1U : group;
            const typename Decode::Raw raw[2] = {ahead[0], ahead[1]};
            ahead[0] = turbo_raw<Decode>(wrow[0], next, lane);
            ahead[1] = turbo_raw<Decode>(wrow[1], next, lane);
            unsigned b[2][2];
            float scale[2][2][2];
            turbo_weights<Decode>(raw[0], group, grid, lane, b[0], scale[0]);
            turbo_weights<Decode>(raw[1], group, grid, lane, b[1], scale[1]);
#pragma unroll
            for (unsigned m = 0; m < 4U; ++m) {
                if (m >= mblocks) break;
                const unsigned row0 = 16U * m + g;
                const float e[2] = {E[4U * row0 + j], E[4U * (row0 + 8U) + j]};
#pragma unroll
                for (unsigned mat = 0; mat < 2U; ++mat)
                    turbo_group<Decode::halves, Limbs>(acc[m][mat], A, row0, j, lane, b[mat], scale[mat], e);
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
                                                                         ts.slice * kTurboProjectOutputs);
    if constexpr (Limbs == 2U) {
        // Down lo limb of this block's (row, slot, 64 features): from the FP32 h, A8 code and delta that activation_tile
        // just wrote (global writes of this block are visible to it after the barrier). A non-finite group is flagged by
        // activation_tile and the down kernel returns on the summary, so its limb is never read.
        __syncthreads();
        const unsigned out0 = ts.slice * kTurboProjectOutputs;
        for (unsigned i = t; i < count * kTurboProjectOutputs; i += kTurboThreads) {
            const unsigned a = i / kTurboProjectOutputs, feature = out0 + i % kTurboProjectOutputs, r = row_of[a], slot = slot_of[a];
            const float h = at(v.intermediates, r)[slot * 2U * ffn + feature];
            const float e = at(v.down_deltas, r)[slot * (ffn / 32U) + feature / 32U];
            const int c1 = at(v.down_codes, r)[slot * ffn + feature];
            at(v.down_lo_codes, r)[slot * ffn + feature] = static_cast<std::int8_t>(ti::lo_code(h, e, c1));
        }
    }
}

// TURBO v2 input limb, one block per row after the prologue (same stream): lo = lo_code(x, A8 delta, A8 code) of the BF16
// MoE input; a row the prologue refused (status != 0, codes not written) gets zeros (its experts are never run).
template <class Binding>
__global__ __launch_bounds__(256) void turbo_input_lo_kernel(typename Binding::Rows v) {
    constexpr unsigned hidden = Binding::hidden;
    const unsigned row = blockIdx.x;
    const bool ok = *at(v.status, row) == 0U;
    const std::uint16_t* input = at(v.input, row);
    const std::int8_t* codes = at(v.codes, row);
    const float* deltas = at(v.deltas, row);
    std::int8_t* lo = at(v.lo_codes, row);
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float x = __uint_as_float(static_cast<unsigned>(input[i]) << 16U);
        lo[i] = ok ? static_cast<std::int8_t>(ti::lo_code(x, deltas[i / 32U], codes[i])) : std::int8_t(0);
    }
}

// grid (tiles, hidden / 128), 256 threads. Reads the A8 down codes / deltas of each (row, slot) and writes FP32
// rawdots[row][slot * hidden + o] (the native route's storage; merge / commit unchanged).
template <unsigned T, unsigned Limbs, class Binding>
__global__ __launch_bounds__(kTurboThreads, 2) void turbo_down_kernel(const std::uint8_t* down, typename Binding::Rows v) {
    using Decode = weight::GsqDecode<T>;
    constexpr unsigned hidden = Binding::hidden, ffn = Binding::ffn;
    constexpr unsigned stages = ffn / ti::kStageK, groups_per_stage = ti::kStageK / 32U;
    constexpr std::uint64_t row_bytes = std::uint64_t(ffn / Decode::be) * Decode::bb;
    static_assert(ffn % ti::kStageK == 0U && ffn % Decode::be == 0U && hidden % kTurboDownOutputs == 0U,
                  "TURBO down covers whole stages, storage blocks and output blocks");
    constexpr unsigned kStage = kStageBytesOf<Limbs>;
    const auto& csr = v.csr;
    const native_mma::TileSlice ts = native_mma::tile_major(blockIdx.x, blockIdx.y, gridDim.x, hidden / kTurboDownOutputs);
    if (csr.summary[0] || ts.tile >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[ts.tile], count = csr.tile_begins[ts.tile + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[ts.tile];
    __shared__ alignas(16) unsigned char smem[2U * kStage];
    __shared__ unsigned row_of[ti::kTileRows], slot_of[ti::kTileRows];
    __shared__ const std::int8_t* code_rows[ti::kTileRows];
    __shared__ const std::int8_t* lo_rows[ti::kTileRows];
    __shared__ const float* delta_rows[ti::kTileRows];
    constexpr unsigned kOwnSmem = 2U * kStage + kTurboRowTables;
    __shared__ std::uint32_t grid_words[kTurboSmemGrid<T, kOwnSmem> ? Decode::grid_words : 1U];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, g = lane >> 2;
    const unsigned out_base = ti::down_output(ts.slice, warp, 0U, 0U);
    if (t < ti::kTileRows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[tile_begin + a]; slot_of[t] = csr.slots[tile_begin + a];
        code_rows[t] = at(v.down_codes, row_of[t]) + slot_of[t] * ffn;
        lo_rows[t] = Limbs == 2U ? at(v.down_lo_codes, row_of[t]) + slot_of[t] * ffn : nullptr;
        delta_rows[t] = at(v.down_deltas, row_of[t]) + slot_of[t] * (ffn / 32U);
    }
    if constexpr (kTurboSmemGrid<T, kOwnSmem>) weight::gsq_grid_fill<T>(grid_words, t, kTurboThreads);
    __syncthreads();
    const auto grid = turbo_grid<T, kOwnSmem>(grid_words);
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
    turbo_stage<Limbs>(smem, count, t, code_rows, lo_rows, delta_rows, 0U);
    constexpr unsigned groups = stages * groups_per_stage;  // fields one group ahead, as in turbo_project_kernel
    typename Decode::Raw ahead[2] = {turbo_raw<Decode>(wrow[0], 0U, lane), turbo_raw<Decode>(wrow[1], 0U, lane)};
#pragma unroll 1
    for (unsigned s = 0; s < stages; ++s) {
        cp_wait_all();
        __syncthreads();
        if (s + 1U < stages) turbo_stage<Limbs>(smem + ((s + 1U) & 1U) * kStage, count, t, code_rows, lo_rows, delta_rows, s + 1U);
        const unsigned char* A = smem + (s & 1U) * kStage;
        const float* E = reinterpret_cast<const float*>(A + Limbs * kStageA);
#pragma unroll
        for (unsigned j = 0; j < groups_per_stage; ++j) {
            const unsigned group = s * groups_per_stage + j, next = group + 1U < groups ? group + 1U : group;
            const typename Decode::Raw raw[2] = {ahead[0], ahead[1]};
            ahead[0] = turbo_raw<Decode>(wrow[0], next, lane);
            ahead[1] = turbo_raw<Decode>(wrow[1], next, lane);
            unsigned b[2][2];
            float scale[2][2][2];
            turbo_weights<Decode>(raw[0], group, grid, lane, b[0], scale[0]);
            turbo_weights<Decode>(raw[1], group, grid, lane, b[1], scale[1]);
#pragma unroll
            for (unsigned m = 0; m < 4U; ++m) {
                if (m >= mblocks) break;
                const unsigned row0 = 16U * m + g;
                const float e[2] = {E[4U * row0 + j], E[4U * (row0 + 8U) + j]};
#pragma unroll
                for (unsigned n = 0; n < 2U; ++n) turbo_group<Decode::halves, Limbs>(acc[m][n], A, row0, j, lane, b[n], scale[n], e);
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
