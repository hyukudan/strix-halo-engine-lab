#pragma once
// sm_120 body of the grouped W4A4 expert projection (contract and identity: moe_core/grouped_w4a4.hpp; stage layout and
// coordinate maps: grouped_w4a4_layout.hpp). The activation quantiser is the family rule of nvfp4_tile.cuh (encode_values),
// stored canonically.
#include <cuda_runtime.h>

#include "moe_core/grouped_w4a4.hpp"
#include "grouped_w4a4_layout.hpp"
#include "strix/cuda/core/nvfp4_tile.hpp"
#include "nvfp4_mma.cuh"
#include "nvfp4_quant.cuh"
#include "../../ptx.cuh"

namespace strix::cuda::core::grouped_w4a4::sm120 {

namespace contract = ::strix::moe_core::grouped_w4a4;
namespace packed = ::strix::weight::microscaled;
namespace quant = ::strix::cuda::core::nvfp4::tile::sm120;
static_assert(contract::kRowScaleDiv == ::strix::cuda::core::nvfp4::tile::kRowScaleDiv, "one activation quantiser rule");
static_assert(contract::kTileRows == kRows && contract::kTileOutputs == kOutputs && contract::kInputBlock == kStageInputs);

// One warp per destination row, 8 rows per CTA. Row amax over 16-byte vectors (exact in any order), ts = amax / 2688,
// then group q -> encode_values (Mse: encode_values_mse, scale rule mse@1 of moe_core/nvfp4_act_rule.hpp) -> 8 code bytes at
// 8 q and scale byte q (canonical row-major). A bool, not the rule enum: the host stub of an enum-alias kernel parameter does
// not compile.
template <class T, bool Mse = false>
__global__ void __launch_bounds__(256) quantize_rows(const contract::QuantizeArgs<T> a) {
    if (a.failure && *a.failure != 0U) return;
    const unsigned lane = threadIdx.x & 31U, row = blockIdx.x * 8U + (threadIdx.x >> 5U);
    if (row >= a.rows) return;
    const std::size_t source_row = a.index ? a.index[row] : row;
    const T* src = a.source + source_row * a.source_row;
    constexpr unsigned kPerVec = 16U / sizeof(T);
    float amax = 0.f;
    for (unsigned i = lane; i < a.width / kPerVec; i += 32U) amax = fmaxf(amax, quant::abs_max16(src, i));
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    const float ts = __fdiv_rn(amax, contract::kRowScaleDiv);
    std::uint8_t* codes = a.output.codes + std::size_t{row} * a.output.code_row;
    std::uint8_t* scales = a.output.scales + std::size_t{row} * a.output.scale_row;
    if (lane == 0U) a.output.row_scale[row] = ts;
    for (unsigned q = lane; q < a.width / contract::kGroup; q += 32U) {
        float v[16];
        quant::load_row_group(src, q, v);
        std::uint32_t word[2];
        if constexpr (Mse) scales[q] = quant::encode_values_mse(v, ts, word);
        else scales[q] = quant::encode_values(v, ts, word);
        *reinterpret_cast<uint2*>(codes + 8U * q) = make_uint2(word[0], word[1]);
    }
}

// Grid (outputs / 128, row slices, tile count); dynamic shared memory kSmem. Slice y takes tile rows 128 y .. 128 y + 127 (one
// slice for every plan tile of <= 128 rows). Stage s (128 inputs) of weight row o: codes + 64 s, scales + 8 s of physical row o;
// activation row r < slice rows: image row first + r; activation code rows past the live rows up to a multiple of 16 are
// zero-filled (the x4 B fragment load reads n8 blocks in pairs; product 0, never stored), scale rows up to a multiple of 8
// (only live n8 blocks read scale words); rows past those are neither copied nor read.
__global__ void __launch_bounds__(kThreads, 1) projection(const contract::Args a) {
    const auto& args = a.tiles;
    if ((args.failure && *args.failure != 0U) || (args.tile_count && blockIdx.z >= *args.tile_count)) return;
    contract::TileRows tile = args.tiles[blockIdx.z];
    const unsigned first_row = blockIdx.y * kRows;
    if (first_row >= tile.rows) return;
    tile.input += std::size_t{first_row} * tile.input_row;
    tile.output += std::size_t{first_row} * tile.output_row;
    tile.rows -= first_row;
    const unsigned rows = tile.rows < kRows ? tile.rows : kRows, rows8 = (rows + 7U) & ~7U, rows16 = (rows + 15U) & ~15U;
    extern __shared__ __align__(16) std::uint8_t smem[];
    const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
    const unsigned thread = threadIdx.x, lane = thread & 31U, warp = thread >> 5U;
    const unsigned first_output = blockIdx.x * kOutputs;
    const std::size_t first = static_cast<std::size_t>(tile.input - a.row_base) / tile.input_row;
    const std::size_t w_code_row = args.inputs / 2U, w_scale_row = args.inputs / contract::kGroup;

    // Per-thread copy sources, resolved once; a stage advances codes by 64 B and scales by 8 B.
    const std::uint8_t* code_src[kCodeChunksPerThread];
    std::uint32_t code_dst[kCodeChunksPerThread];
    bool code_ok[kCodeChunksPerThread], code_issue[kCodeChunksPerThread];
#pragma unroll
    for (unsigned j = 0; j < kCodeChunksPerThread; ++j) {
        const Chunk c = code_chunk(thread, j);
        if (c.activation) {
            code_ok[j] = c.row < rows;
            code_issue[j] = c.row < rows16;
            code_src[j] = a.input.codes + (first + (code_ok[j] ? c.row : 0U)) * a.input.code_row + 16U * c.part;
        } else {
            const auto row = packed::physical_row(tile.weights.row_order, tile.weights.rows, first_output + c.row);
            code_ok[j] = code_issue[j] = true;
            code_src[j] = tile.weights.codes + row * w_code_row + 16U * c.part;
        }
        code_dst[j] = base + stage_offset(c.activation, c.row, 16U * c.part);
    }
    const Chunk sc = scale_chunk(thread);
    bool scale_ok = true, scale_issue = true;
    const std::uint8_t* scale_src = nullptr;
    if (sc.activation) {
        scale_ok = sc.row < rows;
        scale_issue = sc.row < rows8;
        scale_src = a.input.scales + (first + (scale_ok ? sc.row : 0U)) * a.input.scale_row;
    } else {
        const auto row = packed::physical_row(tile.weights.row_order, tile.weights.rows, first_output + sc.row);
        scale_src = tile.weights.block_scales + row * w_scale_row;
    }
    const std::uint32_t scale_dst = base + stage_offset(sc.activation, sc.row, kScaleOffset);
    const auto load = [&](unsigned stage, unsigned slot) {
#pragma unroll
        for (unsigned j = 0; j < kCodeChunksPerThread; ++j)
            if (code_issue[j])
                ::strix::cuda::core::ptx::cp_async16_zfill(code_dst[j] + slot * kStageBytes, code_src[j] + std::size_t{kCodeBytes} * stage, code_ok[j]);
        if (scale_issue)
            ::strix::cuda::core::ptx::cp_async8_zfill(scale_dst + slot * kStageBytes, scale_src + std::size_t{kScaleBytes} * stage, scale_ok);
    };

    // Fragment offsets inside a stage (64-input block kb adds 32 B to codes, 4 B to scales).
    const unsigned live = live_blocks(warp, rows);
    unsigned a_off[4], sa_off[4], b_off[2], sb_off[4];
#pragma unroll
    for (unsigned mt = 0; mt < 4U; ++mt) {
        a_off[mt] = stage_offset(false, ldsm_a_row(warp, mt, lane), ldsm_a_byte(0U, lane));
        sa_off[mt] = stage_offset(false, a_scale_row(warp, mt, lane), scale_byte(0U));
    }
#pragma unroll
    for (unsigned p = 0; p < 2U; ++p) b_off[p] = stage_offset(true, ldsm_b_row(warp, p, lane), ldsm_b_byte(0U, lane));
#pragma unroll
    for (unsigned nt = 0; nt < 4U; ++nt) sb_off[nt] = stage_offset(true, b_scale_row(warp, nt, lane), scale_byte(0U));

    const unsigned stages = args.inputs / kStageInputs;
    float acc[4][4][4]{};
#pragma unroll
    for (unsigned s = 0; s < kStages - 1U; ++s) {
        if (s < stages) load(s, s);
        ::strix::cuda::core::ptx::cp_async_commit();
    }
    for (unsigned s = 0; s < stages; ++s) {
        ::strix::cuda::core::ptx::cp_async_wait<static_cast<int>(kStages - 2U)>();
        __syncthreads();  // stage s visible; every warp is past stage s - 1, whose slot the next load reuses
        if (s + kStages - 1U < stages) load(s + kStages - 1U, (s + kStages - 1U) % kStages);
        ::strix::cuda::core::ptx::cp_async_commit();
        if (live == 0U) continue;  // warp-uniform: no live row in this warp's quarter
        const std::uint8_t* stage = smem + (s % kStages) * kStageBytes;
#pragma unroll
        for (unsigned kb = 0; kb < 2U; ++kb) {
            unsigned af[4][4], sa[4], bq[2][4], sb[4];
#pragma unroll
            for (unsigned mt = 0; mt < 4U; ++mt) {
                ::strix::cuda::core::ptx::ldsm_x4(af[mt], stage + a_off[mt] + 32U * kb);
                sa[mt] = *reinterpret_cast<const unsigned*>(stage + sa_off[mt] + 4U * kb);
            }
#pragma unroll
            for (unsigned p = 0; p < 2U; ++p)
                if (2U * p < live) ::strix::cuda::core::ptx::ldsm_x4(bq[p], stage + b_off[p] + 32U * kb);
#pragma unroll
            for (unsigned nt = 0; nt < 4U; ++nt) {
                if (nt >= live) continue;
                sb[nt] = *reinterpret_cast<const unsigned*>(stage + sb_off[nt] + 4U * kb);
#pragma unroll
                for (unsigned mt = 0; mt < 4U; ++mt)
                    ::strix::cuda::core::nvfp4::mma_detail::m16n8k64(acc[mt][nt], af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                                                                    bq[nt >> 1U][2U * (nt & 1U)], bq[nt >> 1U][2U * (nt & 1U) + 1U],
                                                                    sa[mt], sb[nt]);
            }
        }
    }
    ::strix::cuda::core::ptx::cp_async_wait<0>();

    const float outer = tile.weights.outer_scale;
#pragma unroll
    for (unsigned nt = 0; nt < 4U; ++nt) {
        if (nt >= live) continue;
#pragma unroll
        for (unsigned z = 0; z < 4U; ++z) {
            const unsigned row = acc_row(warp, nt, lane, z);
            if (row >= rows) continue;
            const float rs = a.input.row_scale[first + row];
#pragma unroll
            for (unsigned mt = 0; mt < 4U; ++mt)
                tile.output[std::size_t{row} * tile.output_row + first_output + acc_output(warp, mt, lane, z)] =
                    __fmul_rn(acc[mt][nt][z], __fmul_rn(outer, rs));
        }
    }
}

}  // namespace strix::cuda::core::grouped_w4a4::sm120
