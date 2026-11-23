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
// then group q -> encode_values -> 8 code bytes at 8 q and scale byte q (canonical row-major).
template <class T>
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
        scales[q] = quant::encode_values(v, ts, word);
        *reinterpret_cast<uint2*>(codes + 8U * q) = make_uint2(word[0], word[1]);
    }
}

// Grid (outputs / 128, row slices, tile count). Slice y takes tile rows 64 y .. 64 y + 63. Stage s (128 inputs) of weight row
// o: codes + 64 s, scales + 8 s of physical row o; activation row r < slice rows: image row first + r, else zero-filled (zero
// codes and scales, product 0, never stored).
__global__ void __launch_bounds__(kThreads, 2) projection(const contract::Args a) {
    const auto& args = a.tiles;
    if ((args.failure && *args.failure != 0U) || (args.tile_count && blockIdx.z >= *args.tile_count)) return;
    contract::TileRows tile = args.tiles[blockIdx.z];
    const unsigned first_row = blockIdx.y * kRows;
    if (first_row >= tile.rows) return;
    tile.input += std::size_t{first_row} * tile.input_row;
    tile.output += std::size_t{first_row} * tile.output_row;
    tile.rows -= first_row;
    __shared__ __align__(16) std::uint8_t smem[kSmem];
    const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
    const unsigned thread = threadIdx.x, lane = thread & 31U, warp = thread >> 5U;
    const unsigned first_output = blockIdx.x * kOutputs;
    const std::size_t first = static_cast<std::size_t>(tile.input - a.row_base) / tile.input_row;
    const std::size_t w_code_row = args.inputs / 2U, w_scale_row = args.inputs / contract::kGroup;

    // Per-thread copy sources, resolved once; a stage advances codes by 64 B and scales by 8 B.
    const std::uint8_t* code_src[kCodeChunksPerThread];
    std::uint32_t code_dst[kCodeChunksPerThread];
    bool code_ok[kCodeChunksPerThread];
#pragma unroll
    for (unsigned j = 0; j < kCodeChunksPerThread; ++j) {
        const Chunk c = code_chunk(thread, j);
        if (c.activation) {
            code_ok[j] = c.row < tile.rows;
            code_src[j] = a.input.codes + (first + (code_ok[j] ? c.row : 0U)) * a.input.code_row + 16U * c.part;
        } else {
            const auto row = packed::physical_row(tile.weights.row_order, tile.weights.rows, first_output + c.row);
            code_ok[j] = true;
            code_src[j] = tile.weights.codes + row * w_code_row + 16U * c.part;
        }
        code_dst[j] = base + stage_offset(c.activation, c.row, 16U * c.part);
    }
    const bool scale_copy = has_scale_chunk(thread);
    const std::uint8_t* scale_src = nullptr;
    std::uint32_t scale_dst = 0U;
    bool scale_ok = false;
    if (scale_copy) {
        const Chunk c = scale_chunk(thread);
        if (c.activation) {
            scale_ok = c.row < tile.rows;
            scale_src = a.input.scales + (first + (scale_ok ? c.row : 0U)) * a.input.scale_row;
        } else {
            const auto row = packed::physical_row(tile.weights.row_order, tile.weights.rows, first_output + c.row);
            scale_ok = true;
            scale_src = tile.weights.block_scales + row * w_scale_row;
        }
        scale_dst = base + stage_offset(c.activation, c.row, kScaleOffset);
    }
    const auto load = [&](unsigned stage, unsigned slot) {
#pragma unroll
        for (unsigned j = 0; j < kCodeChunksPerThread; ++j)
            ::strix::cuda::core::ptx::cp_async16_zfill(code_dst[j] + slot * kStageBytes, code_src[j] + std::size_t{kCodeBytes} * stage, code_ok[j]);
        if (scale_copy) ::strix::cuda::core::ptx::cp_async8_zfill(scale_dst + slot * kStageBytes, scale_src + std::size_t{kScaleBytes} * stage, scale_ok);
    };

    const unsigned stages = args.inputs / kStageInputs;
    float acc[2][4][4]{};
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
        const std::uint8_t* stage = smem + (s % kStages) * kStageBytes;
#pragma unroll
        for (unsigned kb = 0; kb < 2U; ++kb) {
            unsigned af[2][4], sa[2];
#pragma unroll
            for (unsigned mb = 0; mb < 2U; ++mb) {
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i)
                    af[mb][i] = *reinterpret_cast<const unsigned*>(
                        stage + stage_offset(false, a_row(warp, mb, lane, i), a_byte(kb, lane, i)));
                sa[mb] = *reinterpret_cast<const unsigned*>(
                    stage + stage_offset(false, a_scale_row(warp, mb, lane), scale_byte(kb)));
            }
#pragma unroll
            for (unsigned nb = 0; nb < 4U; ++nb) {
                const unsigned r = b_row(warp, nb, lane);
                const unsigned bf[2] = {
                    *reinterpret_cast<const unsigned*>(stage + stage_offset(true, r, b_byte(kb, lane, 0U))),
                    *reinterpret_cast<const unsigned*>(stage + stage_offset(true, r, b_byte(kb, lane, 1U)))};
                const unsigned sb = *reinterpret_cast<const unsigned*>(stage + stage_offset(true, r, scale_byte(kb)));
#pragma unroll
                for (unsigned mb = 0; mb < 2U; ++mb)
                    ::strix::cuda::core::nvfp4::mma_detail::m16n8k64(acc[mb][nb], af[mb][0], af[mb][1], af[mb][2],
                                                                    af[mb][3], bf[0], bf[1], sa[mb], sb);
            }
        }
    }
    ::strix::cuda::core::ptx::cp_async_wait<0>();

    const float outer = tile.weights.outer_scale;
#pragma unroll
    for (unsigned mb = 0; mb < 2U; ++mb) {
#pragma unroll
        for (unsigned nb = 0; nb < 4U; ++nb) {
#pragma unroll
            for (unsigned z = 0; z < 4U; ++z) {
                const unsigned row = acc_row(warp, nb, lane, z);
                if (row >= tile.rows) continue;
                const float rs = a.input.row_scale[first + row];
                tile.output[std::size_t{row} * tile.output_row + first_output + acc_output(warp, mb, lane, z)] =
                    __fmul_rn(acc[mb][nb][z], __fmul_rn(outer, rs));
            }
        }
    }
}

}  // namespace strix::cuda::core::grouped_w4a4::sm120
