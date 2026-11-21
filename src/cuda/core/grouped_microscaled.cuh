#pragma once

#include "moe_core/grouped_microscaled_rows.hpp"

#if defined(__CUDACC__)
#include "native_mma.hpp"
#include "ptx.cuh"
#endif

namespace strix::cuda::core::grouped_microscaled {

namespace view = strix::moe_core::grouped_microscaled;
namespace packed = weight::microscaled;
using view::Args;
using view::TileRows;

inline constexpr unsigned kRowsPerTile = 16;
inline constexpr unsigned kOutputTile = 32;
inline constexpr unsigned kInputAlignment = 32;

inline bool fits(const Args& args) {
    return view::fits_args(args, kInputAlignment, kOutputTile);
}

template <packed::MatrixFormat Format>
inline bool fits_tile(const TileRows& tile, unsigned inputs, unsigned outputs) {
    return outputs % kOutputTile == 0 && view::fits_rows<Format>(tile, inputs, outputs, kInputAlignment, kRowsPerTile);
}

#if defined(__CUDACC__)
namespace detail {

inline constexpr unsigned kKStep = 16;
inline constexpr unsigned kPitch = kKStep / 2U + 4U;  // BF16 pairs plus 16-byte padding.
inline constexpr unsigned kThreads = 128;

template <packed::MatrixFormat Format>
__device__ __forceinline__ typename packed::PackedOperand<Format>::Tile16 load_aligned_tile16(
    const packed::PackedOperand<Format>& source, unsigned row, unsigned k) {
    using Operand = packed::PackedOperand<Format>;
    const auto address = source.tile16_address(row, k);
    std::uint32_t words[Operand::kCodes / 4U];
    if constexpr (Operand::kNvfp4) {
        asm("ld.global.v2.u32 {%0, %1}, [%2];" : "=r"(words[0]), "=r"(words[1]) : "l"(address.codes));
    } else {
        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3]) : "l"(address.codes));
    }
    typename Operand::Tile16 tile;
#pragma unroll
    for (unsigned i = 0; i < Operand::kCodes; ++i) {
        tile.codes[i] = static_cast<std::uint8_t>(words[i / 4U] >> (8U * (i % 4U)));
    }
    tile.scale_code = *address.scale_code;
    return tile;
}

// CTA covers up to 16 routed rows and 32 output channels of one expert.
// Each B row is decoded once per K step and reused by all live token rows.
// CUDA and HIP consume the same packed matrices and produce FP32 partials.
template <packed::MatrixFormat Format>
__global__ __launch_bounds__(kThreads) void kernel(Args args) {
    const TileRows rows = args.tiles[blockIdx.z];
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread & 31U;
    const unsigned warp = thread >> 5U;
    const auto first_output = blockIdx.x * kOutputTile;
    __shared__ __align__(16) std::uint32_t activations[kRowsPerTile][kPitch];
    __shared__ __align__(16) std::uint32_t weights[kOutputTile][kPitch];
    const packed::PackedOperand<Format> source{rows.weights};
    float acc[4]{};

    for (unsigned k = 0; k < args.inputs; k += kKStep) {
        const auto input_row = thread / 8U;
        const auto pair = thread % 8U;
        std::uint32_t word = 0;
        if (input_row < rows.rows) {
            const auto* values = rows.input + std::size_t{input_row} * rows.input_row + k + 2U * pair;
            // Tile admission guarantees row and pair alignment for this BF16x2 load.
            asm("ld.global.u32 %0, [%1];" : "=r"(word) : "l"(values));
        }
        activations[input_row][pair] = word;
        if (thread < kOutputTile) {
            auto& weight_row = weights[thread];
            std::uint32_t decoded[8];
            const auto tile = load_aligned_tile16(source, first_output + thread, k);
            source.decode_bf16(tile, decoded);
#pragma unroll
            for (unsigned i = 0; i < 8U; ++i) weight_row[i] = decoded[i];
        }
        __syncthreads();

        unsigned a[4], w[2];
        ptx::ldsm_x4(a, &activations[lane & 15U][(lane >> 4U) * 4U]);
        ptx::ldsm_x2(w, &weights[8U * warp + (lane & 7U)][((lane >> 3U) & 1U) * 4U]);
        ptx::mma_bf16(acc, a[0], a[1], a[2], a[3], w[0], w[1]);
        __syncthreads();
    }

#pragma unroll
    for (unsigned element = 0; element < 4U; ++element) {
        const auto m = native_mma::c_row(lane, element);
        const auto n = first_output + 8U * warp + native_mma::c_col(lane, element);
        if (m < rows.rows) rows.output[std::size_t{m} * rows.output_row + n] = acc[element];
    }
}

}  // namespace detail

// Caller admits sm_80+ and validates device tile descriptors before upload.
// Tensor-core accumulation order requires independent device fidelity checks.
template <packed::MatrixFormat Format>
inline cudaError_t launch(const Args& args, cudaStream_t stream) {
    if (!fits(args)) return cudaErrorInvalidValue;
    detail::kernel<Format><<<dim3(args.outputs / kOutputTile, 1U, args.count), detail::kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

#endif
}  // namespace strix::cuda::core::grouped_microscaled
