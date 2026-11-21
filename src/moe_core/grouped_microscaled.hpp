#pragma once

#include "core/f16_tile_gemm.hpp"
#include "strix/weight/formats/microscaled.hpp"

#include <cmath>

namespace strix::moe_core::grouped_microscaled {

namespace gemm = strix::core::f16_tile_gemm;
namespace packed = strix::weight::microscaled;

// One expert and up to one shared GEMM row tile of its routed activations.
// A larger group supplies more descriptors with advanced input/output pointers.
// Input and output strides are in elements and are independent of each other.
// Descriptor pointers address resident device allocations when used by launch().
struct TileRows {
    packed::PackedMatrix weights;
    const std::uint16_t* input{};
    float* output{};
    std::size_t input_row{};
    std::size_t output_row{};
    unsigned rows{};
};

struct Args {
    const TileRows* tiles{};
    unsigned count{};
    unsigned inputs{};
    unsigned outputs{};
};

inline bool fits(const Args& args) {
    return args.tiles && args.count >= 1U && args.count <= 65535U &&
           args.inputs >= gemm::kBK && args.inputs % gemm::kBK == 0U &&
           args.outputs >= gemm::kBN && args.outputs % gemm::kBN == 0U;
}

template <packed::MatrixFormat Format>
inline bool fits_tile(const TileRows& tile, unsigned inputs, unsigned outputs) {
    const auto& w = tile.weights;
    constexpr auto code_alignment = packed::PackedOperand<Format>::kCodes;
    if (!w.codes || !w.block_scales || w.rows != outputs || w.columns != inputs ||
        inputs % gemm::kBK != 0 || reinterpret_cast<std::uintptr_t>(w.codes) % code_alignment != 0 ||
        !tile.input || !tile.output || tile.rows == 0 || tile.rows > gemm::kBM ||
        tile.input_row < inputs || tile.output_row < outputs || tile.input_row % 8U != 0 ||
        reinterpret_cast<std::uintptr_t>(tile.input) % 16U != 0) return false;
    if (w.row_order != packed::RowOrder::contiguous &&
        (w.row_order != packed::RowOrder::interleaved_halves || w.rows % 2U != 0)) return false;
    if constexpr (Format == packed::MatrixFormat::nvfp4)
        return std::isfinite(w.outer_scale) && w.outer_scale >= 0.0F;
    return true;
}

#if defined(__HIPCC__)

// One launch handles all active experts of one projection and checkpoint
// format. Operands round to BF16 at load, while WMMA accumulation is FP32 in
// hardware order. Qualify the numerical identity against the reference before
// admitting it for a model. Only gfx11 wave32 WMMA devices are supported.
template <packed::MatrixFormat Format>
__global__ __launch_bounds__(gemm::kThreads) void kernel(Args args) {
    __shared__ gemm::Storage storage;
    const TileRows rows = args.tiles[blockIdx.z];
    gemm::Args shape{};
    shape.a16 = rows.input;
    shape.a_row = rows.input_row;
    shape.y_row = rows.output_row;
    shape.rows = rows.rows;
    shape.K = args.inputs;
    shape.N = args.outputs;
    const gemm::RawStore store{rows.output, rows.output_row, 0};
    gemm::core<gemm::Layout::NK, gemm::RawStore, true>(
        shape, storage, store, packed::PackedOperand<Format>{rows.weights});
}

// Host admission checks the tile descriptors before upload with fits_tile().
// Invoke on a stream for a supported gfx11 device after device admission.
template <packed::MatrixFormat Format>
inline hipError_t launch(const Args& args, hipStream_t stream) {
    if (!fits(args)) return hipErrorInvalidValue;
    kernel<Format><<<dim3(args.outputs / gemm::kBN, 1U, args.count), gemm::kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

#endif

}  // namespace strix::moe_core::grouped_microscaled
