#pragma once

#include "core/f16_tile_gemm.hpp"
#include "moe_core/grouped_microscaled_rows.hpp"

namespace strix::moe_core::grouped_microscaled {

namespace gemm = strix::core::f16_tile_gemm;

inline bool fits(const Args& args) {
    return fits_args(args, gemm::kBK, gemm::kBN);
}

template <packed::MatrixFormat Format>
inline bool fits_tile(const TileRows& tile, unsigned inputs, unsigned outputs) {
    return fits_rows<Format>(tile, inputs, outputs, gemm::kBK, gemm::kBM);
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
