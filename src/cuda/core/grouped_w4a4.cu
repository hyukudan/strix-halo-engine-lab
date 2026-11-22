#include "grouped_w4a4.hpp"

#include "arch/sm120/grouped_w4a4.cuh"

namespace strix::cuda::core::grouped_w4a4 {
namespace {
namespace contract = ::strix::moe_core::grouped_w4a4;

template <class T>
cudaError_t quantize_rows(const contract::QuantizeArgs<T>& args, cudaStream_t stream) {
    if (!args.source || args.rows == 0U || args.source_row < args.width ||
        reinterpret_cast<std::uintptr_t>(args.source) % 16U != 0U || (args.source_row * sizeof(T)) % 16U != 0U ||
        !contract::fits_image(args.output, args.width))
        return cudaErrorInvalidValue;
    sm120::quantize_rows<T><<<(args.rows + 7U) / 8U, 256, 0, stream>>>(args);
    return cudaGetLastError();
}
}  // namespace

cudaError_t quantize(const contract::QuantizeArgs<float>& args, cudaStream_t stream) { return quantize_rows(args, stream); }
cudaError_t quantize(const contract::QuantizeArgs<std::uint16_t>& args, cudaStream_t stream) {
    return quantize_rows(args, stream);
}

cudaError_t launch(const contract::Args& args, cudaStream_t stream) {
    if (!contract::fits(args)) return cudaErrorInvalidValue;
    const unsigned bound = args.tiles.max_tile_rows == 0U ? contract::kMaxTileRows : args.tiles.max_tile_rows;
    const dim3 grid(args.tiles.outputs / contract::kTileOutputs, (bound + contract::kTileRows - 1U) / contract::kTileRows,
                    args.tiles.count);
    sm120::projection<<<grid, sm120::kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

}  // namespace strix::cuda::core::grouped_w4a4
