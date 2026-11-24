#include "grouped_w4a4.hpp"

#include "arch/sm120/grouped_w4a4.cuh"

namespace strix::cuda::core::grouped_w4a4 {
namespace {
namespace contract = ::strix::moe_core::grouped_w4a4;

template <class T>
cudaError_t quantize_rows(const contract::QuantizeArgs<T>& args, cudaStream_t stream, contract::ScaleRule rule) {
    if (!args.source || args.rows == 0U || args.source_row < args.width ||
        reinterpret_cast<std::uintptr_t>(args.source) % 16U != 0U || (args.source_row * sizeof(T)) % 16U != 0U ||
        !contract::fits_quantize(args.output, args.width, args.terms))
        return cudaErrorInvalidValue;
    if (args.terms == 2U) {
        if (rule == contract::ScaleRule::mse)
            sm120::quantize_rows<T, true, 2U><<<(args.rows + 7U) / 8U, 256, 0, stream>>>(args);
        else
            sm120::quantize_rows<T, false, 2U><<<(args.rows + 7U) / 8U, 256, 0, stream>>>(args);
    } else if (rule == contract::ScaleRule::mse)
        sm120::quantize_rows<T, true><<<(args.rows + 7U) / 8U, 256, 0, stream>>>(args);
    else
        sm120::quantize_rows<T><<<(args.rows + 7U) / 8U, 256, 0, stream>>>(args);
    return cudaGetLastError();
}

// smem_bytes(Terms) of dynamic shared memory (80 KiB one term, 90 KiB two): above the 48 KiB default, so the kernel's limit is
// raised (per device, idempotent).
template <unsigned Terms>
cudaError_t launch_projection(const contract::Args& args, dim3 grid, cudaStream_t stream) {
    constexpr unsigned smem = sm120::smem_bytes(Terms);
    const cudaError_t attr =
        cudaFuncSetAttribute(sm120::projection<Terms>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (attr != cudaSuccess) return attr;
    sm120::projection<Terms><<<grid, sm120::kThreads, smem, stream>>>(args);
    return cudaGetLastError();
}
}  // namespace

cudaError_t quantize(const contract::QuantizeArgs<float>& args, cudaStream_t stream, contract::ScaleRule rule) {
    return quantize_rows(args, stream, rule);
}
cudaError_t quantize(const contract::QuantizeArgs<std::uint16_t>& args, cudaStream_t stream, contract::ScaleRule rule) {
    return quantize_rows(args, stream, rule);
}

cudaError_t launch(const contract::Args& args, cudaStream_t stream) {
    if (!contract::fits(args)) return cudaErrorInvalidValue;
    const unsigned bound = args.tiles.max_tile_rows == 0U ? contract::kMaxTileRows : args.tiles.max_tile_rows;
    const dim3 grid(args.tiles.outputs / contract::kTileOutputs, (bound + contract::kTileRows - 1U) / contract::kTileRows,
                    args.tiles.count);
    return args.terms == 2U ? launch_projection<2U>(args, grid, stream) : launch_projection<1U>(args, grid, stream);
}

}  // namespace strix::cuda::core::grouped_w4a4
