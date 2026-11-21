#pragma once

#include "cuda/core/grouped_dispatch.cuh"
#include "inkling2_mini_moe.hpp"
#include "inkling2_mini_moe_dispatch.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#include "strix/cuda/core/moe_route_tap.hpp"
#endif

namespace strix::inkling2 {

using MiniCudaMoeBackend = cuda::core::grouped_dispatch::CudaBackend;

#if defined(__CUDACC__) && !defined(__HIPCC__)
inline constexpr auto enqueue_mini_cuda_router_logits = cuda::core::grouped_dispatch::router_logits<MiniMoeConfig>;

inline cudaError_t enqueue_mini_cuda_resident_moe(
    unsigned physical_layer, unsigned rows, std::span<const float> input, std::span<const float> logits,
    std::span<const float> selection_bias, std::span<float> output, MiniMoeResidentRouteBuffers route,
    std::span<const MiniMoeExpert> matrices, moe_core::device_grouped::CsrTables csr,
    moe_core::device_grouped::Output planned, MiniMoeWorkspace workspace, cudaStream_t stream,
    moe::RouteTap* tap = nullptr) {
    const auto select = [physical_layer, tap](std::span<const float> scores, std::span<const float> bias,
                           MiniMoeResidentRouteBuffers buffers, unsigned count, cudaStream_t route_stream) {
        const auto status = moe_core::enqueue_route_wave<kMiniRoute, detail::MiniMoeMath, moe::RouteWaveTraits<true>>(
            scores.data(), bias.data(), buffers.ids.data(), buffers.weights.data(),
            buffers.status.data(), count, route_stream);
        if (status == cudaSuccess)
            cuda::core::route_tap(physical_layer, count, kExpertsPerToken, kExperts, buffers.ids.data(),
                                  buffers.weights.data(), route_stream, tap);
        return status;
    };
    return mini_grouped::enqueue_resident<MiniMoeConfig, MiniCudaMoeBackend>(
        physical_layer, rows, input, logits, selection_bias, output, route, matrices, csr, planned, workspace, stream,
        mini_grouped::LogitExtent::at_least, select);
}
#endif

}  // namespace strix::inkling2
