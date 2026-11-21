#pragma once

#include "inkling2_mini_expert_hip.hpp"
#include "inkling2_mini_moe.hpp"

#if defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#endif

namespace strix::inkling2 {

#if defined(__HIPCC__)
inline hipError_t enqueue_mini_hip_resident_moe(
    unsigned physical, unsigned rows, std::span<const float> input,
    std::span<const float> router_logits, std::span<const float> bias, std::span<float> output,
    MiniMoeResidentRouteBuffers buffers, std::span<const MiniMoeExpert> matrices,
    moe_core::device_grouped::CsrTables csr, moe_core::device_grouped::Output out,
    MiniMoeWorkspace workspace, hipStream_t stream) {
    const auto select = [](std::span<const float> logits, std::span<const float> selection_bias,
                           MiniMoeResidentRouteBuffers route, unsigned count, hipStream_t route_stream) {
        return moe_core::enqueue_route_wave<kMiniRoute, detail::MiniMoeMath, moe::RouteWaveTraits<true, true>>(
            logits.data(), selection_bias.data(), route.ids.data(), route.weights.data(),
            route.status.data(), count, route_stream);
    };
    return mini_grouped::enqueue_resident<MiniMoeConfig, MiniHipMoeBackend>(
        physical, rows, input, router_logits, bias, output, buffers, matrices, csr, out, workspace, stream,
        mini_grouped::LogitExtent::exact, select);
}
#endif

}  // namespace strix::inkling2
