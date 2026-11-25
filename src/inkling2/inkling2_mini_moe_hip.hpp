#pragma once

#include "inkling2_mini_expert_hip.hpp"
#include "inkling2_mini_moe.hpp"

#if defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#include "moe_core/route_tap_hip.hpp"
#endif

namespace strix::inkling2 {

#if defined(__HIPCC__)
// Config = MiniMoeNarrowConfig dispatches a narrow expert image (every layer NVFP4); the checkpoint entry below is unchanged.
template <class Config>
inline hipError_t enqueue_mini_hip_resident_moe_for(
    unsigned physical, unsigned rows, std::span<const float> input,
    std::span<const float> router_logits, std::span<const float> bias, std::span<float> output,
    MiniMoeResidentRouteBuffers buffers, std::span<const MiniMoeExpert> matrices,
    moe_core::device_grouped::CsrTables csr, moe_core::device_grouped::Output out,
    MiniMoeWorkspace workspace, hipStream_t stream, moe::RouteTap* tap = nullptr,
    moe_core::grouped_iu4a4::Selection iu4 = {}) {
    const auto select = [physical, tap](std::span<const float> logits, std::span<const float> selection_bias,
                           MiniMoeResidentRouteBuffers route, unsigned count, hipStream_t route_stream) {
        const auto status = moe_core::enqueue_route_wave<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true, true>>(
            logits.data(), selection_bias.data(), route.ids.data(), route.weights.data(),
            route.status.data(), count, route_stream);
        if (status == hipSuccess)
            moe_core::route_tap(physical, count, kExpertsPerToken, kExperts, route.ids.data(),
                                route.weights.data(), route_stream, tap);
        return status;
    };
    return mini_grouped::enqueue_resident<Config, MiniHipMoeBackend>(
        physical, rows, input, router_logits, bias, output, buffers, matrices, csr, out, workspace, stream,
        mini_grouped::LogitExtent::exact, select, moe_core::grouped_microscaled::Decode::reference,
        moe_core::grouped_w4a4::Activations::bf16, moe_core::grouped_microscaled::Schedule::automatic, iu4);
}

inline hipError_t enqueue_mini_hip_resident_moe(
    unsigned physical, unsigned rows, std::span<const float> input,
    std::span<const float> router_logits, std::span<const float> bias, std::span<float> output,
    MiniMoeResidentRouteBuffers buffers, std::span<const MiniMoeExpert> matrices,
    moe_core::device_grouped::CsrTables csr, moe_core::device_grouped::Output out,
    MiniMoeWorkspace workspace, hipStream_t stream, moe::RouteTap* tap = nullptr) {
    return enqueue_mini_hip_resident_moe_for<MiniMoeConfig>(physical, rows, input, router_logits, bias, output, buffers,
                                                            matrices, csr, out, workspace, stream, tap);
}
#endif

}  // namespace strix::inkling2
