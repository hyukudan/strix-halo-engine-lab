#pragma once

#include "inkling2_mini_expert_hip.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "moe_core/grouped_rows.hpp"
#include "moe_core/expert_activation.hpp"
#if defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#endif

namespace strix::inkling2 {

struct MiniHipMoeBackend {
    static constexpr unsigned rows_per_tile = moe_core::grouped_microscaled::gemm::kBM;
    static constexpr auto fits = moe_core::grouped_microscaled::fits;
    static constexpr auto fits_tile = fits_mini_hip_expert_tile;

#if defined(__HIPCC__)
    using Status = hipError_t;
    using Stream = hipStream_t;
    static constexpr auto success = hipSuccess;
    static constexpr auto invalid_value = hipErrorInvalidValue;
    static constexpr auto gather = moe_core::grouped_rows::gather;
    static constexpr auto combine = moe_core::grouped_rows::combine;
    static constexpr auto activate = moe_core::expert_activation::launch;
    static constexpr auto project = enqueue_mini_hip_expert_projection;
#endif
};

#if defined(__HIPCC__)
// Builds routes and grouped operands entirely on the stream from prepared FP32
// router logits. On completion check plan output failure before using the result.
inline hipError_t enqueue_mini_hip_resident_moe(
    unsigned physical, unsigned rows, std::span<const float> input,
    std::span<const float> router_logits, std::span<const float> bias, std::span<float> output,
    MiniMoeResidentRouteBuffers buffers, std::span<const MiniMoeExpert> matrices,
    moe_core::device_grouped::CsrTables csr, moe_core::device_grouped::Output out,
    MiniMoeWorkspace workspace, hipStream_t stream) {
    namespace plan = moe_core::device_grouped;
    using Planner = plan::Planner<kExperts, kExpertsPerToken>;
    plan::Args args{};
    if (!prepare_mini_resident_moe<MiniHipMoeBackend>(physical, rows, buffers, matrices, csr, out, workspace, args) ||
        router_logits.size() != std::size_t(rows) * kExperts || bias.size() != kExperts ||
        input.size() < std::size_t(rows) * kHiddenWidth || output.size() < std::size_t(rows) * kHiddenWidth ||
        buffers.scratch.size() < Planner::scratch_bytes(rows) / sizeof(unsigned)) return hipErrorInvalidValue;
    auto status = moe_core::enqueue_route_wave<kMiniRoute, detail::MiniMoeMath, moe::RouteWaveTraits<true, true>>(
        router_logits.data(), bias.data(), buffers.ids.data(), buffers.weights.data(), buffers.status.data(), rows, stream);
    if (status != hipSuccess) return status;
    status = plan::enqueue<kExperts, kExpertsPerToken>(args, buffers.scratch, stream);
    if (status != hipSuccess) return status;
    return enqueue_planned_mini_moe<MiniHipMoeBackend>(physical, args, input, output, workspace, stream);
}
#endif

}  // namespace strix::inkling2
