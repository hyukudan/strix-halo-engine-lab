#pragma once

#include "inkling2_mini_moe.hpp"
#include "inkling2_mini_moe_dispatch.hpp"

#include "strix/env_flags.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#include "strix/cuda/core/moe_route_tap.hpp"
#endif

namespace strix::inkling2 {

using MiniCudaMoeBackend = mini_grouped::CudaBackend;

#if defined(__CUDACC__) && !defined(__HIPCC__)
// Expert weight decode of the grouped tile (bitwise either way; RTX 4K prefill 730 -> 706 ms, expert tile 302 -> 281 ms).
// Default: the exponent-shift NVFP4 decode; STRIX_INK_MOE_SHIFT_DECODE=0 restores the reference decode.
inline moe_core::grouped_microscaled::Decode mini_cuda_moe_decode() {
    return env_flag_default_on("STRIX_INK_MOE_SHIFT_DECODE", EnvFlagMatch::first_character)
               ? moe_core::grouped_microscaled::Decode::shift : moe_core::grouped_microscaled::Decode::reference;
}

// Schedule of the grouped expert tile (bitwise either way: moe_core/grouped_tall_plan.hpp). Default: automatic (the tall
// slab schedule where it fits: 128-row CTAs, one weight decode per CTA and stage, cp.async ring, one barrier per stage);
// STRIX_INK_MOE_TALL=0 restores the staged K16 tile.
inline moe_core::grouped_microscaled::Schedule mini_cuda_moe_schedule() {
    return env_flag_default_on("STRIX_INK_MOE_TALL", EnvFlagMatch::first_character)
               ? moe_core::grouped_microscaled::Schedule::automatic : moe_core::grouped_microscaled::Schedule::staged;
}

inline constexpr auto enqueue_mini_cuda_router_logits = mini_grouped::cuda_router_logits<MiniMoeConfig>;

inline cudaError_t enqueue_mini_cuda_resident_moe_decoded(
    moe_core::grouped_microscaled::Decode decode,
    unsigned physical_layer, unsigned rows, std::span<const float> input, std::span<const float> logits,
    std::span<const float> selection_bias, std::span<float> output, MiniMoeResidentRouteBuffers route,
    std::span<const MiniMoeExpert> matrices, moe_core::device_grouped::CsrTables csr,
    moe_core::device_grouped::Output planned, MiniMoeWorkspace workspace, cudaStream_t stream,
    moe::RouteTap* tap = nullptr,
    moe_core::grouped_w4a4::Activations activations = moe_core::grouped_w4a4::Activations::bf16,
    moe_core::grouped_microscaled::Schedule schedule = moe_core::grouped_microscaled::Schedule::automatic) {
    const auto select = [physical_layer, tap](std::span<const float> scores, std::span<const float> bias,
                           MiniMoeResidentRouteBuffers buffers, unsigned count, cudaStream_t route_stream) {
        const auto status = moe_core::enqueue_route_wave<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true>>(
            scores.data(), bias.data(), buffers.ids.data(), buffers.weights.data(),
            buffers.status.data(), count, route_stream);
        if (status == cudaSuccess)
            cuda::core::route_tap(physical_layer, count, kExpertsPerToken, kExperts, buffers.ids.data(),
                                  buffers.weights.data(), route_stream, tap);
        return status;
    };
    return mini_grouped::enqueue_resident<MiniMoeConfig, MiniCudaMoeBackend>(
        physical_layer, rows, input, logits, selection_bias, output, route, matrices, csr, planned, workspace, stream,
        mini_grouped::LogitExtent::at_least, select, decode, activations, schedule);
}

// Reference decode (the fixtures bind this signature as a function pointer).
inline cudaError_t enqueue_mini_cuda_resident_moe(
    unsigned physical_layer, unsigned rows, std::span<const float> input, std::span<const float> logits,
    std::span<const float> selection_bias, std::span<float> output, MiniMoeResidentRouteBuffers route,
    std::span<const MiniMoeExpert> matrices, moe_core::device_grouped::CsrTables csr,
    moe_core::device_grouped::Output planned, MiniMoeWorkspace workspace, cudaStream_t stream,
    moe::RouteTap* tap = nullptr) {
    return enqueue_mini_cuda_resident_moe_decoded(moe_core::grouped_microscaled::Decode::reference, physical_layer,
                                                  rows, input, logits, selection_bias, output, route, matrices, csr,
                                                  planned, workspace, stream, tap);
}
#endif

}  // namespace strix::inkling2
