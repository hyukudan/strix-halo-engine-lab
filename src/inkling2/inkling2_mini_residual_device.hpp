#pragma once

#include "core/residual_mix.hpp"
#include "inkling2_mini_resident_weights.hpp"

#include <array>
#include <span>
#include <string_view>

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace strix::inkling2 {

inline core::residual_mix::Error launch_mini_residual_read(
    const MiniResidentWeights& weights, std::span<const std::span<const float>> previous_blocks,
    std::span<const float> current, std::string_view query_name, std::span<const std::span<float>> outputs,
    core::residual_mix::Stream stream) {
    namespace residual = core::residual_mix;
    return residual::launch_residual_read(previous_blocks, current, weights.f32(query_name), outputs,
                                          kHiddenWidth, kMaxContext, 1e-6F, stream);
}

inline core::residual_mix::Error launch_mini_capture_message_bank(
    const std::array<std::span<const float>, 2>& inputs, const std::array<std::span<float>, 2>& normalized,
    core::residual_mix::Stream stream) {
    namespace residual = core::residual_mix;
    return residual::launch_capture_message_bank(inputs, normalized, kHiddenWidth, kMaxContext, 1e-6F, stream);
}

inline core::residual_mix::Error launch_mini_residual_add(std::span<const float> lhs, std::span<const float> rhs,
                                                            std::span<float> output, bool round_bf16,
                                                            core::residual_mix::Stream stream) {
    namespace residual = core::residual_mix;
    if (lhs.size() != rhs.size() || output.size() != lhs.size()) return residual::invalid_value;
    return residual::launch_add({lhs.data(), rhs.data(), output.data(), output.size(), round_bf16}, stream);
}

}  // namespace strix::inkling2
#endif
