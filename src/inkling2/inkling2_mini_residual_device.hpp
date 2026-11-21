#pragma once

#include "core/residual_device.hpp"
#include "inkling2_mini_resident_weights.hpp"

#include <array>
#include <span>
#include <string_view>

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace strix::inkling2 {

inline core::residual_device::Error launch_mini_residual_read(
    const MiniResidentWeights& weights, std::span<const std::span<const float>> previous_blocks,
    std::span<const float> current, std::string_view query_name, std::span<const std::span<float>> outputs,
    core::residual_device::Stream stream) {
    namespace residual = core::residual_device;
    const auto rows = current.size() / kHiddenWidth;
    if (rows == 0U || rows > kMaxContext || current.size() % kHiddenWidth != 0U ||
        previous_blocks.size() >= residual::kMaxBanks || outputs.empty() || outputs.size() > residual::kMaxQueries) {
        return residual::invalid_value;
    }

    const auto queries = weights.f32(query_name);
    if (queries.size() != outputs.size() * kHiddenWidth) return residual::invalid_value;
    residual::MixArgs args{};
    args.queries = queries.data();
    args.rows = static_cast<unsigned>(rows);
    args.width = kHiddenWidth;
    args.bank_count = static_cast<unsigned>(previous_blocks.size() + 1U);
    args.query_count = static_cast<unsigned>(outputs.size());
    args.epsilon = 1e-6F;
    for (std::size_t i = 0; i < previous_blocks.size(); ++i) {
        if (previous_blocks[i].size() != current.size()) return residual::invalid_value;
        args.banks[i] = previous_blocks[i].data();
    }
    args.banks[previous_blocks.size()] = current.data();
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].size() != current.size()) return residual::invalid_value;
        args.output[i] = outputs[i].data();
    }
    return residual::launch_mix(args, stream);
}

inline core::residual_device::Error launch_mini_capture_message_bank(
    const std::array<std::span<const float>, 2>& inputs, const std::array<std::span<float>, 2>& normalized,
    core::residual_device::Stream stream) {
    namespace residual = core::residual_device;
    const auto width = kHiddenWidth;
    const auto rows = inputs[0].size() / width;
    if (rows == 0U || rows > kMaxContext || inputs[0].size() % width != 0U ||
        inputs[1].size() != inputs[0].size() || normalized[0].size() != inputs[0].size() ||
        normalized[1].size() != inputs[0].size() ||
        inputs[0].data() == normalized[1].data() || inputs[1].data() == normalized[0].data() ||
        normalized[0].data() == normalized[1].data()) return residual::invalid_value;
    const auto args = [&](std::size_t i) {
        return residual::NormalizeArgs{inputs[i].data(), normalized[i].data(), static_cast<unsigned>(rows), width, 1e-6F};
    };
    const auto first = args(0), second = args(1);
    if (!residual::fits(first) || !residual::fits(second)) return residual::invalid_value;
    const auto status = residual::launch_normalize(first, stream);
    return status == residual::success ? residual::launch_normalize(second, stream) : status;
}

inline core::residual_device::Error launch_mini_residual_add(std::span<const float> lhs, std::span<const float> rhs,
                                                            std::span<float> output, bool round_bf16,
                                                            core::residual_device::Stream stream) {
    namespace residual = core::residual_device;
    if (lhs.size() != rhs.size() || output.size() != lhs.size()) return residual::invalid_value;
    return residual::launch_add({lhs.data(), rhs.data(), output.data(), output.size(), round_bf16}, stream);
}

}  // namespace strix::inkling2
#endif
