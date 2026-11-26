#pragma once

#include "core/bf16_reference.hpp"

namespace strix::moe_core {

// Single-row oracle for a routed latent branch and an ungated shared branch.
// The callbacks bind the arithmetic/rounding and accept input and output spans.
// The routed callback returns a weighted reduction in latent space.
template <class DownProject, class RoutedExperts, class UpProject, class SharedExpert>
void latent_experts_reference(std::span<const float> hidden, std::span<const float> latent_gain, float epsilon,
                              const DownProject& down, const RoutedExperts& routed,
                              const UpProject& up, const SharedExpert& shared, std::span<float> output) {
    if (hidden.empty() || output.size() != hidden.size() || latent_gain.empty() ||
        !std::isfinite(epsilon) || epsilon <= 0.0F)
        throw std::invalid_argument("latent experts: invalid row geometry or epsilon");
    std::vector<float> latent(latent_gain.size()), mixture(latent_gain.size()), shared_output(hidden.size());
    down(hidden, std::span(latent));
    core::rms_norm_bf16_reference(latent, latent_gain, epsilon);
    routed(std::span<const float>(latent), std::span(mixture));
    shared(hidden, std::span(shared_output));
    up(std::span<const float>(mixture), output);
    for (std::size_t i = 0; i < output.size(); ++i) output[i] += shared_output[i];
}

} // namespace strix::moe_core
