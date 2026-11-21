#pragma once

#include "gemma4/engine/coordinator.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>

namespace gemma4_test {

inline std::uint32_t reference_token(std::span<const float> logits,
                                     const strix::gemma4::engine::Sampling& policy,
                                     std::uint64_t position, std::uint64_t& draws) {
    if (policy.temperature == 0) {
        return std::uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    const strix::TokenSamplingPolicy parameters{
        policy.temperature, policy.top_k, policy.top_p, policy.min_p, policy.seed};
    strix::TokenSamplerOptions options;
    // The host fixtures use the reproducible request identity unless one is explicit.
    if (policy.request_identity) options.request_identity = policy.request_identity;
    strix::TokenSampler sampler(parameters, options);
    const auto token = sampler.decide(logits, position, nullptr).token;
    ++draws;
    return token;
}

} // namespace gemma4_test
