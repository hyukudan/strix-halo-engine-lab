#pragma once

// One translation from a served request's sampling fields (GenerationJob::sampling) to the shared sampler
// (strix/token_sampling.hpp, identity sampling-v1). Every serving engine binds its final logits through this:
// temperature 0 creates nothing (the engine's greedy route runs unchanged); temperature > 0 creates one sampler per
// request whose uniforms are keyed by (seed, request identity, policy identity, absolute position), so a seed gives
// the same tokens however the request is batched.

#include "strix/serve/engine.hpp"
#include "strix/token_sampling.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace strix::serve {

[[nodiscard]] inline TokenSamplingPolicy job_sampling_policy(const SamplingParameters& s) noexcept {
    TokenSamplingPolicy policy;
    policy.temperature = static_cast<float>(s.temperature);
    policy.top_k = s.top_k;
    policy.top_p = static_cast<float>(s.top_p);
    policy.min_p = static_cast<float>(s.min_p);
    policy.seed = s.seed;
    return policy;
}

// Null at temperature 0. Otherwise a validated sampler with a fresh request identity
// (next_token_sampling_request_identity); throws std::invalid_argument when the policy is refused.
// `acceptance` decides drafted rows only (see SampledAcceptance); a route without drafts ignores it.
[[nodiscard]] inline std::shared_ptr<TokenSampler> make_job_sampler(const SamplingParameters& s,
        SampledAcceptance acceptance = SampledAcceptance::rejection) {
    if (s.temperature == 0.0) return {};
    const auto policy = job_sampling_policy(s);
    validate_token_sampling_policy(policy);
    if (!token_sampling_enabled(policy)) throw std::invalid_argument("temperature is too small to sample with");
    TokenSamplerOptions options;
    options.acceptance = acceptance;
    options.request_identity = next_token_sampling_request_identity();
    return std::make_shared<TokenSampler>(policy, options);
}

// The same for a request that must fail alone: error receives "sampling policy refused: ..." and the result is null.
[[nodiscard]] inline std::shared_ptr<TokenSampler> make_job_sampler(const SamplingParameters& s, std::string& error,
        SampledAcceptance acceptance = SampledAcceptance::rejection) {
    error.clear();
    try {
        return make_job_sampler(s, acceptance);
    } catch (const std::invalid_argument& e) {
        error = std::string("sampling policy refused: ") + e.what();
        return {};
    }
}

} // namespace strix::serve
