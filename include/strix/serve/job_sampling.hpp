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
#include <utility>

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

// Receipt of a finished sampled request: GenerationStats::sampling = "sampling-v1" and an engine_extra member
// "sampling_receipt" {"identity","acceptance","policy_identity","seed","drafts","accepted","rejected"}.
// A null sampler (temperature 0) leaves `stats` unchanged: the engine's greedy word stays.
inline void record_job_sampling(GenerationStats& stats, const TokenSampler* sampler) {
    if (sampler == nullptr) return;
    const auto& counts = sampler->stats();
    stats.sampling = "sampling-v1";
    std::string j = "{\"identity\":\"sampling-v1\",\"acceptance\":\"";
    j += sampler->coupled() ? "coupled" : "rejection";
    j += "\",\"policy_identity\":\"" + std::to_string(token_sampling_policy_identity(sampler->policy()));
    j += "\",\"seed\":" + std::to_string(sampler->policy().seed);
    j += ",\"drafts\":" + std::to_string(counts.drafts) + ",\"accepted\":" + std::to_string(counts.accepted);
    j += ",\"rejected\":" + std::to_string(counts.rejected) + "}";
    stats.engine_extra.emplace_back("sampling_receipt", std::move(j));
}

} // namespace strix::serve
