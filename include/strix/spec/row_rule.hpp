#pragma once
// Home: shared

#include <cstdint>
#include <span>

#include "strix/token_sampling.hpp"

namespace strix::spec {

// Both decisions use the serial-target draw at the zero-based position of the predicted token.
class RowRule {
public:
    RowRule() = default;
    RowRule(TokenSamplingPolicy policy, std::uint64_t request_identity);

    [[nodiscard]] bool sampled() const { return token_sampling_enabled(policy_); }
    [[nodiscard]] std::uint32_t row_token(std::span<const float> logits, std::uint64_t position) const;
    [[nodiscard]] std::uint32_t draft_token(std::span<const float> logits, std::span<const std::uint32_t> ids,
                                            std::uint64_t position) const;
    // Zero for greedy; otherwise identifies the seed, request and sampling transform.
    [[nodiscard]] std::uint64_t identity() const;

private:
    TokenSamplingPolicy policy_{};
    SpeculationRngIdentity rng_{};
};

}  // namespace strix::spec
