#include "strix/spec/row_rule.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace strix::spec {

RowRule::RowRule(TokenSamplingPolicy policy, std::uint64_t request_identity)
    : policy_(policy), sampled_(token_sampling_enabled(policy)) {
    validate_token_sampling_policy(policy_);
    if (sampled()) rng_ = token_sampling_rng(policy_, request_identity);
}

std::uint32_t RowRule::row_token(std::span<const float> logits, std::uint64_t position) const {
    return draft_token(logits, {}, position);
}

std::uint32_t RowRule::draft_token(std::span<const float> logits, std::span<const std::uint32_t> ids,
                                   std::uint64_t position) const {
    if (sampled()) {
        return sample_token_row(logits, ids, policy_,
            speculation_uniform01(rng_, position, SpeculationRandomDrawClass::serial_target));
    }
    if (logits.empty() || logits.size() > std::numeric_limits<std::uint32_t>::max() ||
        (!ids.empty() && ids.size() != logits.size())) {
        throw std::invalid_argument("greedy row rule needs a non-empty row with matching token ids");
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best] ||
            (logits[i] == logits[best] && !ids.empty() && ids[i] < ids[best])) best = i;
    }
    return ids.empty() ? static_cast<std::uint32_t>(best) : ids[best];
}

std::uint64_t RowRule::identity() const {
    return sampled() ? speculation_rng_identity(rng_) : 0;
}

}  // namespace strix::spec
