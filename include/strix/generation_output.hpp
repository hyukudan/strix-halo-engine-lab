#pragma once

#include "strix/serial_request.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace strix {

// Needs may coexist: e.g. a visible token, requested logprobs and RAW logits.
// token_only identifies the small-result need; it does not prohibit an explicit
// full_logits/logprobs need in the same request. state_only instead explicitly
// prohibits head outputs and cannot be combined with any other need.
enum class GenerationOutputNeed : std::uint32_t {
    token_only = 1U << 0U,
    requested_logprobs = 1U << 1U,
    full_logits = 1U << 2U,
    verify_decisions = 1U << 3U,
    state_only = 1U << 4U,
    mtp_hidden = 1U << 5U,
};
[[nodiscard]] constexpr GenerationOutputNeed operator|(GenerationOutputNeed a, GenerationOutputNeed b) noexcept {
    return static_cast<GenerationOutputNeed>(static_cast<std::uint32_t>(a)|static_cast<std::uint32_t>(b));
}

struct GenerationOutputRequest {
    GenerationOutputNeed needs = GenerationOutputNeed::full_logits;
    std::uint32_t input_token = 0;
    std::uint64_t position = 0;
    std::uint64_t vocab_size = 0;
    // Caller order and duplicates are retained. These are presentation queries,
    // NOT a restricted sampling support or a list of speculative candidates.
    std::span<const std::uint32_t> logprob_token_ids;
};
struct RequestedLogprob {
    std::uint32_t token_id = 0;
    // Stable FP64 full-vocabulary log-softmax, with ascending-ID denominator
    // accumulation. This changes no model projection or generation policy.
    double log_probability = 0;
};
struct GenerationOutputResult {
    GenerationOutputNeed needs = GenerationOutputNeed::full_logits;
    std::uint64_t position = 0;
    std::optional<std::uint32_t> token;
    // One target-row greedy selection, consumed by the existing verification
    // controller. No prefix acceptance/rejection or bonus policy is invented.
    std::optional<std::uint32_t> verification_selection;
    std::vector<RequestedLogprob> requested_logprobs;
    std::vector<float> logits;
};

// Validates every need/ID before entering the backend. Exactly one model call:
// execute_state for state_only; execute_greedy for selection-only requests;
// execute for any full-logits/logprobs request. Full-vocabulary finite checking
// remains mandatory even when only a requested subset is returned.
//
// The backend owns producer completion, state transaction and failures. Native
// backends must contain post-submit errors and retain/drain their owners; this
// portable adapter cannot restore a backend's opaque mutable state. No sampling,
// MTP or new EOS/capacity/acceptance policy is supplied. Existing request APIs and
// their default full-output behavior are unchanged.
[[nodiscard]] GenerationOutputResult execute_generation_output(
    const GenerationOutputRequest& request, SerialGreedyStepBackend& backend);

} // namespace strix
