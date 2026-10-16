#pragma once

#include <cstdint>
#include <limits>
#include <span>

namespace strix {

enum class MtpTargetWindowDecision {
    accept_complete_chunk,
    insufficient_output_budget,
    target_input_mismatch,
    eos_would_be_consumed,
};

struct MtpTargetWindowAssessment {
    MtpTargetWindowDecision decision = MtpTargetWindowDecision::insufficient_output_budget;
    // UINT32_MAX unless decision is target_input_mismatch; row zero is an
    // invalid caller binding rather than a speculative mismatch decision.
    std::uint32_t mismatch_input_row = std::numeric_limits<std::uint32_t>::max();
    [[nodiscard]] bool accepts_complete_chunk() const noexcept {
        return decision == MtpTargetWindowDecision::accept_complete_chunk;
    }
};

// Pure policy over already observed finite-head token selections. Does not
// authenticate their producer, mutate state or mint completion/commit proofs.
// All metadata is validated first. Decision precedence is insufficient budget,
// earliest causal input mismatch, then consumed EOS. A final prediction may be
// EOS because that prediction has not been consumed by this chunk.
[[nodiscard]] MtpTargetWindowAssessment assess_mtp_target_window(
    std::span<const std::uint32_t> inputs,
    std::span<const std::uint32_t> actual_selected_predictions,
    std::uint32_t actual_pending,
    std::uint32_t output_budget,
    std::span<const std::uint32_t> eos,
    std::uint64_t vocabulary);

} // namespace strix
