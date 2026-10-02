#pragma once

#include "strix/model_runtime_binding.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct SerialDecodeReplayPolicy {
    ModelContentIdentity expected_model_content_identity;
    std::uint64_t expected_compiled_execution_identity = 0;
    std::uint64_t expected_runtime_compatibility_identity = 0;
    std::uint64_t policy_identity = 0;
    double max_logit_abs_error = 0.0;
    std::uint64_t minimum_steps = 2;
    std::uint64_t minimum_logit_values_compared = 1;
    bool require_exact_argmax = true;
};

struct SerialDecodeReplayObservation {
    ModelContentIdentity model_content_identity;
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t independent_reference_identity = 0;

    std::vector<std::uint32_t> input_tokens;
    std::vector<std::uint32_t> reference_argmax_tokens;
    std::vector<std::uint32_t> engine_argmax_tokens;

    std::uint64_t logit_values_compared = 0;
    double maximum_logit_abs_error = 0.0;
};

struct SerialDecodeReplayAssessment {
    bool identity_valid = false;
    bool multi_step = false;
    bool input_sequence_present = false;
    bool argmax_checked = false;
    bool argmax_match = false;
    bool logits_checked = false;
    bool logits_within_policy = false;
    bool passes_declared_policy = false;
};

void validate_serial_decode_engine_logits(
    std::span<const float> logits,
    std::uint64_t expected_vocab_size);

[[nodiscard]] SerialDecodeReplayAssessment
assess_serial_decode_replay(
    const SerialDecodeReplayPolicy& policy,
    const SerialDecodeReplayObservation& observation);

}  // namespace strix
