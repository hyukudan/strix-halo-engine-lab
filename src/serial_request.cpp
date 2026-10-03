#include "strix/serial_request.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix {

std::uint32_t select_greedy_token(
    std::span<const float> logits,
    std::uint64_t vocab_size) {
    if (vocab_size == 0 ||
        vocab_size >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) +
                1ULL) {
        throw std::invalid_argument(
            "greedy selection vocabulary size is invalid");
    }
    if (logits.size() !=
        static_cast<std::size_t>(
            vocab_size)) {
        throw std::invalid_argument(
            "greedy selection logits do not match vocabulary size");
    }

    std::uint32_t best = 0;
    float best_value = 0.0F;
    bool initialized = false;
    for (std::size_t index = 0;
         index < logits.size();
         ++index) {
        const float value = logits[index];
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "greedy selection logits must be finite");
        }
        if (!initialized ||
            value > best_value) {
            initialized = true;
            best_value = value;
            best =
                static_cast<std::uint32_t>(
                    index);
        }
    }

    if (!initialized) {
        throw std::invalid_argument(
            "greedy selection requires non-empty logits");
    }
    return best;
}

SerialGreedyRequestResult execute_serial_greedy_request(
    const SerialGreedyRequest& request,
    SerialGreedyStepBackend& backend) {
    if (request.prefix_tokens.empty()) {
        throw std::invalid_argument(
            "serial greedy request requires a non-empty prefix");
    }
    if (request.vocab_size == 0 ||
        request.vocab_size >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) +
                1ULL) {
        throw std::invalid_argument(
            "serial greedy request vocabulary size is invalid");
    }
    if (request.context_capacity == 0) {
        throw std::invalid_argument(
            "serial greedy request context capacity must be non-zero");
    }
    if (request.prefix_tokens.size() >
        request.context_capacity) {
        throw std::invalid_argument(
            "serial greedy request prefix exceeds context capacity");
    }
    if (static_cast<std::uint64_t>(
            request.eos_token_id) >=
        request.vocab_size) {
        throw std::invalid_argument(
            "serial greedy request EOS token is outside vocabulary");
    }
    for (const auto token :
         request.prefix_tokens) {
        if (static_cast<std::uint64_t>(
                token) >=
            request.vocab_size) {
            throw std::invalid_argument(
                "serial greedy request prefix token is outside vocabulary");
        }
    }

    SerialGreedyRequestResult result;
    std::vector<float> logits;
    std::uint64_t position = 0;

    for (const auto token :
         request.prefix_tokens) {
        logits =
            backend.execute(
                token,
                position);
        ++result.steps_executed;
        ++position;
    }

    result.final_position =
        position;

    if (request.max_new_tokens == 0) {
        result.stop_reason =
            SerialRequestStopReason::
                max_new_tokens;
        return result;
    }

    while (result.generated_tokens.size() <
           request.max_new_tokens) {
        const auto next =
            select_greedy_token(
                logits,
                request.vocab_size);
        result.generated_tokens.push_back(
            next);

        if (next ==
            request.eos_token_id) {
            result.stop_reason =
                SerialRequestStopReason::eos;
            result.final_position =
                position;
            return result;
        }

        if (position >=
            request.context_capacity) {
            result.stop_reason =
                SerialRequestStopReason::
                    context_capacity;
            result.final_position =
                position;
            return result;
        }

        if (result.generated_tokens.size() >=
            request.max_new_tokens) {
            result.stop_reason =
                SerialRequestStopReason::
                    max_new_tokens;
            result.final_position =
                position;
            return result;
        }

        logits =
            backend.execute(
                next,
                position);
        ++result.steps_executed;
        ++position;
        result.final_position =
            position;
    }

    result.stop_reason =
        SerialRequestStopReason::
            max_new_tokens;
    return result;
}

}  // namespace strix
