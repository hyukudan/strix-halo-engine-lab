#include "strix/serial_request.hpp"
#include "strix/serial_decode_replay.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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

SerialGreedyRequestResult execute_serial_greedy_continuation(
    const SerialGreedyContinuationRequest& request,
    SerialGreedyStepBackend& backend) {
    if (request.vocab_size == 0 ||
        request.vocab_size >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) +
                1ULL) {
        throw std::invalid_argument(
            "serial greedy continuation vocabulary size is invalid");
    }
    if (request.initial_logits.size() !=
        static_cast<std::size_t>(
            request.vocab_size)) {
        throw std::invalid_argument(
            "serial greedy continuation logits do not match vocabulary size");
    }
    if (request.context_capacity == 0 ||
        request.start_position >
            request.context_capacity) {
        throw std::invalid_argument(
            "serial greedy continuation position/capacity is invalid");
    }
    if (static_cast<std::uint64_t>(
            request.eos_token_id) >=
        request.vocab_size) {
        throw std::invalid_argument(
            "serial greedy continuation EOS token is outside vocabulary");
    }

    SerialGreedyRequestResult result;
    result.final_position =
        request.start_position;
    std::vector<float> logits(
        request.initial_logits.begin(),
        request.initial_logits.end());

    if (request.max_new_tokens == 0) {
        return result;
    }

    std::uint64_t position =
        request.start_position;
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
                SerialRequestStopReason::context_capacity;
            result.final_position =
                position;
            return result;
        }

        if (result.generated_tokens.size() >=
            request.max_new_tokens) {
            result.stop_reason =
                SerialRequestStopReason::max_new_tokens;
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

    return result;
}

namespace {
void validate_greedy_request(const SerialGreedyRequest& request) {
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

}
}  // namespace

SerialGreedyRequestResult execute_serial_greedy_request(
    const SerialGreedyRequest& request,
    SerialGreedyStepBackend& backend) {
    validate_greedy_request(request);

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

    SerialGreedyContinuationRequest continuation;
    continuation.initial_logits =
        logits;
    continuation.start_position =
        position;
    continuation.vocab_size =
        request.vocab_size;
    continuation.context_capacity =
        request.context_capacity;
    continuation.max_new_tokens =
        request.max_new_tokens;
    continuation.eos_token_id =
        request.eos_token_id;

    auto generated =
        execute_serial_greedy_continuation(
            continuation,
            backend);
    generated.steps_executed +=
        result.steps_executed;
    return generated;
}

KnownPromptGreedyResult execute_known_prompt_greedy_request(
    const KnownPromptGreedyRequest& request,
    KnownPromptWindowBackend& windows, SerialGreedyStepBackend& serial) {
    // Validate the entire suffix before either backend can consume state.
    validate_greedy_request({request.prompt_tokens, request.vocab_size,
        request.context_capacity, request.max_new_tokens, request.eos_token_id});
    if (request.start_position > request.context_capacity ||
        request.prompt_tokens.size() > request.context_capacity - request.start_position) {
        throw std::invalid_argument("known prompt suffix exceeds remaining context capacity");
    }

    KnownPromptGreedyResult result;
    std::vector<float> logits;
    std::size_t consumed = 0;
    auto position = request.start_position;
    while (request.prompt_tokens.size() - consumed >= 2U) {
        if (consumed != 0 && request.stop_between_windows && request.stop_between_windows(position)) {
            return result;
        }
        logits = windows.execute_window(
            {request.prompt_tokens[consumed], request.prompt_tokens[consumed + 1U]}, position);
        consumed += 2U;
        position += 2U;
        result.consumed_prompt_tokens += 2U;
        ++result.windows_executed;
    }
    if (consumed < request.prompt_tokens.size()) {
        logits = serial.execute(request.prompt_tokens[consumed], position);
        ++position;
        ++result.consumed_prompt_tokens;
        ++result.serial_prompt_steps;
    }

    SerialGreedyContinuationRequest continuation;
    continuation.initial_logits = logits;
    continuation.start_position = position;
    continuation.vocab_size = request.vocab_size;
    continuation.context_capacity = request.context_capacity;
    continuation.max_new_tokens = request.max_new_tokens;
    continuation.eos_token_id = request.eos_token_id;
    result.generation = execute_serial_greedy_continuation(continuation, serial);
    return result;
}

std::uint32_t SerialGreedyStepBackend::execute_greedy(
    std::uint32_t input_token, std::uint64_t position, std::uint64_t vocab_size) {
    return select_greedy_token(execute(input_token, position), vocab_size);
}

void SerialGreedyStepBackend::execute_state(std::uint32_t input_token, std::uint64_t position) {
    const auto logits = execute(input_token, position);
    // A generic full-output backend still produced these values. Validate its
    // finite contract even when the caller does not request a token from them.
    validate_serial_decode_engine_logits(logits, logits.size());
}

SerialGreedyRequestResult execute_serial_greedy_token_request(
    const SerialGreedyRequest& request, SerialGreedyStepBackend& backend) {
    SerialGreedyTokenCursor cursor(request);
    while (!cursor.done()) cursor.advance(backend, std::numeric_limits<std::uint64_t>::max());
    return cursor.take_result();
}

SerialGreedyTokenCursor::SerialGreedyTokenCursor(SerialGreedyRequest request) : request_(request) {
    validate_greedy_request(request_);
}

void SerialGreedyTokenCursor::emit_pending() {
    if (result_.generated_tokens.size() >= request_.max_new_tokens) { done_ = true; return; }
    result_.generated_tokens.push_back(pending_);
    if (pending_ == request_.eos_token_id) {
        result_.stop_reason = SerialRequestStopReason::eos;
        done_ = true;
    } else if (result_.final_position >= request_.context_capacity) {
        result_.stop_reason = SerialRequestStopReason::context_capacity;
        done_ = true;
    } else if (result_.generated_tokens.size() >= request_.max_new_tokens) done_ = true;
}

void SerialGreedyTokenCursor::advance(SerialGreedyStepBackend& backend, std::uint64_t prefill_rows) {
    if (failed_) throw std::logic_error("greedy token cursor cannot resume after backend failure");
    if (done_) return;
    if (!prefill_rows) throw std::invalid_argument("greedy prefill chunk must contain at least one row");
    const auto consume = [&](std::uint32_t token) {
        const auto selected = backend.execute_greedy(token, result_.final_position, request_.vocab_size);
        if (static_cast<std::uint64_t>(selected) >= request_.vocab_size)
            throw std::invalid_argument("greedy backend selected token outside vocabulary");
        return selected;
    };
    try {
        // Reserve the bounded response before the first backend can publish
        // state. Later token delivery cannot allocate after a committed step.
        if (result_.steps_executed == 0) {
            const auto emissions = std::min(request_.max_new_tokens,
                request_.context_capacity - request_.prefix_tokens.size() + 1U);
            if (emissions > result_.generated_tokens.max_size())
                throw std::length_error("greedy response exceeds host vector capacity");
            result_.generated_tokens.reserve(static_cast<std::size_t>(emissions));
        }
        if (prefilling()) {
            std::uint64_t consumed = 0;
            while (prefilling() && consumed < prefill_rows) {
                const auto token = request_.prefix_tokens[prefix_offset_];
                if (prefix_offset_ + 1U == request_.prefix_tokens.size()) pending_ = consume(token);
                else backend.execute_state(token, result_.final_position);
                ++prefix_offset_; ++consumed;
                ++result_.steps_executed; ++result_.final_position;
            }
            if (prefilling()) return;
        } else {
            pending_ = consume(pending_);
            ++result_.steps_executed; ++result_.final_position;
        }
        emit_pending();
    } catch (...) {
        failed_ = true;
        throw;
    }
}

SerialGreedyRequestResult SerialGreedyTokenCursor::take_result() {
    if (!done_ || failed_) throw std::logic_error("greedy token cursor result is incomplete or failed");
    return std::move(result_);
}

}  // namespace strix
