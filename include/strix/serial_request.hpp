#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

enum class SerialRequestStopReason {
    eos,
    max_new_tokens,
    context_capacity,
};

struct SerialGreedyRequest {
    std::span<const std::uint32_t> prefix_tokens;
    std::uint64_t vocab_size = 0;
    std::uint64_t context_capacity = 0;
    std::uint64_t max_new_tokens = 0;
    std::uint32_t eos_token_id = 0;
};

struct SerialGreedyRequestResult {
    std::vector<std::uint32_t> generated_tokens;
    std::uint64_t steps_executed = 0;
    std::uint64_t final_position = 0;
    SerialRequestStopReason stop_reason =
        SerialRequestStopReason::max_new_tokens;
};

class SerialGreedyStepBackend {
public:
    virtual ~SerialGreedyStepBackend() = default;

    [[nodiscard]] virtual std::vector<float> execute(
        std::uint32_t input_token,
        std::uint64_t position) = 0;
};

[[nodiscard]] std::uint32_t select_greedy_token(
    std::span<const float> logits,
    std::uint64_t vocab_size);

[[nodiscard]] SerialGreedyRequestResult execute_serial_greedy_request(
    const SerialGreedyRequest& request,
    SerialGreedyStepBackend& backend);

}  // namespace strix
