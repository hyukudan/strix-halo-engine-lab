#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
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

struct SerialGreedyContinuationRequest {
    std::span<const float> initial_logits;
    std::uint64_t start_position = 0;
    std::uint64_t vocab_size = 0;
    std::uint64_t context_capacity = 0;
    std::uint64_t max_new_tokens = 0;
    std::uint32_t eos_token_id = 0;
};

class SerialGreedyStepBackend {
public:
    virtual ~SerialGreedyStepBackend() = default;

    [[nodiscard]] virtual std::vector<float> execute(
        std::uint32_t input_token,
        std::uint64_t position) = 0;

    // Scalar generation seam. The default retains the complete-logits backend;
    // a device backend may validate/select without downloading the vocabulary.
    [[nodiscard]] virtual std::uint32_t execute_greedy(
        std::uint32_t input_token, std::uint64_t position,
        std::uint64_t vocab_size);

    // Prefix positions with no requested output still consume all model state.
    // The default preserves the complete existing execution.
    virtual void execute_state(std::uint32_t input_token, std::uint64_t position);
};

// Incremental form of the token-output request controller. The prefix span is
// borrowed and must stay immutable until completion. Each advance consumes at
// most prefill_rows known inputs, or one pending generated input. The last
// emitted token stays unconsumed at EOS/budget/capacity/cancellation boundaries.
// Backend calls must contain their own completion and post-submit failures.
class SerialGreedyTokenCursor final {
public:
    explicit SerialGreedyTokenCursor(SerialGreedyRequest request);
    SerialGreedyTokenCursor(const SerialGreedyTokenCursor&) = delete;
    SerialGreedyTokenCursor& operator=(const SerialGreedyTokenCursor&) = delete;
    void advance(SerialGreedyStepBackend&, std::uint64_t prefill_rows = 1);
    void cancel() noexcept { if (!done_) { cancelled_ = true; done_ = true; } }
    [[nodiscard]] bool done() const noexcept { return done_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] bool cancelled() const noexcept { return cancelled_; }
    [[nodiscard]] bool prefilling() const noexcept { return prefix_offset_ < request_.prefix_tokens.size(); }
    [[nodiscard]] const SerialGreedyRequestResult& result() const noexcept { return result_; }
    [[nodiscard]] SerialGreedyRequestResult take_result();
private:
    void emit_pending();
    SerialGreedyRequest request_;
    SerialGreedyRequestResult result_;
    std::size_t prefix_offset_ = 0;
    std::uint32_t pending_ = 0;
    bool done_ = false, failed_ = false, cancelled_ = false;
};

// Experimental full-output prompt route. Both inputs are already known; this
// interface does not predict a second autoregressive token. The backend commits
// one causal window and returns the complete FP32 logits of its terminal row.
class KnownPromptWindowBackend {
public:
    virtual ~KnownPromptWindowBackend() = default;
    [[nodiscard]] virtual std::vector<float> execute_window(
        std::array<std::uint32_t, 2> tokens, std::uint64_t position) = 0;
};

struct KnownPromptGreedyRequest {
    // A non-empty known suffix, immutable for this call. An already consumed
    // prefix is represented by start_position, never by re-executing its token.
    std::span<const std::uint32_t> prompt_tokens;
    std::uint64_t start_position = 0;
    std::uint64_t vocab_size = 0;
    std::uint64_t context_capacity = 0;
    std::uint64_t max_new_tokens = 0;
    std::uint32_t eos_token_id = 0;
    // Consulted before another pair only after a completed window in this call.
    // It is not called before the first pair, the odd tail, or generation.
    std::function<bool(std::uint64_t next_position)> stop_between_windows;
};

struct KnownPromptGreedyResult {
    std::uint64_t consumed_prompt_tokens = 0;
    std::uint64_t windows_executed = 0;
    std::uint64_t serial_prompt_steps = 0;
    // Empty only when stopped between windows. Continuation retains the usual
    // last-emitted-token/unconsumed frontier and counts generation steps only.
    std::optional<SerialGreedyRequestResult> generation;
};

[[nodiscard]] KnownPromptGreedyResult execute_known_prompt_greedy_request(
    const KnownPromptGreedyRequest& request,
    KnownPromptWindowBackend& windows, SerialGreedyStepBackend& serial);

[[nodiscard]] std::uint32_t select_greedy_token(
    std::span<const float> logits,
    std::uint64_t vocab_size);

[[nodiscard]] SerialGreedyRequestResult execute_serial_greedy_continuation(
    const SerialGreedyContinuationRequest& request,
    SerialGreedyStepBackend& backend);

[[nodiscard]] SerialGreedyRequestResult execute_serial_greedy_request(
    const SerialGreedyRequest& request,
    SerialGreedyStepBackend& backend);

// Explicit token-output route; existing full-logits request/continuation APIs
// retain their behavior. Every scalar result must have passed finite validation.
[[nodiscard]] SerialGreedyRequestResult execute_serial_greedy_token_request(
    const SerialGreedyRequest& request,
    SerialGreedyStepBackend& backend);

}  // namespace strix
