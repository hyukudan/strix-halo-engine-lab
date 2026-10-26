#pragma once

#include "strix/known_window_output_policy.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
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
    // Logical known inputs restored from a complete-state cache. They do not
    // contribute to steps_executed, which counts actual backend executions.
    std::uint64_t restored_prefix_tokens = 0;
};

struct SerialGreedyContinuationRequest {
    std::span<const float> initial_logits;
    std::uint64_t start_position = 0;
    std::uint64_t vocab_size = 0;
    std::uint64_t context_capacity = 0;
    std::uint64_t max_new_tokens = 0;
    std::uint32_t eos_token_id = 0;
};

struct GreedyPrefixRestore {
    std::uint64_t consumed_prefix = 0;
    // Present only for an exact full-prefix hit. This token has NOT been
    // consumed; a proper-prefix hit continues with the known request suffix.
    std::optional<std::uint32_t> pending_token;
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
    // Explicit same-session known-input capability. Zero preserves scalar calls.
    // Nonzero is a fixed admitted width2..2048; the scheduler never waits for
    // other requests. Only a complete known span is submitted. The backend must
    // retain its actual window/state/stream owners, complete and commit ALL rows
    // before return, and contain post-submit failures. A miss is not supported.
    [[nodiscard]] virtual std::uint32_t prefill_chunk_rows() const noexcept { return 0; }
    // Return a finite validated firstmax iff terminal_output is true. The final
    // prompt row still selects when max_new_tokens==0, matching scalar policy.
    [[nodiscard]] virtual std::optional<std::uint32_t> execute_prefill_chunk(
        std::span<const std::uint32_t>, std::uint64_t, std::uint64_t, bool) {
        throw std::invalid_argument("backend has no admitted known-input chunk owner");
    }
    // Optional complete-state prefix cache, queried once after response
    // reservation and before model execution. The concrete backend verifies
    // token/model/runtime/layout/history/owner identity before restoring all
    // required state. A miss must not mutate state. Post-submit exceptions
    // must poison/drain the backend's owned session before returning.
    [[nodiscard]] virtual std::optional<GreedyPrefixRestore> try_restore_greedy_prefix(
        const SerialGreedyRequest&) { return std::nullopt; }
    // Called once after a newly consumed complete prompt and before emission.
    // Capacity/admission misses are benign; transfer failures remain contained
    // by the native owner. No token is consumed by this cache operation.
    virtual void capture_greedy_prefix(std::span<const std::uint32_t>, std::uint32_t, std::uint64_t) {}
};

class SerialGreedyTokenCursor;
// An owner-bound single-input lease, not proof that any model work completed.
// A cohort executor returns a finite validated selection only for token output;
// state-only leases have no selection. Copying does not authorize a second use.
struct SerialGreedyTargetInput {
    std::uint32_t token = 0;
    std::uint64_t position = 0, vocab_size = 0, ticket = 0;
    bool state_only = false;
    bool operator==(const SerialGreedyTargetInput&) const = default;
private:
    const SerialGreedyTokenCursor* owner_ = nullptr;
    const SerialGreedyStepBackend* backend_ = nullptr;
    friend class SerialGreedyTokenCursor;
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
    // Split the SAME controller policy around one target call. Preparation
    // reserves response storage and performs the existing optional cache lookup.
    // An exact cache hit can emit its pending selection and return no input.
    // Only one lease may be outstanding; all access remains serialized.
    [[nodiscard]] std::optional<SerialGreedyTargetInput> prepare_target(SerialGreedyStepBackend&);
    // Call only AFTER the backend has completed/admitted/committed that input.
    // This validates the lease and updates cursor accounting, capture and output.
    // selected must be absent iff input.state_only. No state rollback is implied.
    void accept_target(SerialGreedyStepBackend&, const SerialGreedyTargetInput&,
        std::optional<std::uint32_t> selected);
    // Only the coordinator that has NOT submitted the input may abandon it.
    // After submission it must accept completed work or fail the target instead.
    void cancel_prepared_target(const SerialGreedyTargetInput&);
    void fail_target() noexcept { prepared_target_.reset(); failed_ = true; }

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
    std::optional<SerialGreedyTargetInput> prepared_target_;
    std::uint64_t next_target_ticket_ = 1;
    bool done_ = false, failed_ = false, cancelled_ = false, started_ = false;
};

// Experimental full-output prompt route. All inputs are already known. The
// compatibility entry is R2; explicitly wider backends override execute_rows.
// The backend commits one causal window and returns its terminal FP32 logits.
class KnownPromptWindowBackend {
public:
    virtual ~KnownPromptWindowBackend() = default;
    [[nodiscard]] virtual std::uint32_t logical_rows() const noexcept { return 2; }
    // True only for an explicitly compiled tiled prefill owner; never inferred
    // from row count or a generic execute_rows override.
    [[nodiscard]] virtual bool supports_tiled_prefill() const noexcept { return false; }
    [[nodiscard]] virtual KnownWindowOutputPolicy output_policy() const noexcept {
        return KnownWindowOutputPolicy::all_rows;
    }
    [[nodiscard]] virtual std::vector<float> execute_window(
        std::array<std::uint32_t, 2> tokens, std::uint64_t position) = 0;
    // Wider backends explicitly override both this entry and logical_rows().
    // The compatibility implementation never disguises repeated pairs as a
    // single larger native window.
    [[nodiscard]] virtual std::vector<float> execute_rows(
        std::span<const std::uint32_t> tokens, std::uint64_t position);
    virtual void execute_state_rows(std::span<const std::uint32_t>, std::uint64_t) {
        throw std::invalid_argument("known prompt backend does not support state-only windows");
    }
    // STRIX_QPF2K_SCHED bit 2: a state-only block that the caller follows with another block of this backend (which
    // completes it) or with finish_deferred(); default: the plain state-only block.
    virtual void execute_state_rows_deferred(std::span<const std::uint32_t> tokens, std::uint64_t position) {
        execute_state_rows(tokens, position);
    }
    virtual void finish_deferred() {}
    // STRIX_QPF2K_SCHED bit 4: a block of the first tokens.size() rows of a tiled backend (16..T rows, a multiple of 8).
    [[nodiscard]] virtual std::vector<float> execute_rows_partial(std::span<const std::uint32_t>, std::uint64_t) {
        throw std::invalid_argument("known prompt backend does not support partial blocks");
    }
    virtual void execute_state_rows_partial(std::span<const std::uint32_t>, std::uint64_t, bool) {
        throw std::invalid_argument("known prompt backend does not support partial blocks");
    }
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
