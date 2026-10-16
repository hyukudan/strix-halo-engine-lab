#pragma once

#include "strix/distributed_control_wire.hpp"
#include "strix/serial_request.hpp"
#include <array>
#include <cstdint>
#include <optional>

namespace strix {
// Agreed immutable request metadata, not the prompt payload or a qualification
// permit. Both peers retain/verify the actual prompt against prompt_sha256.
struct DistributedGreedyRequestIdentity {
    std::uint32_t contract_version = 1;
    ModelContentIdentity model_content_identity, prompt_sha256;
    std::uint64_t serial_execution_identity = 0, kernel_plan_identity = 0, placement_plan_identity = 0;
    std::uint64_t request_id = 0, connection_epoch = 0;
    std::uint64_t initial_binding_identity = 0, initial_generation = 0, initial_history_identity = 0;
    std::uint64_t prompt_tokens = 0, vocab_size = 0, context_capacity = 0;
    std::uint64_t max_new_tokens = 0, max_target_inputs = 0;
    std::uint32_t eos_token_id = 0, boundary_dtype_id = 0;
    bool operator==(const DistributedGreedyRequestIdentity&) const = default;
};
void validate_distributed_greedy_request_identity(const DistributedGreedyRequestIdentity&);
[[nodiscard]] std::uint64_t distributed_greedy_request_identity(const DistributedGreedyRequestIdentity&) noexcept;

enum class DistributedGreedyMessageKind : std::uint32_t {
    input_command = 1, target_result = 2, finish = 3, finished = 4, failure = 5,
};
enum class DistributedGreedyOutputIntent : std::uint32_t { none = 0, state_only = 1, finite_firstmax = 2 };
enum class DistributedGreedyFinishReason : std::uint32_t {
    none = 0, eos = 1, max_new_tokens = 2, context_capacity = 3, cancelled = 4,
};
enum class DistributedGreedyFailure : std::uint32_t {
    none = 0, protocol = 1, native = 2, uncertain = 3, unavailable = 4,
};
[[nodiscard]] DistributedGreedyFinishReason distributed_greedy_finish_reason(SerialRequestStopReason);

// Fixed-size payload v1 inside outer frame v2/kind6, never kind5. All unused
// fields are canonical zero. There are no pointers, prompt arrays or full-logit
// arrays here. finite winner bits assert the terminal producer's checked result;
// a scalar receiver cannot independently verify firstmax or full-row finiteness.
struct DistributedGreedyMessage {
    std::uint32_t payload_version = 1;
    DistributedGreedyMessageKind kind = DistributedGreedyMessageKind::input_command;
    DistributedGreedyRequestIdentity request;
    std::uint64_t request_identity = 0, sequence = 0;
    std::uint32_t sender_node = 0, receiver_node = 1;
    std::uint64_t generation = 0, binding_identity = 0, decision_identity = 0;
    // Input position for command; committed next_position for result/finish.
    std::uint64_t position = 0, committed_tokens = 0, history_identity = 0;
    std::uint64_t previous_binding_identity = 0, previous_decision_identity = 0;
    std::uint32_t input_token = 0;
    DistributedGreedyOutputIntent output_intent = DistributedGreedyOutputIntent::none;
    bool selected_present = false;
    std::uint32_t selected_token = 0, selected_logit_bits = 0;
    std::uint64_t emitted_tokens = 0;
    bool pending_present = false;
    std::uint32_t pending_token = 0;
    DistributedGreedyFinishReason finish_reason = DistributedGreedyFinishReason::none;
    DistributedGreedyFailure failure = DistributedGreedyFailure::none;
};
void validate_distributed_greedy_message(const DistributedGreedyMessage&);
[[nodiscard]] std::vector<std::byte> encode_distributed_greedy_message(
    const DistributedGreedyMessage&, DistributedControlWireLimits = {});
[[nodiscard]] DistributedGreedyMessage decode_distributed_greedy_message(
    std::span<const std::byte>, DistributedControlWireLimits = {});
// Metadata comparison only. Caller MUST obtain commit from its own actual
// two-ACK participant and hold actual minted native retirement before any
// target reply or cursor acceptance. This function cannot create those proofs.
void validate_distributed_greedy_target_commit(const DistributedGreedyMessage&,
    const DistributedTokenBinding&, const DistributedTokenCommit&);

// Pure bounded transcript validator. It does not run a generation policy,
// execute a target, retire DMA, mutate a session or grant a commit. Failed or
// finished transcripts cannot be reused; no retries or duplicate acceptance.
class DistributedGreedyControlState final {
public:
    explicit DistributedGreedyControlState(DistributedGreedyRequestIdentity);
    [[nodiscard]] DistributedGreedyMessage make_message(DistributedGreedyMessageKind, std::uint32_t sender) const;
    void validate(const DistributedGreedyMessage&) const;
    void observe(const DistributedGreedyMessage&);
    void communication_failed() noexcept { failed_ = true; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] bool finished() const noexcept { return finished_; }
private:
    DistributedGreedyRequestIdentity request_;
    std::array<std::uint64_t, 2> next_{1, 1};
    std::optional<DistributedGreedyMessage> pending_, last_, finish_;
    std::uint64_t target_inputs_ = 0;
    bool failed_ = false, finished_ = false;
};
// A distinct, bounded protected channel. Legacy remote-step servers reject6.
// No stream ownership is invented: caller pins actual stream through this owner.
// Each returned result still needs actual native commit/retirement admission.
class FramedDistributedGreedyWire final {
public:
    FramedDistributedGreedyWire(DistributedControlByteStream&, DistributedGreedyRequestIdentity,
        std::uint32_t local_node, DistributedControlWireLimits = {});
    [[nodiscard]] DistributedGreedyMessage make_message(DistributedGreedyMessageKind) const;
    void send(const DistributedGreedyMessage&);
    [[nodiscard]] DistributedGreedyMessage receive();
    [[nodiscard]] bool failed() const noexcept { return state_.failed(); }
private:
    DistributedControlByteStream* stream_;
    DistributedGreedyControlState state_;
    std::uint32_t node_;
    DistributedControlWireLimits limits_;
};
} // namespace strix
