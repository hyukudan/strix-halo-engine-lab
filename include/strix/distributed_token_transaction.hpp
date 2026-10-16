#pragma once
#include "strix/distributed_placement.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/model_runtime_binding.hpp"
#include <array>
#include <cstdint>
#include <optional>

namespace strix {
struct DistributedTokenPeer {
    std::uint32_t node = 0, first_step = 0, last_step = 0;
    std::uint64_t runtime_compatibility_identity = 0, state_layout_identity = 0;
    std::uint64_t context_capacity = 0;
    PersistentStateGenerations initial_state;
};
// Explicit known input and complete logical frontier. This is not an ordinary
// SerialSessionContext: each peer owns only its stage's logical state slices.
struct DistributedTokenBinding {
    std::uint32_t contract_version = 1;
    std::uint64_t serial_execution_identity = 0, kernel_plan_identity = 0, placement_plan_identity = 0;
    ModelContentIdentity model_content_identity;
    std::uint64_t request_id = 0, connection_epoch = 0, generation = 0;
    std::uint64_t position = 0, committed_tokens = 0, previous_history_identity = 0;
    std::uint32_t input_token = 0;
    // Existing transport descriptor dtype negotiated by the controller. It is
    // bound explicitly, never inferred from an epoch or assigned by the owner.
    std::uint32_t boundary_dtype_id = 0;
    PersistentStateGenerations initial_global_state;
    std::array<DistributedTokenPeer,2> peers;
};
[[nodiscard]] std::uint64_t distributed_token_binding_identity(const DistributedTokenBinding&) noexcept;
struct DistributedTokenReceipt {
    std::uint64_t snapshot_plan_identity = 0, snapshot_generation = 0;
    PersistentStateGenerations local_state;
    std::uint32_t completed_steps = 0;
};
enum class DistributedTokenMessageKind : std::uint32_t {
    prepared = 1, ready = 2, commit_decision = 3, committed = 4,
    abort_decision = 5, restored = 6,
};
struct DistributedTokenMessage {
    DistributedTokenBinding binding;
    std::uint64_t binding_identity = 0, sequence = 0, decision_identity = 0;
    std::uint32_t sender_node = 0;
    DistributedTokenMessageKind kind = DistributedTokenMessageKind::prepared;
    DistributedTokenReceipt receipt;
};
enum class DistributedTokenPhase : std::uint32_t {
    preparing, executing, commit_decided, committed, abort_decided, rolled_back, in_doubt,
};
struct DistributedTokenCommit {
    std::uint64_t binding_identity = 0, decision_identity = 0;
    std::uint64_t next_position = 0, committed_tokens = 0, history_identity = 0;
    PersistentStateGenerations global_state;
    std::array<PersistentStateGenerations,2> local_state;
};
// One transaction only; no durable-log or crash recovery claim. A local commit
// decision is irreversible even when its send/ACK fails. Unknown outcomes retain
// snapshots and disable reuse; never restore one peer after deciding commit.
class DistributedTokenTransaction final {
public:
    DistributedTokenTransaction(const CompiledSerialExecution&,const DistributedPlacementPlan&,
        const ModelManifest&,const ExecutionTensorAccessPlan&,DistributedTokenBinding);
    [[nodiscard]] const DistributedTokenBinding& binding() const noexcept {return binding_;}
    [[nodiscard]] std::uint64_t binding_identity() const noexcept {return identity_;}
    [[nodiscard]] DistributedTokenPhase phase() const noexcept {return phase_;}
    // Remains nonzero after a lost decision/ACK moves the phase to in_doubt.
    [[nodiscard]] std::uint64_t decision_identity() const noexcept {return decision_;}
    [[nodiscard]] bool both_prepared() const noexcept {return prepared_[0] && prepared_[1];}
    [[nodiscard]] bool both_ready() const noexcept {return ready_[0] && ready_[1];}
    [[nodiscard]] std::uint64_t next_sequence(std::uint32_t node) const;
    [[nodiscard]] std::uint64_t expected_commit_decision() const;
    [[nodiscard]] PersistentStateGenerations expected_local_state(std::uint32_t node) const;
    // Strict exactly-once phase/sequence/binding admission. Invalid wire input
    // enters in_doubt, including before a decision; it never authorizes reuse.
    void observe(const DistributedTokenMessage&);
    void transport_failed() noexcept {phase_=DistributedTokenPhase::in_doubt;}
    [[nodiscard]] std::optional<DistributedTokenCommit> committed_frontier() const;
private:
    DistributedTokenBinding binding_;
    std::uint64_t identity_=0,decision_=0;
    DistributedTokenPhase phase_=DistributedTokenPhase::preparing;
    std::array<bool,2> prepared_{},ready_{},acknowledged_{},restored_{};
    std::array<std::uint64_t,2> sequence_{1,1};
    std::array<DistributedTokenReceipt,2> receipt_{};
    std::array<PersistentStateGenerations,2> expected_local_{};
    PersistentStateGenerations expected_global_;
};
} // namespace strix
