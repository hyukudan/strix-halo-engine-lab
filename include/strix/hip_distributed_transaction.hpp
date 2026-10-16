#pragma once
#include "strix/distributed_control_wire.hpp"
#include "strix/hip_distributed_stage.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/serial_session.hpp"
#include <memory>

namespace strix {
// One explicitly bound two-peer token transaction, not a distributed decoder or
// a persistent crash-recovery service. Real stage execution and all seven local
// snapshot transfers are owned here. The caller retains allocator/generations,
// original session, resources, stream and model through actual_owner.
// Single serialized progress owner; the protected control channel and transport
// leases are caller-owned. This object is not a durable coordinator journal.
// Before invoking a transport that can write boundary scratch, the controller
// must observe BOTH prepared messages (phase()==executing). The same condition
// is enforced internally for every target callback. Legacy remote-step servers
// do not consume transaction frames implicitly.
//
// prepare quarantines the borrowed ordinary SerialSessionContext before peers
// may execute. A committed distributed frontier is NOT a healthy full local
// serial context. Only a confirmed two-peer predecision rollback may republish
// its saved history; D1/transfer failures preserve poison. No D1 clear/retag.
class HipDistributedTokenParticipant final : public DistributedNativeLocalExecutor {
public:
    HipDistributedTokenParticipant(CompiledSerialExecution serial,DistributedPlacementPlan canonical_plan,
        ModelContract model,HipLauncherCatalogConfig config,const ExecutionRuntimeResources& resources,
        SerialSessionContext& ordinary_session,HipLaunchContext scope,std::shared_ptr<const void> actual_owner,
        RuntimeMemoryAllocator& snapshot_allocator,AllocationGenerationSource& generations,
        DistributedTokenBinding binding,std::uint32_t local_node,std::uint64_t snapshot_budget_bytes);
    ~HipDistributedTokenParticipant();
    HipDistributedTokenParticipant(const HipDistributedTokenParticipant&) = delete;
    HipDistributedTokenParticipant& operator=(const HipDistributedTokenParticipant&) = delete;
    [[nodiscard]] DistributedTokenMessage prepare();
    void execute_local_step(const DistributedStepPlacement&) override;
    [[nodiscard]] DistributedTokenMessage finish_local();
    void observe_peer(const DistributedTokenMessage&);
    // Coordinator0 only. The returned commit decision has ALREADY become
    // irreversible locally, before any attempt to send it to the peer.
    [[nodiscard]] DistributedTokenMessage decide_commit();
    [[nodiscard]] DistributedTokenMessage decide_abort();
    [[nodiscard]] DistributedTokenMessage acknowledge_commit();
    [[nodiscard]] DistributedTokenMessage restore_local();
    // Requires both restored receipts; only healthy cancellation may restore
    // saved ordinary-session history. Copy/D1 failures stay poisoned.
    void finalize_rollback();
    void send(FramedDistributedTokenWire&,const DistributedTokenMessage&);
    void receive(FramedDistributedTokenWire&);
    // Source views stay owned by actual_owner and may be handed to a retained
    // transport lease. No subsequent transaction/release before that lease
    // retires. The caller's transport owner, not a CPU ACK, proves DMA lifetime.
    [[nodiscard]] std::vector<ExecutionBufferAddress> completed_boundary_views(const DistributedExecutionBoundary&) const;
    void communication_failed() noexcept;
    [[nodiscard]] DistributedTokenPhase phase() const noexcept;
    [[nodiscard]] std::uint64_t decision_identity() const noexcept;
    [[nodiscard]] std::optional<DistributedTokenCommit> committed_frontier() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
