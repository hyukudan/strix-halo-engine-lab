#pragma once
#include "strix/distributed_control_wire.hpp"
#include "strix/distributed_position_completion.hpp"
#include "strix/hip_distributed_stage.hpp"
#include "strix/hip_distributed_stage_resources.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/uma_owner_inventory.hpp"
#include "strix/serial_session.hpp"
#include <memory>

namespace strix {
class NativeDistributedTokenRetirement;
// Retained sequence of explicitly bound two-peer token transactions, not a distributed decoder or
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
// serial context. Only initial confirmed two-peer predecision rollback may
// republish its original serial history; later rollback retains only a typed
// distributed frontier. D1/transfer failures preserve poison. No D1 clear/retag.
class HipDistributedTokenParticipant final : public DistributedNativeLocalExecutor,
    public std::enable_shared_from_this<HipDistributedTokenParticipant> {
public:
    HipDistributedTokenParticipant(CompiledSerialExecution serial,DistributedPlacementPlan canonical_plan,
        ModelContract model,HipLauncherCatalogConfig config,const ExecutionRuntimeResources& resources,
        SerialSessionContext& ordinary_session,HipLaunchContext scope,std::shared_ptr<const void> actual_owner,
        RuntimeMemoryAllocator& snapshot_allocator,AllocationGenerationSource& generations,
        DistributedTokenBinding binding,std::uint32_t local_node,std::uint64_t snapshot_budget_bytes);
    // Capacity source: actual selected backing and either default FULL state or
    // typed local owned ranges/snapshot; scratch stays full. No ordinary
    // healthy serial context is fabricated.
    HipDistributedTokenParticipant(std::shared_ptr<HipDistributedStageResources>,DistributedTokenBinding);
    ~HipDistributedTokenParticipant();
    HipDistributedTokenParticipant(const HipDistributedTokenParticipant&) = delete;
    HipDistributedTokenParticipant& operator=(const HipDistributedTokenParticipant&) = delete;
    // Does not prepare or mutate the next target. The current transaction must
    // have both commit ACKs, or a healthy bilateral rollback after a previous
    // distributed commit. The move-only proof comes from actual retired native
    // handles, not a token/control ACK. Source owners remain the same.
    void begin_next(std::uint32_t input_token,std::uint64_t generation,NativeDistributedTokenRetirement&&);
    [[nodiscard]] const DistributedTokenBinding& current_binding() const noexcept;
    [[nodiscard]] std::uint64_t binding_identity() const noexcept;
    [[nodiscard]] std::uint32_t local_node() const noexcept;
    // Exact retained capacity owner, null for the original full serial anchor.
    [[nodiscard]] const HipDistributedStageResources* stage_resources() const noexcept;
    // Stage-owned participants only. Includes the actual retained full or typed
    // snapshot backing plus the stage inventory, excluding transport/caches.
    [[nodiscard]] HipDistributedOwnerInventory owner_inventory(
        std::shared_ptr<const HipDistributedTokenParticipant>,UmaMemoryObservation = {}) const;
    // One logical-control capability per genuinely committed token. This does
    // not attest DMA; all data boundaries require independent retirement.
    [[nodiscard]] NativeDistributedPositionCompletion mint_position_dependency_completion();
    [[nodiscard]] std::optional<DistributedSequenceFrontier> frontier() const;
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
    // saved ordinary-session history ONLY before any distributed commit. Later
    // rollback republishes only the previous typed distributed frontier; the
    // ordinary serial anchor stays poisoned. Copy/D1 failures stay poisoned.
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
