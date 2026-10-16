#pragma once
#include "strix/distributed_stage_resources.hpp"
#include "strix/distributed_token_transaction.hpp"
#include "strix/execution_resources.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_model_residency.hpp"
#include <memory>

namespace strix {
class HipDistributedTokenParticipant;
class HipDistributedStageResources;
// Produced only after genuine p0 initialization of COMPLETE local state pools.
// It carries no fabricated healthy ordinary serial session or nonzero prefix.
class HipDistributedStageInitialFrontier final {
public:
    [[nodiscard]] const DistributedTokenPeer& peer() const noexcept {return peer_;}
    [[nodiscard]] const PersistentStateGenerations& global_state() const noexcept {return global_;}
private:
    friend class HipDistributedStageResources;
    HipDistributedStageInitialFrontier(DistributedTokenPeer peer,PersistentStateGenerations global):peer_(peer),global_(global) {}
    DistributedTokenPeer peer_;
    PersistentStateGenerations global_;
};
struct HipDistributedStageBudget {
    std::uint64_t retained_bytes=0,setup_peak_bytes=0,headroom_bytes=0,snapshot_bytes=0;
};
struct HipDistributedStageReservation {
    HipModelResidencyReservation weights;
    std::uint64_t full_state_bytes=0,full_scratch_bytes=0,d1_bytes=8,snapshot_budget_bytes=0;
    std::uint64_t allocator_padding_bytes=0,headroom_bytes=0,retained_bytes=0,setup_peak_bytes=0;
};
// Capacity PHASE1: actual selected registered/copy backing; full state/snapshot/
// scratch retained. Full file mapping is a retained source, not a claim that
// all its pages are pinned/device-resident. No driver fit/performance claim.
// One serialized caller owns setup/claim/progress; initial_frontier is claimed
// exactly once by the participant, not a thread-safe shared session factory.
class HipDistributedStageResources final : public std::enable_shared_from_this<HipDistributedStageResources> {
public:
    [[nodiscard]] static std::shared_ptr<HipDistributedStageResources> create(
        std::shared_ptr<const MappedModelFile>,const CompiledSerialExecution&,const ModelManifest&,
        const ExecutionTensorAccessPlan&,const DistributedPlacementPlan& canonical,std::uint32_t node,
        const ModelContract&,const HipLauncherCatalogConfig&,std::uint64_t context_capacity,
        HipDistributedStageBudget,HipModelResidencyPolicy = {});
    ~HipDistributedStageResources();
    HipDistributedStageResources(const HipDistributedStageResources&) = delete;
    HipDistributedStageResources& operator=(const HipDistributedStageResources&) = delete;
    [[nodiscard]] const CompiledSerialExecution& serial() const noexcept;
    [[nodiscard]] const ModelContract& model() const noexcept;
    [[nodiscard]] const HipLauncherCatalogConfig& config() const noexcept;
    [[nodiscard]] const DistributedStageResourcePlan& plan() const noexcept;
    [[nodiscard]] const ExecutionRuntimeResources& resources() const noexcept;
    [[nodiscard]] const HipLaunchContext& scope() const noexcept;
    [[nodiscard]] const HipDistributedStageInitialFrontier& initial_frontier() const;
    [[nodiscard]] const HipDistributedStageReservation& reservation() const noexcept;
    [[nodiscard]] RuntimeMemoryAllocator& snapshot_allocator() noexcept;
    [[nodiscard]] AllocationGenerationSource& generations() noexcept;
    [[nodiscard]] std::uint64_t snapshot_budget_bytes() const noexcept;
    // Revalidates actual retained backing and immutable source before borrowing.
    void validate() const;
private:
    friend class HipDistributedTokenParticipant;
    void validate_initial_binding(const DistributedTokenBinding&) const;
    void claim_initial(const DistributedTokenBinding&);
    HipDistributedStageResources();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
