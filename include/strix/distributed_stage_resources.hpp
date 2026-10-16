#pragma once
#include "strix/distributed_placement.hpp"
#include "strix/model_runtime_binding.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace strix {
// Immutable capacity phase1 scope. Global source/IDs/accesses remain unchanged;
// only real weight materialization is local. State, snapshot and scratch remain
// COMPLETE per peer; this plan never claims state sharding or hardware fit.
class DistributedStageResourcePlan final {
public:
    [[nodiscard]] static DistributedStageResourcePlan derive(const CompiledSerialExecution&,
        const ModelManifest&,const ExecutionTensorAccessPlan&,const DistributedPlacementPlan& canonical,
        std::uint32_t local_node);
    void validate_source(const CompiledSerialExecution&,const ModelManifest&,const ExecutionTensorAccessPlan&) const;
    [[nodiscard]] bool contains_step(std::uint32_t id) const noexcept {return id>=first_ && id<=last_;}
    [[nodiscard]] bool requires_tensor(std::uint32_t index) const noexcept {return index<required_.size() && required_[index]!=0;}
    [[nodiscard]] std::span<const std::uint32_t> tensor_indices() const noexcept {return indices_;}
    [[nodiscard]] std::uint64_t identity() const noexcept {return identity_;}
    [[nodiscard]] std::uint64_t execution_identity() const noexcept {return execution_;}
    [[nodiscard]] std::uint64_t manifest_identity() const noexcept {return manifest_;}
    [[nodiscard]] std::uint64_t access_identity() const noexcept {return access_;}
    [[nodiscard]] std::uint64_t placement_identity() const noexcept {return placement_.plan_identity;}
    [[nodiscard]] const DistributedPlacementPlan& canonical_placement() const noexcept {return placement_;}
    [[nodiscard]] std::uint32_t local_node() const noexcept {return node_;}
    [[nodiscard]] std::uint32_t first_step() const noexcept {return first_;}
    [[nodiscard]] std::uint32_t last_step() const noexcept {return last_;}
    [[nodiscard]] std::uint64_t required_weight_bytes() const noexcept {return weight_bytes_;}
private:
    DistributedStageResourcePlan() = default;
    DistributedPlacementPlan placement_;
    std::vector<std::uint8_t> required_;
    std::vector<std::uint32_t> indices_;
    std::uint64_t execution_=0,manifest_=0,access_=0,identity_=0,weight_bytes_=0;
    std::uint32_t node_=0,first_=0,last_=0;
};
// Dense nonlocal entries are EXPLICITLY absent/remote and cannot resolve on
// this stage. This does not relax validate_model_runtime_binding: the ordinary
// full-model validator must still reject such a manifest-local remote entry.
void validate_stage_model_runtime_binding(const DistributedStageResourcePlan&,
    const ModelManifest&,const ModelRuntimeBinding&);
} // namespace strix
