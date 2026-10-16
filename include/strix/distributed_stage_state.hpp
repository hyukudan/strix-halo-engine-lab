#pragma once
#include "strix/distributed_stage_resources.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include <array>
#include <span>
#include <vector>

namespace strix {
enum class DistributedStageStatePresence : std::uint8_t { absent, partial, complete };
struct DistributedStageStateRange {
    SessionStateComponent component;
    PersistentState state_flag;
    SessionStateStorage storage;
    std::uint64_t logical_offset = 0, bytes = 0, local_offset = 0;
    bool provisional = false;
    bool operator==(const DistributedStageStateRange&) const = default;
};
// Immutable global-logical to local-packed mapping. A borrowed memory pointer
// must be retained by the upstream actual owner together with this object and
// both real pools; the mapping does not retain that owner (no self-cycle).
class DistributedStageStateMapping final {
public:
    [[nodiscard]] static DistributedStageStateMapping derive(const CompiledSerialExecution&,
        const ModelContract&,const SerialSessionStateLayout&,const DistributedStageResourcePlan&);
    void validate_source(const CompiledSerialExecution&,const ModelContract&,
        const SerialSessionStateLayout&,const DistributedStageResourcePlan&) const;
    void validate_layout(const SerialSessionStateLayout&) const;
    void validate_memory(const SerialSessionStateLayout&,const SessionStateMemoryBinding&) const;
    [[nodiscard]] SessionStateAddress resolve(const SerialSessionStateLayout&,
        const SessionStateMemoryBinding&,SessionStateComponent,std::uint64_t logical_offset,std::uint64_t bytes) const;
    [[nodiscard]] std::uint64_t identity() const noexcept {return identity_;}
    [[nodiscard]] std::uint64_t source_stage_identity() const noexcept {return stage_;}
    [[nodiscard]] std::uint64_t layout_identity() const noexcept {return layout_.layout_identity;}
    [[nodiscard]] std::uint64_t execution_identity() const noexcept {return execution_;}
    [[nodiscard]] std::uint64_t device_bytes() const noexcept {return device_;}
    [[nodiscard]] std::uint64_t host_bytes() const noexcept {return host_;}
    [[nodiscard]] std::uint32_t local_node() const noexcept {return node_;}
    [[nodiscard]] std::span<const DistributedStageStateRange> ranges() const noexcept {return ranges_;}
    [[nodiscard]] DistributedStageStatePresence presence(SessionStateComponent) const;
private:
    DistributedStageStateMapping() = default;
    SerialSessionStateLayout layout_;
    std::vector<DistributedStageStateRange> ranges_;
    std::array<DistributedStageStatePresence,7> presence_{};
    std::uint64_t stage_=0,execution_=0,identity_=0,device_=0,host_=0;
    std::uint32_t node_=0;
};
struct DistributedStageSnapshotRange {
    SessionStateStorage storage;
    std::uint64_t source_offset=0,destination_offset=0,bytes=0;
    bool operator==(const DistributedStageSnapshotRange&) const = default;
};
// Different type from SerialSessionSnapshotPlan: ALL and ONLY actually owned
// ranges, no invented complete-state validity or ordinary-session publication.
struct DistributedStageSnapshotPlan {
    std::uint64_t mapping_identity=0,logical_plan_identity=0,plan_identity=0;
    std::uint64_t device_bytes=0,host_bytes=0;
    std::vector<DistributedStageSnapshotRange> ranges;
};
struct DistributedStageSnapshotMemoryBinding {
    std::uint64_t snapshot_plan_identity=0;
    std::uintptr_t device_section_base=0;
    std::uint64_t device_section_bytes=0;
    std::uintptr_t host_section_base=0;
    std::uint64_t host_section_bytes=0,allocation_generation=0;
};
[[nodiscard]] DistributedStageSnapshotPlan build_stage_snapshot_plan(
    const DistributedStageStateMapping&,std::uint64_t logical_plan_identity);
void validate_stage_snapshot_binding(const DistributedStageSnapshotPlan&,
    const DistributedStageSnapshotMemoryBinding&);
void capture_stage_snapshot_payload(const DistributedStageSnapshotPlan&,const SerialSessionStateLayout&,
    const SessionStateMemoryBinding&,const DistributedStageSnapshotMemoryBinding&,SessionSnapshotTransferBackend&);
void restore_stage_snapshot_payload(const DistributedStageSnapshotPlan&,const SerialSessionStateLayout&,
    const SessionStateMemoryBinding&,const DistributedStageSnapshotMemoryBinding&,SessionSnapshotTransferBackend&);
} // namespace strix
