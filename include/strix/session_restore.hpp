#pragma once

#include "strix/model_contract.hpp"
#include "strix/session_snapshot.hpp"
#include "strix/session_state_layout.hpp"

#include <cstdint>
#include <vector>

namespace strix {

struct SessionRestoreRange {
    SessionStateComponent component =
        SessionStateComponent::attention_kv;
    SessionStateStorage storage =
        SessionStateStorage::device;
    std::uint64_t snapshot_offset = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes = 0;
};

struct SerialSessionRestorePlan {
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t snapshot_plan_identity = 0;
    std::uint64_t state_layout_identity = 0;
    std::uint64_t next_position = 0;
    std::uint64_t committed_tokens = 0;
    PersistentStateGenerations state;

    std::uint64_t snapshot_device_bytes = 0;
    std::uint64_t snapshot_host_bytes = 0;
    std::uint64_t plan_identity = 0;

    std::vector<SessionRestoreRange> ranges;
};

[[nodiscard]] SerialSessionRestorePlan
build_serial_session_restore_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& destination_layout,
    std::uint64_t runtime_compatibility_identity,
    const SerialSessionSnapshotPlan& snapshot);

}  // namespace strix
