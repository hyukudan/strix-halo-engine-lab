#pragma once

#include "strix/execution_runtime.hpp"
#include "strix/model_contract.hpp"
#include "strix/session_state_layout.hpp"

#include <cstdint>
#include <vector>

namespace strix {

struct SessionSnapshotRange {
    SessionStateComponent component =
        SessionStateComponent::attention_kv;
    SessionStateStorage storage =
        SessionStateStorage::device;
    std::uint64_t source_offset = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes = 0;
    bool provisional = false;
};

struct SerialSessionSnapshotRequest {
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t next_position = 0;
    std::uint64_t committed_tokens = 0;
    PersistentStateGenerations state;
    std::uint64_t alignment = 64;
};

struct SerialSessionSnapshotPlan {
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t state_layout_identity = 0;
    std::uint64_t next_position = 0;
    std::uint64_t committed_tokens = 0;
    std::uint64_t alignment = 64;
    PersistentStateGenerations state;

    std::uint64_t device_bytes = 0;
    std::uint64_t host_bytes = 0;
    std::uint64_t plan_identity = 0;

    std::vector<SessionSnapshotRange> ranges;
    GdnRecurrentStorage recurrent_storage = GdnRecurrentStorage::key_value;
};

[[nodiscard]] SerialSessionSnapshotPlan
build_serial_session_snapshot_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& layout,
    const SerialSessionSnapshotRequest& request);

}  // namespace strix
