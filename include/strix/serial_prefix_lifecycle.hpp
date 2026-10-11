#pragma once

#include "strix/arithmetic_lineage.hpp"

#include "strix/model_contract.hpp"
#include "strix/session_restore.hpp"
#include "strix/session_snapshot.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include "strix/session_state_layout.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct SerialPrefixSnapshot {
    std::vector<std::uint32_t> prefix_tokens;
    SerialSessionSnapshotPlan snapshot;
    ArithmeticLineage arithmetic_lineage;
    std::uint64_t prefix_identity = 0;
};

[[nodiscard]] SerialPrefixSnapshot make_serial_prefix_snapshot(
    std::span<const std::uint32_t> prefix_tokens,
    const SerialSessionSnapshotPlan& snapshot,
    const ArithmeticLineage& arithmetic_lineage);

[[nodiscard]] bool serial_prefix_snapshot_matches(
    const SerialPrefixSnapshot& prefix,
    std::span<const std::uint32_t> requested_prefix,
    std::uint64_t compiled_execution_identity,
    std::uint64_t runtime_compatibility_identity,
    std::uint64_t state_layout_identity,
    const ArithmeticLineage* requested_history = nullptr) noexcept;

[[nodiscard]] SerialSessionRestorePlan
build_serial_prefix_restore_plan(
    const ModelContract& model,
    const SerialSessionStateLayout& destination_layout,
    std::uint64_t compiled_execution_identity,
    std::uint64_t runtime_compatibility_identity,
    std::span<const std::uint32_t> requested_prefix,
    const SerialPrefixSnapshot& prefix,
    const ArithmeticLineage* requested_history = nullptr);

void capture_serial_prefix_snapshot(
    const SerialPrefixSnapshot& prefix,
    const SerialSessionContext& session,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend);

void restore_serial_prefix_snapshot(
    SerialSessionContext& session,
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& backend,
    std::span<const std::uint32_t> requested_prefix,
    const SerialPrefixSnapshot& prefix,
    const ArithmeticLineage* requested_history = nullptr);

}  // namespace strix
