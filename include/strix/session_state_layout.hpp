#pragma once

#include "strix/gdn_recurrent_storage.hpp"

#include "strix/execution_plan.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace strix {

enum class SessionStateComponent {
    attention_kv,
    recurrent_matrix,
    gdn_convolution,
    ple_convolution,
    sparse_index,
    ngram_history,
    position_history,
};

enum class SessionStateStorage {
    device,
    host,
};

struct SessionStateAllocation {
    SessionStateComponent component =
        SessionStateComponent::attention_kv;
    PersistentState state_flag =
        PersistentState::none;
    SessionStateStorage storage =
        SessionStateStorage::device;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    bool provisional = false;
};

struct SerialSessionStateLayout {
    std::uint64_t context_tokens = 0;
    std::uint64_t device_alignment = 256;
    std::uint64_t host_alignment = 64;
    std::uint64_t device_total_bytes = 0;
    std::uint64_t host_total_bytes = 0;
    std::uint64_t layout_identity = 0;
    std::vector<SessionStateAllocation> allocations;
    GdnRecurrentStorage recurrent_storage = GdnRecurrentStorage::key_value;
};

struct SerialSessionStateLayoutRequest {
    std::uint64_t context_tokens = 32'768;
    std::uint64_t device_alignment = 256;
    std::uint64_t host_alignment = 64;
    bool include_sparse_index_candidate = true;
    GdnRecurrentStorage recurrent_storage = GdnRecurrentStorage::key_value;
};

[[nodiscard]] SerialSessionStateLayout
build_serial_session_state_layout(
    const ModelContract& model,
    const SerialSessionStateLayoutRequest& request);

[[nodiscard]] const SessionStateAllocation*
find_session_state_allocation(
    const SerialSessionStateLayout& layout,
    SessionStateComponent component) noexcept;

// Explicit import/export operation on host-accessible staging only. It never
// reads a device pointer or changes a live session's layout/identity. Owners,
// generations and spans remain stable until return. Fence must establish that
// all producers/consumers of BOTH views are quiescent; it may throw before writes.
struct GdnRecurrentHostView {
    std::span<std::uint32_t> words;
    std::shared_ptr<const void> owner;
    std::uintptr_t owner_base = 0;
    std::uint64_t owner_bytes = 0;
    std::uint64_t allocation_generation = 0;
};

void convert_gdn_recurrent_host_storage(
    GdnRecurrentHostView source, GdnRecurrentStorage source_storage,
    GdnRecurrentHostView destination, GdnRecurrentStorage destination_storage,
    std::uint64_t heads, std::uint64_t keys, std::uint64_t values,
    const std::function<void()>& quiesce);

void validate_serial_session_state_layout_identity(
    const ModelContract& model, const SerialSessionStateLayout& layout);

// Deterministic metadata digest for explicitly authored layouts. This binds
// physical storage and offsets; it does not admit a layout to a native kernel.
[[nodiscard]] std::uint64_t serial_session_state_layout_identity(
    const ModelContract& model, const SerialSessionStateLayout& layout);

}  // namespace strix
