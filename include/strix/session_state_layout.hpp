#pragma once

#include "strix/execution_plan.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>
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
};

struct SerialSessionStateLayoutRequest {
    std::uint64_t context_tokens = 32'768;
    std::uint64_t device_alignment = 256;
    std::uint64_t host_alignment = 64;
    bool include_sparse_index_candidate = true;
};

[[nodiscard]] SerialSessionStateLayout
build_serial_session_state_layout(
    const ModelContract& model,
    const SerialSessionStateLayoutRequest& request);

[[nodiscard]] const SessionStateAllocation*
find_session_state_allocation(
    const SerialSessionStateLayout& layout,
    SessionStateComponent component) noexcept;

}  // namespace strix
