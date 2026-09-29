#pragma once

#include "strix/session_state_layout.hpp"

#include <cstdint>

namespace strix {

struct SessionStateMemoryBinding {
    std::uint64_t layout_identity = 0;

    std::uintptr_t device_base = 0;
    std::uint64_t device_bytes = 0;
    std::uint64_t device_allocation_generation = 0;

    std::uintptr_t host_base = 0;
    std::uint64_t host_bytes = 0;
    std::uint64_t host_allocation_generation = 0;
};

struct SessionStateAddress {
    SessionStateComponent component =
        SessionStateComponent::attention_kv;
    PersistentState state_flag =
        PersistentState::none;
    SessionStateStorage storage =
        SessionStateStorage::device;
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t allocation_generation = 0;
    bool provisional = false;
};

void validate_session_state_memory_binding(
    const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory);

[[nodiscard]] SessionStateAddress resolve_session_state(
    const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory,
    SessionStateComponent component);

}  // namespace strix
