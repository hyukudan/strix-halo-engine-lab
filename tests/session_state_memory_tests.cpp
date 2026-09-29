#include "strix/session_state_memory.hpp"

#include "strix/model_contract.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SerialSessionStateLayoutRequest request;
    request.context_tokens = 32'768;
    const auto layout =
        strix::build_serial_session_state_layout(
            model, request);

    strix::SessionStateMemoryBinding memory;
    memory.layout_identity =
        layout.layout_identity;
    memory.device_base =
        static_cast<std::uintptr_t>(
            0x40000000ULL);
    memory.device_bytes =
        layout.device_total_bytes;
    memory.device_allocation_generation = 3;
    memory.host_base =
        static_cast<std::uintptr_t>(
            0x50000000ULL);
    memory.host_bytes =
        layout.host_total_bytes;
    memory.host_allocation_generation = 5;

    strix::validate_session_state_memory_binding(
        layout, memory);

    const auto kv =
        strix::resolve_session_state(
            layout,
            memory,
            strix::SessionStateComponent::attention_kv);
    expect(kv.storage ==
               strix::SessionStateStorage::device &&
               kv.address >= memory.device_base &&
               kv.allocation_generation == 3,
           "KV resolves into device state pool");

    const auto position =
        strix::resolve_session_state(
            layout,
            memory,
            strix::SessionStateComponent::position_history);
    expect(position.storage ==
               strix::SessionStateStorage::host &&
               position.address >= memory.host_base &&
               position.allocation_generation == 5,
           "position metadata resolves into host state pool");

    const auto sparse =
        strix::resolve_session_state(
            layout,
            memory,
            strix::SessionStateComponent::sparse_index);
    expect(sparse.provisional,
           "resolved sparse state preserves provisional marker");

    auto wrong = memory;
    ++wrong.layout_identity;
    bool identity_rejected = false;
    try {
        strix::validate_session_state_memory_binding(
            layout, wrong);
    } catch (const std::invalid_argument&) {
        identity_rejected = true;
    }
    expect(identity_rejected,
           "state memory rejects layout identity mismatch");

    auto undersized = memory;
    --undersized.device_bytes;
    bool size_rejected = false;
    try {
        strix::validate_session_state_memory_binding(
            layout, undersized);
    } catch (const std::invalid_argument&) {
        size_rejected = true;
    }
    expect(size_rejected,
           "state memory rejects undersized device pool");

    auto overflow = memory;
    overflow.device_base =
        std::numeric_limits<std::uintptr_t>::max() -
        static_cast<std::uintptr_t>(
            layout.device_total_bytes / 2U);
    bool overflow_rejected = false;
    try {
        strix::validate_session_state_memory_binding(
            layout, overflow);
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    expect(overflow_rejected,
           "state memory rejects device address overflow");

    if (failures == 0) {
        std::cout << "session state memory tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
