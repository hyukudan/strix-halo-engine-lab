#include "strix/session_state_memory.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

void validate_pool(
    std::uintptr_t base,
    std::uint64_t bytes,
    std::uint64_t required,
    std::uint64_t generation,
    const char* name) {
    if (required == 0) {
        return;
    }
    if (base == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " session-state pool has a null base");
    }
    if (bytes < required) {
        throw std::invalid_argument(
            std::string(name) +
            " session-state pool is undersized");
    }
    if (generation == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " session-state allocation generation must be non-zero");
    }

    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (required >
        static_cast<std::uint64_t>(
            maximum - base)) {
        throw std::overflow_error(
            std::string(name) +
            " session-state address range overflows uintptr_t");
    }
}

}  // namespace

void validate_session_state_memory_binding(
    const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory) {
    if (layout.layout_identity == 0 ||
        memory.layout_identity !=
            layout.layout_identity) {
        throw std::invalid_argument(
            "session-state memory identity does not match layout");
    }

    validate_pool(
        memory.device_base,
        memory.device_bytes,
        layout.device_total_bytes,
        memory.device_allocation_generation,
        "device");

    validate_pool(
        memory.host_base,
        memory.host_bytes,
        layout.host_total_bytes,
        memory.host_allocation_generation,
        "host");
}

SessionStateAddress resolve_session_state(
    const SerialSessionStateLayout& layout,
    const SessionStateMemoryBinding& memory,
    SessionStateComponent component) {
    validate_session_state_memory_binding(
        layout, memory);

    const auto* allocation =
        find_session_state_allocation(
            layout, component);
    if (allocation == nullptr) {
        throw std::invalid_argument(
            "session-state component is not present in layout");
    }

    const bool device =
        allocation->storage ==
        SessionStateStorage::device;
    const auto base =
        device
            ? memory.device_base
            : memory.host_base;
    const auto pool_bytes =
        device
            ? memory.device_bytes
            : memory.host_bytes;
    const auto generation =
        device
            ? memory.device_allocation_generation
            : memory.host_allocation_generation;

    if (allocation->offset > pool_bytes ||
        allocation->bytes >
            pool_bytes - allocation->offset) {
        throw std::invalid_argument(
            "session-state component exceeds bound pool");
    }

    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (allocation->offset >
        static_cast<std::uint64_t>(
            maximum - base)) {
        throw std::overflow_error(
            "session-state component address overflows uintptr_t");
    }

    return SessionStateAddress{
        allocation->component,
        allocation->state_flag,
        allocation->storage,
        base +
            static_cast<std::uintptr_t>(
                allocation->offset),
        allocation->bytes,
        generation,
        allocation->provisional,
    };
}

}  // namespace strix
