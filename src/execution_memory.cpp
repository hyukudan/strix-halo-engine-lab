#include "strix/execution_memory.hpp"

#include <limits>
#include <stdexcept>

namespace strix {

void validate_execution_memory_binding(
    const CompiledSerialExecution& compiled,
    const ExecutionMemoryBinding& memory) {
    if (compiled.execution_identity == 0 ||
        memory.execution_identity !=
            compiled.execution_identity) {
        throw std::invalid_argument(
            "execution memory binding identity does not match compiled plan");
    }
    if (compiled.arena.total_bytes == 0) {
        throw std::invalid_argument(
            "compiled execution arena must be non-empty");
    }
    if (memory.arena_base == 0) {
        throw std::invalid_argument(
            "execution memory binding has a null arena base");
    }
    if (memory.arena_bytes <
        compiled.arena.total_bytes) {
        throw std::invalid_argument(
            "execution memory binding is smaller than compiled arena");
    }
    if (memory.allocation_generation == 0) {
        throw std::invalid_argument(
            "execution memory allocation generation must be non-zero");
    }

    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (compiled.arena.total_bytes >
        static_cast<std::uint64_t>(
            maximum - memory.arena_base)) {
        throw std::overflow_error(
            "execution arena address range overflows uintptr_t");
    }
}

ExecutionBufferAddress resolve_execution_buffer(
    const CompiledSerialExecution& compiled,
    const ExecutionMemoryBinding& memory,
    ExecutionBufferUse use) {
    validate_execution_memory_binding(
        compiled, memory);

    const auto* allocation =
        find_arena_allocation(
            compiled.arena, use);
    if (allocation == nullptr) {
        throw std::invalid_argument(
            "execution buffer generation is not present in compiled arena");
    }

    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (allocation->offset >
        static_cast<std::uint64_t>(
            maximum - memory.arena_base)) {
        throw std::overflow_error(
            "execution buffer address overflows uintptr_t");
    }

    const auto address =
        memory.arena_base +
        static_cast<std::uintptr_t>(
            allocation->offset);

    if (allocation->bytes >
        memory.arena_bytes -
            allocation->offset) {
        throw std::invalid_argument(
            "execution buffer exceeds bound arena allocation");
    }

    return ExecutionBufferAddress{
        use,
        address,
        allocation->bytes,
        memory.allocation_generation,
    };
}

}  // namespace strix
