#include "strix/execution_memory.hpp"

#include <limits>
#include <stdexcept>

namespace strix {

void validate_execution_memory_binding(
    std::uint64_t execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionMemoryBinding& memory) {
    if (execution_identity == 0 ||
        memory.execution_identity !=
            execution_identity) {
        throw std::invalid_argument(
            "execution memory binding identity does not match compiled plan");
    }
    if (arena.total_bytes == 0) {
        throw std::invalid_argument(
            "compiled execution arena must be non-empty");
    }
    if (memory.arena_base == 0) {
        throw std::invalid_argument(
            "execution memory binding has a null arena base");
    }
    if (memory.arena_bytes <
        arena.total_bytes) {
        throw std::invalid_argument(
            "execution memory binding is smaller than compiled arena");
    }
    if (memory.allocation_generation == 0) {
        throw std::invalid_argument(
            "execution memory allocation generation must be non-zero");
    }

    const auto maximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (memory.arena_bytes >
        static_cast<std::uint64_t>(
            maximum - memory.arena_base)) {
        throw std::overflow_error(
            "execution arena address range overflows uintptr_t");
    }
}

namespace {
ExecutionBufferAddress bound_execution_buffer(
    const ExecutionArenaPlan& arena,
    const ExecutionMemoryBinding& memory,
    ExecutionBufferUse use,
    const ExecutionArenaAllocation* allocation) {
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

    if (allocation->offset > arena.total_bytes ||
        allocation->bytes > arena.total_bytes - allocation->offset ||
        allocation->offset > memory.arena_bytes || allocation->bytes >
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
}  // namespace

ExecutionBufferAddress resolve_execution_buffer(
    std::uint64_t execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionMemoryBinding& memory,
    ExecutionBufferUse use) {
    validate_execution_memory_binding(
        execution_identity,
        arena,
        memory);
    return bound_execution_buffer(arena, memory, use, find_arena_allocation(arena, use));
}

ExecutionBufferAddress resolve_execution_buffer_hinted(
    std::uint64_t execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionMemoryBinding& memory,
    ExecutionBufferUse use,
    std::size_t& hint) {
    validate_execution_memory_binding(
        execution_identity,
        arena,
        memory);
    const auto& allocations = arena.allocations;
    const auto* allocation = hint < allocations.size() && allocations[hint].use == use
        ? &allocations[hint] : find_arena_allocation(arena, use);
    if (allocation != nullptr) hint = static_cast<std::size_t>(allocation - allocations.data());
    return bound_execution_buffer(arena, memory, use, allocation);
}

}  // namespace strix
