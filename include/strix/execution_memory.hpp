#pragma once

#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"

#include <cstdint>

namespace strix {

struct ExecutionMemoryBinding {
    std::uint64_t execution_identity = 0;
    std::uintptr_t arena_base = 0;
    std::uint64_t arena_bytes = 0;
    std::uint64_t allocation_generation = 0;
};

struct ExecutionBufferAddress {
    ExecutionBufferUse use;
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t allocation_generation = 0;
};

void validate_execution_memory_binding(
    std::uint64_t execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionMemoryBinding& memory);

[[nodiscard]] ExecutionBufferAddress resolve_execution_buffer(
    std::uint64_t execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionMemoryBinding& memory,
    ExecutionBufferUse use);

}  // namespace strix
