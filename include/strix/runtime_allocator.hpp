#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_memory.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>

namespace strix {

class RuntimeMemoryAllocator {
public:
    virtual ~RuntimeMemoryAllocator() = default;

    [[nodiscard]] virtual std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) = 0;

    [[nodiscard]] virtual std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t alignment) = 0;

    virtual void free_device(
        std::uintptr_t address,
        std::uint64_t bytes) noexcept = 0;

    virtual void free_host(
        std::uintptr_t address,
        std::uint64_t bytes) noexcept = 0;
};

class AllocationGenerationSource {
public:
    explicit AllocationGenerationSource(
        std::uint64_t next_generation = 1);

    [[nodiscard]] std::uint64_t next();

private:
    std::uint64_t next_generation_ = 1;
};

class RuntimeExecutionMemoryOwner {
public:
    RuntimeExecutionMemoryOwner() = default;
    RuntimeExecutionMemoryOwner(
        const RuntimeExecutionMemoryOwner&) = delete;
    RuntimeExecutionMemoryOwner& operator=(
        const RuntimeExecutionMemoryOwner&) = delete;

    RuntimeExecutionMemoryOwner(
        RuntimeExecutionMemoryOwner&& other) noexcept;
    RuntimeExecutionMemoryOwner& operator=(
        RuntimeExecutionMemoryOwner&& other) noexcept;

    ~RuntimeExecutionMemoryOwner();

    [[nodiscard]] static RuntimeExecutionMemoryOwner allocate(
        const CompiledSerialExecution& compiled,
        const SerialSessionStateLayout& state_layout,
        RuntimeMemoryAllocator& allocator,
        AllocationGenerationSource& generations);

    [[nodiscard]] const ExecutionMemoryBinding&
    scratch() const noexcept {
        return scratch_;
    }

    [[nodiscard]] const SessionStateMemoryBinding&
    state() const noexcept {
        return state_;
    }

    [[nodiscard]] bool valid() const noexcept {
        return allocator_ != nullptr;
    }

private:
    void release() noexcept;

    RuntimeMemoryAllocator* allocator_ = nullptr;
    ExecutionMemoryBinding scratch_;
    SessionStateMemoryBinding state_;
};

}  // namespace strix
