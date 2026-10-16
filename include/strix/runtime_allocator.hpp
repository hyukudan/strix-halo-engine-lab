#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_memory.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>
#include <array>
#include <memory>

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

    // Read-only exact live allocation lease, scoped to the implementation's
    // issuer. Zero means opaque; it never creates allocation/ownership evidence.
    [[nodiscard]] virtual std::uint64_t allocation_lease_identity(
        std::uintptr_t, std::uint64_t, bool /*host*/) const { return 0; }
};

class AllocationGenerationSource {
public:
    explicit AllocationGenerationSource(
        std::uint64_t next_generation = 1);

    [[nodiscard]] std::uint64_t next();

private:
    std::uint64_t next_generation_ = 1;
};

class RuntimeExecutionMemoryOwner : public std::enable_shared_from_this<RuntimeExecutionMemoryOwner> {
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

    // Retains the actual allocator through release of every owned pool. The
    // borrowed overload above preserves its original lifetime contract.
    [[nodiscard]] static RuntimeExecutionMemoryOwner allocate(
        const CompiledSerialExecution& compiled,
        const SerialSessionStateLayout& state_layout,
        std::shared_ptr<RuntimeMemoryAllocator> allocator,
        AllocationGenerationSource& generations);

    [[nodiscard]] const std::shared_ptr<RuntimeMemoryAllocator>& retained_allocator() const noexcept {
        return allocator_owner_;
    }
    // Source bindings and nonopaque allocation leases saved by shared allocate
    // must still match. No GPU call, release or mutation; callers serialize
    // collection with use/release, do not move/replace this owner while its
    // inventory is used, and keep the returned inventory's pins alive.
    void validate_inventory_ownership() const;

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
    std::shared_ptr<RuntimeMemoryAllocator> allocator_owner_;
    ExecutionMemoryBinding scratch_;
    SessionStateMemoryBinding state_;
    ExecutionMemoryBinding original_scratch_;
    SessionStateMemoryBinding original_state_;
    std::array<std::uint64_t,3> allocation_leases_{};
};

}  // namespace strix
