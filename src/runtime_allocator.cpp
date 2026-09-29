#include "strix/runtime_allocator.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

void validate_returned_address(
    std::uintptr_t address,
    std::uint64_t alignment,
    const char* kind) {
    if (address == 0) {
        throw std::runtime_error(
            std::string(kind) +
            " allocator returned null address");
    }
    if (alignment == 0 ||
        address % alignment != 0) {
        throw std::runtime_error(
            std::string(kind) +
            " allocator did not honor requested alignment");
    }
}

}  // namespace

AllocationGenerationSource::AllocationGenerationSource(
    std::uint64_t next_generation)
    : next_generation_(next_generation) {
    if (next_generation_ == 0) {
        throw std::invalid_argument(
            "allocation generation source must start non-zero");
    }
}

std::uint64_t AllocationGenerationSource::next() {
    if (next_generation_ ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "allocation generation source exhausted");
    }
    return next_generation_++;
}

RuntimeExecutionMemoryOwner::RuntimeExecutionMemoryOwner(
    RuntimeExecutionMemoryOwner&& other) noexcept
    : allocator_(other.allocator_),
      scratch_(other.scratch_),
      state_(other.state_) {
    other.allocator_ = nullptr;
    other.scratch_ = {};
    other.state_ = {};
}

RuntimeExecutionMemoryOwner&
RuntimeExecutionMemoryOwner::operator=(
    RuntimeExecutionMemoryOwner&& other) noexcept {
    if (this != &other) {
        release();
        allocator_ = other.allocator_;
        scratch_ = other.scratch_;
        state_ = other.state_;
        other.allocator_ = nullptr;
        other.scratch_ = {};
        other.state_ = {};
    }
    return *this;
}

RuntimeExecutionMemoryOwner::~RuntimeExecutionMemoryOwner() {
    release();
}

RuntimeExecutionMemoryOwner
RuntimeExecutionMemoryOwner::allocate(
    const CompiledSerialExecution& compiled,
    const SerialSessionStateLayout& state_layout,
    RuntimeMemoryAllocator& allocator,
    AllocationGenerationSource& generations) {
    if (compiled.execution_identity == 0 ||
        compiled.arena.total_bytes == 0 ||
        state_layout.layout_identity == 0) {
        throw std::invalid_argument(
            "runtime allocation requires compiled execution and state layout");
    }

    RuntimeExecutionMemoryOwner owner;
    owner.allocator_ = &allocator;

    try {
        owner.scratch_.execution_identity =
            compiled.execution_identity;
        owner.scratch_.arena_bytes =
            compiled.arena.total_bytes;
        owner.scratch_.allocation_generation =
            generations.next();
        owner.scratch_.arena_base =
            allocator.allocate_device(
                owner.scratch_.arena_bytes,
                compiled.arena.alignment);
        validate_returned_address(
            owner.scratch_.arena_base,
            compiled.arena.alignment,
            "scratch");

        owner.state_.layout_identity =
            state_layout.layout_identity;

        if (state_layout.device_total_bytes != 0) {
            owner.state_.device_bytes =
                state_layout.device_total_bytes;
            owner.state_.device_allocation_generation =
                generations.next();
            owner.state_.device_base =
                allocator.allocate_device(
                    owner.state_.device_bytes,
                    state_layout.device_alignment);
            validate_returned_address(
                owner.state_.device_base,
                state_layout.device_alignment,
                "device state");
        }

        if (state_layout.host_total_bytes != 0) {
            owner.state_.host_bytes =
                state_layout.host_total_bytes;
            owner.state_.host_allocation_generation =
                generations.next();
            owner.state_.host_base =
                allocator.allocate_host(
                    owner.state_.host_bytes,
                    state_layout.host_alignment);
            validate_returned_address(
                owner.state_.host_base,
                state_layout.host_alignment,
                "host state");
        }

        validate_execution_memory_binding(
            compiled.execution_identity,
            compiled.arena,
            owner.scratch_);
        validate_session_state_memory_binding(
            state_layout,
            owner.state_);
    } catch (...) {
        owner.release();
        throw;
    }

    return owner;
}

void RuntimeExecutionMemoryOwner::release() noexcept {
    if (allocator_ == nullptr) {
        return;
    }

    if (state_.host_base != 0) {
        allocator_->free_host(
            state_.host_base,
            state_.host_bytes);
    }
    if (state_.device_base != 0) {
        allocator_->free_device(
            state_.device_base,
            state_.device_bytes);
    }
    if (scratch_.arena_base != 0) {
        allocator_->free_device(
            scratch_.arena_base,
            scratch_.arena_bytes);
    }

    allocator_ = nullptr;
    scratch_ = {};
    state_ = {};
}

}  // namespace strix
