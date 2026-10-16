#include "strix/runtime_allocator.hpp"

#include <limits>
#include <stdexcept>
#include <string>
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
      allocator_owner_(std::move(other.allocator_owner_)),
      scratch_(other.scratch_),
      state_(other.state_),
      original_scratch_(other.original_scratch_),
      original_state_(other.original_state_),
      allocation_leases_(other.allocation_leases_) {
    other.allocator_ = nullptr;
    other.scratch_ = {};
    other.state_ = {};
    other.original_scratch_ = {};
    other.original_state_ = {};
    other.allocation_leases_ = {};
}

RuntimeExecutionMemoryOwner&
RuntimeExecutionMemoryOwner::operator=(
    RuntimeExecutionMemoryOwner&& other) noexcept {
    if (this != &other) {
        release();
        allocator_ = other.allocator_;
        allocator_owner_ = std::move(other.allocator_owner_);
        scratch_ = other.scratch_;
        state_ = other.state_;
        original_scratch_ = other.original_scratch_;
        original_state_ = other.original_state_;
        allocation_leases_ = other.allocation_leases_;
        other.allocator_ = nullptr;
        other.scratch_ = {};
        other.state_ = {};
        other.original_scratch_ = {};
        other.original_state_ = {};
        other.allocation_leases_ = {};
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

RuntimeExecutionMemoryOwner RuntimeExecutionMemoryOwner::allocate(
    const CompiledSerialExecution& compiled,const SerialSessionStateLayout& state_layout,
    std::shared_ptr<RuntimeMemoryAllocator> allocator,AllocationGenerationSource& generations) {
    if(!allocator) throw std::invalid_argument("retained runtime allocation requires an actual shared allocator");
    auto owner=allocate(compiled,state_layout,*allocator,generations);
    owner.allocator_owner_=std::move(allocator);
    owner.original_scratch_=owner.scratch_;
    owner.original_state_=owner.state_;
    owner.allocation_leases_[0]=owner.allocator_->allocation_lease_identity(
        owner.scratch_.arena_base,owner.scratch_.arena_bytes,false);
    if(owner.state_.device_bytes) owner.allocation_leases_[1]=owner.allocator_->allocation_lease_identity(
        owner.state_.device_base,owner.state_.device_bytes,false);
    if(owner.state_.host_bytes) owner.allocation_leases_[2]=owner.allocator_->allocation_lease_identity(
        owner.state_.host_base,owner.state_.host_bytes,true);
    return owner;
}

void RuntimeExecutionMemoryOwner::validate_inventory_ownership() const {
    const auto& a=scratch_;const auto& b=original_scratch_;
    const auto& x=state_;const auto& y=original_state_;
    if(!allocator_ || !allocator_owner_ || allocator_owner_.get()!=allocator_ ||
        a.execution_identity!=b.execution_identity || a.arena_base!=b.arena_base ||
        a.arena_bytes!=b.arena_bytes || a.allocation_generation!=b.allocation_generation ||
        x.layout_identity!=y.layout_identity || x.device_base!=y.device_base || x.device_bytes!=y.device_bytes ||
        x.device_allocation_generation!=y.device_allocation_generation || x.host_base!=y.host_base ||
        x.host_bytes!=y.host_bytes || x.host_allocation_generation!=y.host_allocation_generation ||
        x.stage_mapping!=y.stage_mapping)
        throw std::invalid_argument("runtime inventory requires unchanged actual retained allocator/bindings");
    const std::array<std::uintptr_t,3> addresses{a.arena_base,x.device_base,x.host_base};
    const std::array<std::uint64_t,3> bytes{a.arena_bytes,x.device_bytes,x.host_bytes};
    for(std::size_t i=0;i<3U;++i) {
        if(!bytes[i]) {
            if(addresses[i] || allocation_leases_[i])
                throw std::invalid_argument("runtime inventory absent pool carries an allocation lease");
        } else if(!allocation_leases_[i] || allocator_->allocation_lease_identity(addresses[i],bytes[i],i==2U)!=allocation_leases_[i]) {
            throw std::invalid_argument("runtime inventory allocation lease changed or is opaque");
        }
    }
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
    original_scratch_ = {};
    original_state_ = {};
    allocation_leases_ = {};
    // Last: freeing pools above still needs the actual allocator alive.
    allocator_owner_.reset();
}

}  // namespace strix
