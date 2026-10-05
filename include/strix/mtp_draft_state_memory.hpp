#pragma once

#include "strix/mtp_draft_state_contract.hpp"
#include "strix/runtime_allocator.hpp"

#include <cstdint>

namespace strix {

struct MtpDraftStateMemoryPlan {
    std::uint64_t proposal_identity = 0;
    std::uint64_t state_width = 0;
    std::uint64_t rows = 0;
    std::uint64_t element_bytes = 0;
    std::uint64_t alignment = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t plan_identity = 0;
};

struct MtpDraftStateMemoryBinding {
    std::uint64_t plan_identity = 0;
    std::uint64_t proposal_identity = 0;
    std::uint64_t state_width = 0;
    std::uint64_t rows = 0;
    std::uint64_t element_bytes = 0;
    std::uintptr_t device_base = 0;
    std::uint64_t bytes = 0;
    std::uint64_t allocation_generation = 0;
};

[[nodiscard]] MtpDraftStateMemoryPlan
plan_mtp_draft_state_memory(
    std::uint64_t proposal_identity,
    const MtpDraftStateContract& contract,
    std::uint64_t rows = 1,
    std::uint64_t element_bytes = 2,
    std::uint64_t alignment = 256);

void validate_mtp_draft_state_memory_binding(
    const MtpDraftStateMemoryPlan& plan,
    const MtpDraftStateMemoryBinding& binding);

class MtpDraftStateMemoryOwner {
public:
    MtpDraftStateMemoryOwner() = default;
    MtpDraftStateMemoryOwner(
        const MtpDraftStateMemoryOwner&) = delete;
    MtpDraftStateMemoryOwner& operator=(
        const MtpDraftStateMemoryOwner&) = delete;

    MtpDraftStateMemoryOwner(
        MtpDraftStateMemoryOwner&& other) noexcept;
    MtpDraftStateMemoryOwner& operator=(
        MtpDraftStateMemoryOwner&& other) noexcept;

    ~MtpDraftStateMemoryOwner();

    [[nodiscard]] static MtpDraftStateMemoryOwner allocate(
        const MtpDraftStateMemoryPlan& plan,
        RuntimeMemoryAllocator& allocator,
        AllocationGenerationSource& generations);

    [[nodiscard]] const MtpDraftStateMemoryBinding&
    binding() const noexcept {
        return binding_;
    }

    [[nodiscard]] bool valid() const noexcept {
        return allocator_ != nullptr;
    }

private:
    void release() noexcept;

    RuntimeMemoryAllocator* allocator_ = nullptr;
    MtpDraftStateMemoryBinding binding_;
};

}  // namespace strix
