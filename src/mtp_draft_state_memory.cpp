#include "strix/mtp_draft_state_memory.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            "MTP draft-state memory size overflow");
    }
    return left * right;
}

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t build_plan_identity(
    const MtpDraftStateMemoryPlan& plan) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, plan.proposal_identity);
    hash_u64(hash, plan.state_width);
    hash_u64(hash, plan.rows);
    hash_u64(hash, plan.element_bytes);
    hash_u64(hash, plan.alignment);
    hash_u64(hash, plan.total_bytes);
    return hash;
}

}  // namespace

MtpDraftStateMemoryPlan
plan_mtp_draft_state_memory(
    std::uint64_t proposal_identity,
    const MtpDraftStateContract& contract,
    std::uint64_t rows,
    std::uint64_t element_bytes,
    std::uint64_t alignment) {
    if (proposal_identity == 0 ||
        contract.state_width == 0 ||
        !contract.one_state_row_per_request) {
        throw std::invalid_argument(
            "MTP draft-state memory requires initialized proposal/state contract");
    }
    if (rows == 0 ||
        element_bytes == 0 ||
        alignment == 0) {
        throw std::invalid_argument(
            "MTP draft-state memory dimensions/alignment must be non-zero");
    }

    MtpDraftStateMemoryPlan plan;
    plan.proposal_identity = proposal_identity;
    plan.state_width = contract.state_width;
    plan.rows = rows;
    plan.element_bytes = element_bytes;
    plan.alignment = alignment;
    plan.total_bytes =
        checked_mul(
            checked_mul(
                contract.state_width,
                rows),
            element_bytes);
    plan.plan_identity =
        build_plan_identity(plan);
    if (plan.plan_identity == 0) {
        throw std::logic_error(
            "MTP draft-state memory plan identity must be non-zero");
    }
    return plan;
}

void validate_mtp_draft_state_memory_binding(
    const MtpDraftStateMemoryPlan& plan,
    const MtpDraftStateMemoryBinding& binding) {
    if (plan.plan_identity == 0 ||
        binding.plan_identity !=
            plan.plan_identity ||
        binding.proposal_identity !=
            plan.proposal_identity ||
        binding.state_width !=
            plan.state_width ||
        binding.rows != plan.rows ||
        binding.element_bytes !=
            plan.element_bytes ||
        binding.device_base == 0 ||
        binding.bytes !=
            plan.total_bytes ||
        binding.allocation_generation == 0 ||
        binding.device_base %
                plan.alignment !=
            0) {
        throw std::invalid_argument(
            "MTP draft-state memory binding is incompatible with plan");
    }
}

MtpDraftStateMemoryOwner::MtpDraftStateMemoryOwner(
    MtpDraftStateMemoryOwner&& other) noexcept
    : allocator_(other.allocator_),
      binding_(other.binding_) {
    other.allocator_ = nullptr;
    other.binding_ = {};
}

MtpDraftStateMemoryOwner&
MtpDraftStateMemoryOwner::operator=(
    MtpDraftStateMemoryOwner&& other) noexcept {
    if (this != &other) {
        release();
        allocator_ = other.allocator_;
        binding_ = other.binding_;
        other.allocator_ = nullptr;
        other.binding_ = {};
    }
    return *this;
}

MtpDraftStateMemoryOwner::~MtpDraftStateMemoryOwner() {
    release();
}

MtpDraftStateMemoryOwner
MtpDraftStateMemoryOwner::allocate(
    const MtpDraftStateMemoryPlan& plan,
    RuntimeMemoryAllocator& allocator,
    AllocationGenerationSource& generations) {
    if (plan.plan_identity == 0 ||
        plan.total_bytes == 0 ||
        plan.alignment == 0) {
        throw std::invalid_argument(
            "MTP draft-state allocation requires valid memory plan");
    }

    MtpDraftStateMemoryOwner owner;
    owner.allocator_ = &allocator;
    owner.binding_.plan_identity =
        plan.plan_identity;
    owner.binding_.proposal_identity =
        plan.proposal_identity;
    owner.binding_.state_width =
        plan.state_width;
    owner.binding_.rows =
        plan.rows;
    owner.binding_.element_bytes =
        plan.element_bytes;
    owner.binding_.bytes =
        plan.total_bytes;
    owner.binding_.allocation_generation =
        generations.next();

    try {
        owner.binding_.device_base =
            allocator.allocate_device(
                plan.total_bytes,
                plan.alignment);
        validate_mtp_draft_state_memory_binding(
            plan,
            owner.binding_);
    } catch (...) {
        owner.release();
        throw;
    }

    return owner;
}

void MtpDraftStateMemoryOwner::release() noexcept {
    if (allocator_ == nullptr) {
        return;
    }
    if (binding_.device_base != 0) {
        allocator_->free_device(
            binding_.device_base,
            binding_.bytes);
    }
    allocator_ = nullptr;
    binding_ = {};
}

}  // namespace strix
