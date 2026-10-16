#pragma once

#include "strix/mapped_model_file.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/uma_owner_inventory.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace strix {

enum class HipModelResidencyMode {
    registered_file_mapping,
    resident_copy,
};

struct HipModelResidencyPolicy {
    HipModelResidencyMode mode =
        HipModelResidencyMode::registered_file_mapping;

    // File-backed candidates, such as the giant PLE table, remain ordinary
    // host mappings unless explicitly promoted.
    bool promote_file_backed_candidates = false;

    // Coalesce adjacent registered ranges whose page-aligned gap is at most
    // this many bytes. Zero means only overlapping/touching ranges.
    std::uint64_t coalesce_gap_bytes = 0;
    // Declared model backing reservations, not observed RAM or free capacity.
    // Optional limits are checked before registration or allocation.
    UmaReservationLimits reservation_limits;
};

struct HipModelResidencyReservation {
    std::uint64_t mapped_reserved_bytes = 0;
    std::uint64_t registered_alias_bytes = 0;
    std::uint64_t copied_reserved_bytes = 0;
    std::uint64_t retained_reserved_bytes = 0;
    std::uint64_t setup_peak_reserved_bytes = 0;
};
[[nodiscard]] HipModelResidencyReservation plan_hip_model_residency_reservation(
    const MappedModelFile&, const ModelManifest&, HipModelResidencyPolicy = {});

struct CopyResidencyRange {
    std::uint64_t begin = 0, end = 0;
};
// Pure layout shared by allocation and pre-allocation kernel eligibility.
[[nodiscard]] std::vector<CopyResidencyRange> build_copy_residency_ranges(
    const ModelManifest&, const HipModelResidencyPolicy&);
[[nodiscard]] std::uint64_t copy_residency_tensor_offset(
    const ManifestTensor&, const std::vector<CopyResidencyRange>&);

class HipModelResidency : public std::enable_shared_from_this<HipModelResidency> {
public:
    HipModelResidency() = default;
    HipModelResidency(
        const HipModelResidency&) = delete;
    HipModelResidency& operator=(
        const HipModelResidency&) = delete;

    HipModelResidency(
        HipModelResidency&& other) noexcept;
    HipModelResidency& operator=(
        HipModelResidency&& other) noexcept;

    ~HipModelResidency();

    [[nodiscard]] static HipModelResidency create(
        const MappedModelFile& mapped,
        const ModelManifest& manifest,
        AllocationGenerationSource& generations,
        HipModelResidencyPolicy policy = {});
    // Strong mapping form: registration and host-only tensor views cannot
    // outlive their source. No extra copy or promotion is performed.
    [[nodiscard]] static HipModelResidency create(
        std::shared_ptr<const MappedModelFile> mapped, const ModelManifest& manifest,
        AllocationGenerationSource& generations, HipModelResidencyPolicy policy = {});

    // Snapshot actual host/device alias ranges from strong-create ownership.
    // An optional mapping argument must share that exact retained control block.
    // Borrowed-create objects cannot manufacture strong ownership after setup.
    // Keep metadata stable and owners unmoved until consumers quiesce. Returned
    // pins retain residency before releasing the mapping, including snapshots.
    // This is backing-lease evidence, not PFN/coherency/residency measurement.
    [[nodiscard]] UmaOwnerInventory owner_inventory(
        std::shared_ptr<const HipModelResidency> actual_owner,
        std::shared_ptr<const MappedModelFile> mapped_owner = {},
        UmaMemoryObservation observation = {}) const;
    [[nodiscard]] const HipModelResidencyReservation& reservation() const noexcept { return reservation_; }

    [[nodiscard]] const ModelRuntimeBinding&
    binding() const noexcept {
        return binding_;
    }

    [[nodiscard]] HipModelResidencyMode
    mode() const noexcept {
        return mode_;
    }

    [[nodiscard]] std::uint64_t
    registered_bytes() const noexcept {
        return registered_bytes_;
    }

    [[nodiscard]] std::uint64_t
    copied_bytes() const noexcept {
        return copied_bytes_;
    }

    [[nodiscard]] std::uint64_t generation() const noexcept {
        return generation_;
    }

    // Pure ownership query over the actual registered device aliases and
    // copied device allocations, not host mapping addresses or tensor labels.
    // The caller retains this object and the source mapping through quiescence.
    // Empty, overflowing, or cross-allocation views are never covered.
    [[nodiscard]] bool covers_device_range(
        std::uintptr_t address, std::uint64_t bytes) const noexcept;

private:
    struct RegisteredRange {
        std::uintptr_t host_base = 0;
        std::uintptr_t device_base = 0;
        std::uint64_t bytes = 0;
    };

    struct CopiedRange {
        std::uint64_t file_offset = 0;
        std::uintptr_t device_base = 0;
        std::uint64_t bytes = 0;
        std::uint64_t allocation_identity = 0;
    };

    void release() noexcept;
    [[nodiscard]] static HipModelResidency create_impl(const MappedModelFile&,
        std::shared_ptr<const MappedModelFile>, const ModelManifest&,
        AllocationGenerationSource&, HipModelResidencyPolicy);

    HipModelResidencyMode mode_ =
        HipModelResidencyMode::registered_file_mapping;
    std::vector<RegisteredRange> registered_ranges_;
    std::vector<CopiedRange> copied_ranges_;
    std::uint64_t registered_bytes_ = 0;
    std::uint64_t copied_bytes_ = 0;
    std::uint64_t generation_ = 0;
    const MappedModelFile* source_mapping_ = nullptr;
    std::uintptr_t source_base_ = 0;
    std::uint64_t source_bytes_ = 0, source_generation_ = 0;
    // Declared before bindings; release() explicitly clears it only after HIP
    // unregister/free, and shared snapshots pin both owners in that order.
    std::shared_ptr<const MappedModelFile> mapping_owner_;
    HipModelResidencyReservation reservation_;
    ModelRuntimeBinding binding_;
};

}  // namespace strix
