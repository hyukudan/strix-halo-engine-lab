#pragma once

#include "strix/mapped_model_file.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/runtime_allocator.hpp"

#include <cstdint>
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
};

struct CopyResidencyRange {
    std::uint64_t begin = 0, end = 0;
};
// Pure layout shared by allocation and pre-allocation kernel eligibility.
[[nodiscard]] std::vector<CopyResidencyRange> build_copy_residency_ranges(
    const ModelManifest&, const HipModelResidencyPolicy&);
[[nodiscard]] std::uint64_t copy_residency_tensor_offset(
    const ManifestTensor&, const std::vector<CopyResidencyRange>&);

class HipModelResidency {
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
    };

    void release() noexcept;

    HipModelResidencyMode mode_ =
        HipModelResidencyMode::registered_file_mapping;
    std::vector<RegisteredRange> registered_ranges_;
    std::vector<CopiedRange> copied_ranges_;
    std::uint64_t registered_bytes_ = 0;
    std::uint64_t copied_bytes_ = 0;
    std::uint64_t generation_ = 0;
    ModelRuntimeBinding binding_;
};

}  // namespace strix
