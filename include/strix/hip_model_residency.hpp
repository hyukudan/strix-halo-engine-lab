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

private:
    struct RegisteredRange {
        std::uintptr_t host_base = 0;
        std::uintptr_t device_base = 0;
        std::uint64_t bytes = 0;
    };

    void release() noexcept;

    HipModelResidencyMode mode_ =
        HipModelResidencyMode::registered_file_mapping;
    std::vector<RegisteredRange> registered_ranges_;
    std::uintptr_t copied_device_base_ = 0;
    std::uint64_t copied_device_bytes_ = 0;
    std::uint64_t registered_bytes_ = 0;
    std::uint64_t copied_bytes_ = 0;
    ModelRuntimeBinding binding_;
};

}  // namespace strix
