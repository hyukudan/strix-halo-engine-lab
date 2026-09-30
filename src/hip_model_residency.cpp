#include "strix/hip_model_residency.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace strix {
namespace {

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() -
            left) {
        throw std::overflow_error(
            "HIP model residency range overflow");
    }
    return left + right;
}

std::uint64_t page_size() {
#if defined(__linux__)
    const auto value =
        ::sysconf(_SC_PAGESIZE);
    if (value <= 0) {
        throw std::runtime_error(
            "cannot determine host page size");
    }
    return static_cast<std::uint64_t>(
        value);
#else
    return 4096;
#endif
}

std::uint64_t align_down(
    std::uint64_t value,
    std::uint64_t alignment) {
    return value -
           value % alignment;
}

std::uint64_t align_up(
    std::uint64_t value,
    std::uint64_t alignment) {
    const auto remainder =
        value % alignment;
    if (remainder == 0) {
        return value;
    }
    return checked_add(
        value,
        alignment - remainder);
}

bool should_promote(
    const ManifestTensor& tensor,
    const HipModelResidencyPolicy& policy) noexcept {
    if (tensor.placement.kind ==
        PlacementKind::remote) {
        return false;
    }
    if (tensor.residency ==
            ResidencyPolicy::file_backed_candidate &&
        !policy.promote_file_backed_candidates) {
        return false;
    }
    return true;
}

struct HostRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

std::vector<HostRange> build_registration_ranges(
    const MappedModelFile& mapped,
    const ModelManifest& manifest,
    const HipModelResidencyPolicy& policy) {
    const auto page =
        page_size();
    std::vector<HostRange> ranges;

    for (const auto& tensor :
         manifest.tensors) {
        if (!should_promote(
                tensor,
                policy)) {
            continue;
        }
        if (tensor.absolute_offset >
                mapped.bytes() ||
            tensor.byte_size >
                mapped.bytes() -
                    tensor.absolute_offset) {
            throw std::invalid_argument(
                "manifest tensor exceeds mapped model file");
        }

        const auto begin =
            align_down(
                tensor.absolute_offset,
                page);
        const auto tensor_end =
            checked_add(
                tensor.absolute_offset,
                tensor.byte_size);
        const auto end =
            std::min(
                align_up(
                    tensor_end,
                    page),
                align_up(
                    mapped.bytes(),
                    page));
        if (end <= begin) {
            throw std::logic_error(
                "registered tensor range is empty");
        }
        ranges.push_back(
            HostRange{
                begin,
                end,
            });
    }

    std::sort(
        ranges.begin(),
        ranges.end(),
        [](const auto& left,
           const auto& right) {
            if (left.begin != right.begin) {
                return left.begin <
                       right.begin;
            }
            return left.end <
                   right.end;
        });

    std::vector<HostRange> merged;
    for (const auto range :
         ranges) {
        if (merged.empty()) {
            merged.push_back(range);
            continue;
        }

        auto& last =
            merged.back();
        const auto maximum_gap_end =
            checked_add(
                last.end,
                policy.coalesce_gap_bytes);
        if (range.begin <=
            maximum_gap_end) {
            last.end =
                std::max(
                    last.end,
                    range.end);
        } else {
            merged.push_back(range);
        }
    }
    return merged;
}


std::vector<HostRange> build_copy_ranges(
    const MappedModelFile& mapped,
    const ModelManifest& manifest,
    const HipModelResidencyPolicy& policy) {
    std::vector<HostRange> ranges;
    for (const auto& tensor : manifest.tensors) {
        if (!should_promote(tensor, policy)) {
            continue;
        }
        if (tensor.absolute_offset > mapped.bytes() ||
            tensor.byte_size >
                mapped.bytes() - tensor.absolute_offset) {
            throw std::invalid_argument(
                "manifest tensor exceeds mapped model file");
        }
        ranges.push_back(
            HostRange{
                tensor.absolute_offset,
                checked_add(
                    tensor.absolute_offset,
                    tensor.byte_size),
            });
    }

    std::sort(
        ranges.begin(),
        ranges.end(),
        [](const auto& left,
           const auto& right) {
            if (left.begin != right.begin) {
                return left.begin < right.begin;
            }
            return left.end < right.end;
        });

    std::vector<HostRange> merged;
    for (const auto range : ranges) {
        if (merged.empty()) {
            merged.push_back(range);
            continue;
        }
        auto& last = merged.back();
        const auto maximum_gap_end =
            checked_add(
                last.end,
                policy.coalesce_gap_bytes);
        if (range.begin <= maximum_gap_end) {
            last.end =
                std::max(
                    last.end,
                    range.end);
        } else {
            merged.push_back(range);
        }
    }
    return merged;
}

}  // namespace

HipModelResidency::HipModelResidency(
    HipModelResidency&& other) noexcept
    : mode_(other.mode_),
      registered_ranges_(
          std::move(
              other.registered_ranges_)),
      copied_ranges_(
          std::move(
              other.copied_ranges_)),
      registered_bytes_(
          other.registered_bytes_),
      copied_bytes_(
          other.copied_bytes_),
      binding_(
          std::move(
              other.binding_)) {
    other.registered_bytes_ = 0;
    other.copied_bytes_ = 0;
    other.registered_ranges_.clear();
}

HipModelResidency&
HipModelResidency::operator=(
    HipModelResidency&& other) noexcept {
    if (this != &other) {
        release();
        mode_ = other.mode_;
        registered_ranges_ =
            std::move(
                other.registered_ranges_);
        copied_ranges_ =
            std::move(
                other.copied_ranges_);
        registered_bytes_ =
            other.registered_bytes_;
        copied_bytes_ =
            other.copied_bytes_;
        binding_ =
            std::move(
                other.binding_);

        other.registered_bytes_ = 0;
        other.copied_bytes_ = 0;
        other.registered_ranges_.clear();
    }
    return *this;
}

HipModelResidency::~HipModelResidency() {
    release();
}

HipModelResidency HipModelResidency::create(
    const MappedModelFile& mapped,
    const ModelManifest& manifest,
    AllocationGenerationSource& generations,
    HipModelResidencyPolicy policy) {
    if (!mapped.valid() ||
        !mapped.content_identity().valid()) {
        throw std::invalid_argument(
            "HIP model residency requires a valid mapped model");
    }
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(
                manifest)) {
        throw std::invalid_argument(
            "HIP model residency requires a valid manifest");
    }
    if (mapped.bytes() >
        static_cast<std::uint64_t>(
            std::numeric_limits<
                std::size_t>::max())) {
        throw std::overflow_error(
            "mapped model size does not fit size_t");
    }

    HipModelResidency result;
    result.mode_ =
        policy.mode;
    const auto generation =
        generations.next();

    result.binding_ =
        mapped.build_runtime_binding(
            manifest);

    if (policy.mode ==
        HipModelResidencyMode::
            registered_file_mapping) {
        const auto ranges =
            build_registration_ranges(
                mapped,
                manifest,
                policy);

        try {
            result.registered_ranges_.
                reserve(
                    ranges.size());

            for (const auto range :
                 ranges) {
                const auto bytes =
                    range.end -
                    range.begin;
                const auto host =
                    mapped.base_address() +
                    static_cast<
                        std::uintptr_t>(
                            range.begin);

                check_hip(
                    hipHostRegister(
                        reinterpret_cast<void*>(
                            host),
                        static_cast<std::size_t>(
                            bytes),
                        hipHostRegisterMapped),
                    "hipHostRegister(model range)");

                void* device = nullptr;
                try {
                    check_hip(
                        hipHostGetDevicePointer(
                            &device,
                            reinterpret_cast<void*>(
                                host),
                            0),
                        "hipHostGetDevicePointer(model range)");
                } catch (...) {
                    static_cast<void>(
                        hipHostUnregister(
                            reinterpret_cast<void*>(
                                host)));
                    throw;
                }

                result.registered_ranges_.
                    push_back(
                        RegisteredRange{
                            host,
                            reinterpret_cast<
                                std::uintptr_t>(
                                    device),
                            bytes,
                        });
                result.registered_bytes_ =
                    checked_add(
                        result.registered_bytes_,
                        bytes);
            }

            for (std::size_t index = 0;
                 index <
                    manifest.tensors.size();
                 ++index) {
                const auto& tensor =
                    manifest.tensors[index];
                auto& runtime =
                    result.binding_.
                        tensors[index];

                if (!should_promote(
                        tensor,
                        policy)) {
                    continue;
                }

                const auto host =
                    mapped.base_address() +
                    static_cast<
                        std::uintptr_t>(
                            tensor.absolute_offset);
                const auto found =
                    std::find_if(
                        result.registered_ranges_.begin(),
                        result.registered_ranges_.end(),
                        [&](const RegisteredRange& range) {
                            if (host < range.host_base) {
                                return false;
                            }
                            const auto offset =
                                static_cast<std::uint64_t>(
                                    host - range.host_base);
                            return offset <= range.bytes &&
                                   tensor.byte_size <=
                                       range.bytes - offset;
                        });
                if (found ==
                    result.registered_ranges_.end()) {
                    throw std::logic_error(
                        "promoted tensor is not covered by a registered range");
                }

                const auto offset =
                    static_cast<std::uintptr_t>(
                        host -
                        found->host_base);
                runtime.storage =
                    RuntimeTensorStorage::
                        registered_file_mapped;
                runtime.address =
                    found->device_base +
                    offset;
                runtime.allocation_generation =
                    generation;
            }
        } catch (...) {
            result.release();
            throw;
        }
    } else {
        const auto ranges =
            build_copy_ranges(
                mapped,
                manifest,
                policy);

        try {
            result.copied_ranges_.reserve(
                ranges.size());

            for (const auto range : ranges) {
                const auto bytes =
                    range.end - range.begin;
                void* device = nullptr;
                check_hip(
                    hipMalloc(
                        &device,
                        static_cast<std::size_t>(
                            bytes)),
                    "hipMalloc(model range copy)");
                try {
                    check_hip(
                        hipMemcpy(
                            device,
                            reinterpret_cast<const void*>(
                                mapped.base_address() +
                                static_cast<std::uintptr_t>(
                                    range.begin)),
                            static_cast<std::size_t>(
                                bytes),
                            hipMemcpyHostToDevice),
                        "hipMemcpy(model range copy)");
                } catch (...) {
                    static_cast<void>(
                        hipFree(device));
                    throw;
                }

                result.copied_ranges_.push_back(
                    CopiedRange{
                        range.begin,
                        reinterpret_cast<std::uintptr_t>(
                            device),
                        bytes,
                    });
                result.copied_bytes_ =
                    checked_add(
                        result.copied_bytes_,
                        bytes);
            }

            for (std::size_t index = 0;
                 index < manifest.tensors.size();
                 ++index) {
                const auto& tensor =
                    manifest.tensors[index];
                auto& runtime =
                    result.binding_.tensors[index];

                if (!should_promote(
                        tensor,
                        policy)) {
                    continue;
                }

                const auto found =
                    std::find_if(
                        result.copied_ranges_.begin(),
                        result.copied_ranges_.end(),
                        [&](const CopiedRange& range) {
                            if (tensor.absolute_offset <
                                range.file_offset) {
                                return false;
                            }
                            const auto offset =
                                tensor.absolute_offset -
                                range.file_offset;
                            return offset <= range.bytes &&
                                   tensor.byte_size <=
                                       range.bytes - offset;
                        });
                if (found ==
                    result.copied_ranges_.end()) {
                    throw std::logic_error(
                        "promoted tensor is not covered by a copied range");
                }

                runtime.storage =
                    RuntimeTensorStorage::resident;
                runtime.address =
                    found->device_base +
                    static_cast<std::uintptr_t>(
                        tensor.absolute_offset -
                        found->file_offset);
                runtime.allocation_generation =
                    generation;
            }
        } catch (...) {
            result.release();
            throw;
        }

    validate_model_runtime_binding(
        manifest,
        result.binding_);
    return result;
}

void HipModelResidency::release() noexcept {
    for (const auto& range :
         registered_ranges_) {
        static_cast<void>(
            hipHostUnregister(
                reinterpret_cast<void*>(
                    range.host_base)));
    }
    registered_ranges_.clear();

    for (const auto& range :
         copied_ranges_) {
        static_cast<void>(
            hipFree(
                reinterpret_cast<void*>(
                    range.device_base)));
    }
    copied_ranges_.clear();
    registered_bytes_ = 0;
    copied_bytes_ = 0;
    binding_ = {};
}

}  // namespace strix
