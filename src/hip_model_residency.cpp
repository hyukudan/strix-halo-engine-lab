#include "strix/hip_model_residency.hpp"
#include "strix/hip_setup_quiescence.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace strix {
namespace {
// Process-local allocator lease IDs, never GPU/host virtual addresses.
std::atomic<std::uint64_t> next_residency_identity{1};
std::uint64_t residency_identity() {
    auto value = next_residency_identity.load(std::memory_order_relaxed);
    for (;;) {
        if (!value || value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("HIP residency lease identity exhausted");
        if (next_residency_identity.compare_exchange_weak(value, value + 1U, std::memory_order_relaxed)) return value;
    }
}

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
    const auto planned=build_copy_residency_ranges(manifest,policy);
    std::vector<HostRange> ranges;
    for(const auto range:planned){
        if(range.begin>mapped.bytes() || range.end>mapped.bytes())throw std::invalid_argument("manifest tensor exceeds mapped model file");
        ranges.push_back({range.begin,range.end});
    }
    return ranges;
}

}  // namespace

HipModelResidencyReservation plan_hip_model_residency_reservation(const MappedModelFile& mapped,
    const ModelManifest& manifest, HipModelResidencyPolicy policy) {
    if (!mapped.valid() || !mapped.content_identity().valid() || !mapped.allocation_generation() ||
        !manifest.layout_identity || manifest.layout_identity != model_manifest_layout_identity(manifest))
        throw std::invalid_argument("HIP residency reservation requires live mapping and valid manifest");
    if (policy.mode != HipModelResidencyMode::registered_file_mapping && policy.mode != HipModelResidencyMode::resident_copy)
        throw std::invalid_argument("HIP model residency mode is invalid");
    HipModelResidencyReservation result;
    result.mapped_reserved_bytes = align_up(mapped.bytes(), page_size());
    const auto ranges = policy.mode == HipModelResidencyMode::registered_file_mapping
        ? build_registration_ranges(mapped, manifest, policy) : build_copy_ranges(mapped, manifest, policy);
    for (const auto range : ranges) {
        if (policy.mode == HipModelResidencyMode::registered_file_mapping)
            result.registered_alias_bytes = checked_add(result.registered_alias_bytes, range.end - range.begin);
        else result.copied_reserved_bytes = checked_add(result.copied_reserved_bytes, range.end - range.begin);
    }
    // Registration aliases the source mapping; only a real copy adds backing.
    result.retained_reserved_bytes = checked_add(result.mapped_reserved_bytes, result.copied_reserved_bytes);
    // No explicit temporary payload allocation is owned by this setup path.
    // Driver-internal memory and physical-page residency remain unobserved.
    result.setup_peak_reserved_bytes = result.retained_reserved_bytes;
    if (policy.reservation_limits.retained_bytes && result.retained_reserved_bytes > *policy.reservation_limits.retained_bytes)
        throw std::invalid_argument("HIP residency retained reservation exceeds explicit budget");
    if (policy.reservation_limits.setup_peak_bytes && result.setup_peak_reserved_bytes > *policy.reservation_limits.setup_peak_bytes)
        throw std::invalid_argument("HIP residency setup peak reservation exceeds explicit budget");
    return result;
}

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
      generation_(other.generation_),
      source_mapping_(other.source_mapping_), source_base_(other.source_base_),
      source_bytes_(other.source_bytes_), source_generation_(other.source_generation_),
      mapping_owner_(std::move(other.mapping_owner_)), reservation_(other.reservation_),
      binding_(
          std::move(
              other.binding_)) {
    other.registered_bytes_ = 0;
    other.copied_bytes_ = 0;
    other.generation_ = 0;
    other.source_mapping_ = nullptr; other.source_base_ = 0;
    other.source_bytes_ = 0; other.source_generation_ = 0; other.reservation_ = {};
    other.registered_ranges_.clear();
    other.copied_ranges_.clear();
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
        generation_ = other.generation_;
        source_mapping_ = other.source_mapping_; source_base_ = other.source_base_;
        source_bytes_ = other.source_bytes_; source_generation_ = other.source_generation_;
        mapping_owner_ = std::move(other.mapping_owner_); reservation_ = other.reservation_;
        binding_ =
            std::move(
                other.binding_);

        other.registered_bytes_ = 0;
        other.copied_bytes_ = 0;
        other.generation_ = 0;
        other.source_mapping_ = nullptr; other.source_base_ = 0;
        other.source_bytes_ = 0; other.source_generation_ = 0; other.reservation_ = {};
        other.registered_ranges_.clear();
        other.copied_ranges_.clear();
    }
    return *this;
}

HipModelResidency::~HipModelResidency() {
    release();
}

HipModelResidency HipModelResidency::create(std::shared_ptr<const MappedModelFile> mapped,
    const ModelManifest& manifest, AllocationGenerationSource& generations, HipModelResidencyPolicy policy) {
    if (!mapped) throw std::invalid_argument("HIP residency requires an actual shared mapping owner");
    // Pin before the first registration, including partial-setup cleanup.
    const auto& source = *mapped;
    return create_impl(source, std::move(mapped), manifest, generations, policy);
}

UmaOwnerInventory HipModelResidency::owner_inventory(std::shared_ptr<const HipModelResidency> actual_owner,
    std::shared_ptr<const MappedModelFile> mapped_owner, UmaMemoryObservation observation) const {
    const auto self = weak_from_this().lock();
    if (!mapped_owner) mapped_owner = mapping_owner_;
    if (!generation_ || !self || actual_owner.get() != this || actual_owner.owner_before(self) || self.owner_before(actual_owner) ||
        !mapping_owner_ || !mapped_owner ||
        mapped_owner.owner_before(mapping_owner_) || mapping_owner_.owner_before(mapped_owner) ||
        mapped_owner.get() != source_mapping_ ||
        !mapped_owner->valid() || mapped_owner->base_address() != source_base_ || mapped_owner->bytes() != source_bytes_ ||
        mapped_owner->allocation_generation() != source_generation_ ||
        mapped_owner->content_identity() != binding_.model_content_identity)
        throw std::invalid_argument("HIP residency inventory requires its unchanged actual mapping and residency owners");
    // Destruction order is explicit: unregister/free residency before unmapping.
    struct Owners {
        std::shared_ptr<const MappedModelFile> mapping;
        std::shared_ptr<const HipModelResidency> residency;
    };
    auto pin = std::make_shared<Owners>(Owners{mapped_owner, std::move(actual_owner)});
    // The allocation component is the retained mapping OBJECT lease, not its
    // CPU address. Its generation and strong pin prohibit reuse while live.
    const UmaBackingIdentity mapping_key{0x4849504d41503031ULL,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(mapped_owner.get())), source_generation_};
    std::vector<UmaOwnerView> views;
    views.push_back({next_uma_owner_view_identity(), mapping_key, UmaBackingKind::file_mapping, "weights.mapping",
        reservation_.mapped_reserved_bytes, 0, source_bytes_, pin, UmaAddressSpace::host, source_base_});
    for (const auto& range : registered_ranges_) {
        if (range.host_base < source_base_ || range.host_base - source_base_ > reservation_.mapped_reserved_bytes ||
            range.bytes > reservation_.mapped_reserved_bytes - (range.host_base - source_base_) ||
            !covers_device_range(range.device_base, range.bytes))
            throw std::invalid_argument("HIP residency registered alias exceeds its actual mapping owner");
        const auto offset = static_cast<std::uint64_t>(range.host_base - source_base_);
        views.push_back({next_uma_owner_view_identity(), mapping_key, UmaBackingKind::file_mapping, "weights.registered-host",
            reservation_.mapped_reserved_bytes, offset, range.bytes, pin, UmaAddressSpace::host, range.host_base});
        views.push_back({next_uma_owner_view_identity(), mapping_key, UmaBackingKind::file_mapping, "weights.registered-device-alias",
            reservation_.mapped_reserved_bytes, offset, range.bytes, pin, UmaAddressSpace::device, range.device_base});
    }
    for (const auto& range : copied_ranges_) {
        if (!range.allocation_identity || !covers_device_range(range.device_base, range.bytes))
            throw std::invalid_argument("HIP residency copy lacks its actual owned backing lease");
        views.push_back({next_uma_owner_view_identity(), {0x484950434f505931ULL, range.allocation_identity, generation_},
            UmaBackingKind::allocated_storage, "weights.copy", range.bytes, 0, range.bytes, pin,
            UmaAddressSpace::device, range.device_base});
    }
    auto result = build_uma_owner_inventory(views, std::move(observation));
    if (result.unique_backing_reserved_bytes != reservation_.retained_reserved_bytes)
        throw std::logic_error("HIP residency actual owner inventory differs from its declared reservation");
    return result;
}

bool HipModelResidency::covers_device_range(
    std::uintptr_t address, std::uint64_t bytes) const noexcept {
    constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
    if (!generation_ || !address || !bytes || bytes > maximum - address) return false;
    const auto contains = [&](std::uintptr_t base, std::uint64_t extent) noexcept {
        if (!base || !extent || extent > maximum - base || address < base) return false;
        const auto offset = static_cast<std::uint64_t>(address - base);
        return offset < extent && bytes <= extent - offset;
    };
    for (const auto& range : registered_ranges_)
        if (contains(range.device_base, range.bytes)) return true;
    for (const auto& range : copied_ranges_)
        if (contains(range.device_base, range.bytes)) return true;
    return false;
}

HipModelResidency HipModelResidency::create(
    const MappedModelFile& mapped,
    const ModelManifest& manifest,
    AllocationGenerationSource& generations,
    HipModelResidencyPolicy policy) {
    return create_impl(mapped, {}, manifest, generations, policy);
}

HipModelResidency HipModelResidency::create_impl(const MappedModelFile& mapped,
    std::shared_ptr<const MappedModelFile> mapping_owner, const ModelManifest& manifest,
    AllocationGenerationSource& generations, HipModelResidencyPolicy policy) {
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

    const auto reservation = plan_hip_model_residency_reservation(mapped, manifest, policy);
    HipModelResidency result;
    result.mapping_owner_ = std::move(mapping_owner);
    result.reservation_ = reservation;
    result.source_mapping_ = &mapped; result.source_base_ = mapped.base_address();
    result.source_bytes_ = mapped.bytes(); result.source_generation_ = mapped.allocation_generation();
    result.mode_ =
        policy.mode;
    const auto generation =
        generations.next();
    result.generation_ = generation;

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

                // Track the registration immediately: a device-alias query
                // failure must use the same owner-preserving cleanup path.
                result.registered_ranges_.
                    push_back(
                        RegisteredRange{
                            host,
                            0,
                            bytes,
                        });
                void* device = nullptr;
                check_hip(hipHostGetDevicePointer(&device, reinterpret_cast<void*>(host), 0),
                    "hipHostGetDevicePointer(model range)");
                result.registered_ranges_.back().device_base = reinterpret_cast<std::uintptr_t>(device);
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

        const HipSetupQuiescenceGuard setup_guard;

        try {
            result.copied_ranges_.reserve(
                ranges.size());

            for (const auto range : ranges) {
                const auto bytes =
                    range.end - range.begin;
                const auto allocation_identity = residency_identity();
                void* device = nullptr;
                check_hip(
                    hipMalloc(
                        &device,
                        static_cast<std::size_t>(
                            bytes)),
                    "hipMalloc(model range copy)");
                try {
                    setup_guard.run([&] {
                        check_hip(
                            hipMemcpy(
                                device,
                                reinterpret_cast<const void*>(
                                    mapped.base_address() +
                                    static_cast<std::uintptr_t>(range.begin)),
                                static_cast<std::size_t>(bytes),
                                hipMemcpyHostToDevice),
                            "hipMemcpy(model range copy)");
                    });
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
                        allocation_identity,
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
    }

    validate_model_runtime_binding(
        manifest,
        result.binding_);
    return result;
}

void HipModelResidency::release() noexcept {
    bool unregister_failed = false;
    for (const auto& range :
         registered_ranges_) {
        if (hipHostUnregister(reinterpret_cast<void*>(range.host_base)) != hipSuccess)
            unregister_failed = true;
    }
    if (unregister_failed && mapping_owner_) {
        // Quarantine only this failed strong lease. Do not unmap storage while
        // the runtime may still retain a registration. Allocation failure is
        // fail-stop rather than releasing a possibly live backing.
        auto* retained = new (std::nothrow) std::shared_ptr<const MappedModelFile>(std::move(mapping_owner_));
        if (!retained) std::terminate();
        static_cast<void>(retained);
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
    generation_ = 0;
    binding_ = {};
    source_mapping_ = nullptr; source_base_ = 0; source_bytes_ = 0; source_generation_ = 0;
    reservation_ = {};
    mapping_owner_.reset(); // Only after every unregister/free attempt above.
}

}  // namespace strix
