#include "strix/hip_model_residency.hpp"

#include "strix/model_manifest.hpp"
#include "strix/runtime_allocator.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            hipGetErrorString(error));
    }
}

std::filesystem::path fixture_path() {
#if defined(__linux__)
    const auto suffix =
        std::to_string(
            static_cast<unsigned long long>(
                ::getpid()));
#else
    const auto suffix = "portable";
#endif
    return std::filesystem::temp_directory_path() /
           ("strix-hip-model-residency-" +
            suffix + ".bin");
}

void write_fixture(
    const std::filesystem::path& path) {
    constexpr std::size_t bytes = 16U * 1024U;
    std::vector<std::uint8_t> data(bytes);
    for (std::size_t index = 0;
         index < data.size();
         ++index) {
        data[index] =
            static_cast<std::uint8_t>(
                (index * 37U + 11U) &
                0xffU);
    }

    std::ofstream output(
        path,
        std::ios::binary |
        std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot create residency fixture");
    }
    output.write(
        reinterpret_cast<const char*>(
            data.data()),
        static_cast<std::streamsize>(
            data.size()));
    if (!output) {
        throw std::runtime_error(
            "cannot write residency fixture");
    }
}

strix::ModelManifest fixture_manifest() {
    strix::ModelManifest manifest;
    manifest.architecture =
        "residency-fixture";

    const auto add =
        [&](std::string name,
            std::uint64_t offset,
            strix::ResidencyPolicy residency) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.dimensions = {1024};
            tensor.ggml_type = 0;
            tensor.byte_size = 1024;
            tensor.absolute_offset =
                offset;
            tensor.role =
                strix::TensorRole::
                    dense_projection;
            tensor.residency =
                residency;
            manifest.total_tensor_bytes +=
                tensor.byte_size;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add(
        "dense.a",
        4096,
        strix::ResidencyPolicy::resident);
    add(
        "ple.table",
        8192,
        strix::ResidencyPolicy::
            file_backed_candidate);
    add(
        "dense.b",
        12288,
        strix::ResidencyPolicy::resident);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);
    return manifest;
}

std::uint8_t expected_byte(
    std::uint64_t file_offset) {
    return static_cast<std::uint8_t>(
        (file_offset * 37ULL + 11ULL) &
        0xffULL);
}

void verify_binding(
    const strix::MappedModelFile& mapped,
    const strix::ModelRuntimeBinding& binding,
    strix::HipModelResidencyMode mode) {
    expect(
        binding.tensors.size() == 3,
        "residency binding preserves tensor count");

    const auto promoted_storage =
        mode ==
                strix::HipModelResidencyMode::
                    registered_file_mapping
            ? strix::RuntimeTensorStorage::
                  registered_file_mapped
            : strix::RuntimeTensorStorage::
                  resident;

    expect(
        binding.tensors[0].storage ==
                promoted_storage &&
            binding.tensors[2].storage ==
                promoted_storage &&
            strix::runtime_tensor_gpu_visible(
                binding.tensors[0].storage) &&
            strix::runtime_tensor_gpu_visible(
                binding.tensors[2].storage),
        "dense tensors become GPU-visible under selected residency mode");

    expect(
        binding.tensors[1].storage ==
                strix::RuntimeTensorStorage::
                    file_mapped &&
            !strix::runtime_tensor_gpu_visible(
                binding.tensors[1].storage) &&
            binding.tensors[1].address ==
                mapped.base_address() +
                    8192U,
        "file-backed candidate remains ordinary host mapping");

    std::vector<std::uint8_t> copied(1024);
    check(
        hipMemcpy(
            copied.data(),
            reinterpret_cast<const void*>(
                binding.tensors[0].address),
            copied.size(),
            hipMemcpyDeviceToHost),
        "copy promoted tensor back");
    expect(
        copied.front() ==
                expected_byte(4096) &&
            copied.back() ==
                expected_byte(
                    4096 + 1023),
        "GPU-visible tensor bytes match mapped file contents");

    const auto* host_table =
        reinterpret_cast<
            const std::uint8_t*>(
                binding.tensors[1].address);
    expect(
        host_table[0] ==
                expected_byte(8192) &&
            host_table[1023] ==
                expected_byte(
                    8192 + 1023),
        "unpromoted file-backed tensor remains host-readable");
}

void shared_owner_inventory(const std::filesystem::path& path, const strix::ModelManifest& manifest) {
    for (const auto mode : {strix::HipModelResidencyMode::registered_file_mapping,
                           strix::HipModelResidencyMode::resident_copy}) {
        strix::AllocationGenerationSource generations;
        auto mapped = std::make_shared<strix::MappedModelFile>(
            strix::MappedModelFile::open_read_only(path, "shared-residency-fixture", generations));
        strix::HipModelResidencyPolicy policy; policy.mode = mode;
        const auto planned = strix::plan_hip_model_residency_reservation(*mapped, manifest, policy);
        expect(planned.retained_reserved_bytes == planned.mapped_reserved_bytes + planned.copied_reserved_bytes &&
            planned.setup_peak_reserved_bytes == planned.retained_reserved_bytes &&
            (mode != strix::HipModelResidencyMode::registered_file_mapping || planned.copied_reserved_bytes == 0),
            "registration aliases do not add backing reservations; real copies do");
        {
            auto impossible = policy; impossible.reservation_limits.retained_bytes = 0;
            strix::AllocationGenerationSource before_allocation(100000);
            bool rejected = false;
            try { static_cast<void>(strix::HipModelResidency::create(mapped, manifest, before_allocation, impossible)); }
            catch (const std::invalid_argument& error) {
                rejected = std::string_view(error.what()) == "HIP residency retained reservation exceeds explicit budget";
            }
            expect(rejected && before_allocation.next() == 100000,
                "insufficient reservation budget rejects before allocation-generation admission");
            impossible.reservation_limits = {}; impossible.reservation_limits.setup_peak_bytes = planned.setup_peak_reserved_bytes - 1U;
            rejected = false;
            try { static_cast<void>(strix::plan_hip_model_residency_reservation(*mapped, manifest, impossible)); }
            catch (const std::invalid_argument& error) {
                rejected = std::string_view(error.what()) == "HIP residency setup peak reservation exceeds explicit budget";
            }
            expect(rejected, "setup peak budget is independent from the retained budget");
        }
        policy.reservation_limits = {planned.retained_reserved_bytes, planned.setup_peak_reserved_bytes};
        auto owner = std::make_shared<strix::HipModelResidency>(
            strix::HipModelResidency::create(mapped, manifest, generations, policy));
        auto inventory = owner->owner_inventory(owner);
        expect(inventory.unique_backing_reserved_bytes == planned.retained_reserved_bytes &&
            inventory.known_file_mapping_bytes == planned.mapped_reserved_bytes &&
            inventory.known_allocated_storage_bytes == owner->copied_bytes() &&
            !inventory.observation.process_rss_bytes && !inventory.observation.gtt_reported_bytes,
            "actual residency backing inventory agrees with plan without inventing system observations");
        std::size_t aliases = 0;
        for (const auto& view : inventory.pinned_views) {
            expect(view.address != 0 && view.backing.generation != 0 && view.allocation_pin,
                "actual alias carries live address, backing generation and owners");
            if (view.address_space == strix::UmaAddressSpace::device) {
                ++aliases;
                expect(owner->covers_device_range(view.address, view.view_bytes),
                    "inventory device view is covered by the actual residency owner");
            }
        }
        expect(aliases != 0, "inventory exports actual HIP device aliases rather than host address guesses");
        if (mode == strix::HipModelResidencyMode::registered_file_mapping)
            expect(inventory.known_alias_view_bytes == owner->registered_bytes() * 2U,
                "registered host/device views share one source backing and are counted once");
        // Export IDs are distinct across snapshots so inventories can be joined
        // while the source backing lease still deduplicates.
        auto second = owner->owner_inventory(owner);
        auto joined = inventory.pinned_views;
        joined.insert(joined.end(), second.pinned_views.begin(), second.pinned_views.end());
        expect(strix::build_uma_owner_inventory(joined).unique_backing_reserved_bytes == planned.retained_reserved_bytes,
            "repeated actual owner exports duplicate logical aliases, not physical backing reservations");
        second = {}; joined.clear();
        auto unrelated = std::make_shared<strix::MappedModelFile>(
            strix::MappedModelFile::open_read_only(path, "other-mapping", generations));
        bool rejected = false;
        try { static_cast<void>(owner->owner_inventory(owner, unrelated)); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "equal file content cannot impersonate the retained mapping lease");
        unrelated.reset();
        auto fake_mapping_pin = std::shared_ptr<const strix::MappedModelFile>(mapped.get(), [](const auto*) {});
        rejected = false;
        try { static_cast<void>(owner->owner_inventory(owner, fake_mapping_pin)); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "same mapping pointer with a different control block is not its owning lease");
        fake_mapping_pin.reset();
        auto fake_residency_pin = std::shared_ptr<const strix::HipModelResidency>(owner.get(), [](const auto*) {});
        rejected = false;
        try { static_cast<void>(owner->owner_inventory(fake_residency_pin)); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "same residency pointer with a different control block is not its owning lease");
        fake_residency_pin.reset();
        inventory = {}; // Quiescent; no snapshot may observe mutable owner transfer.
        const auto original_generation = owner->generation();
        auto moved = std::make_shared<strix::HipModelResidency>(std::move(*owner));
        rejected = false;
        try { static_cast<void>(owner->owner_inventory(owner)); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && owner->generation() == 0 && moved->generation() == original_generation,
            "move preserves the live allocation generation and invalidates the old owner");
        owner = std::move(moved);
        inventory = owner->owner_inventory(owner);
        std::weak_ptr<const strix::MappedModelFile> mapping_weak = mapped;
        std::weak_ptr<const strix::HipModelResidency> residency_weak = owner;
        mapped.reset(); owner.reset();
        expect(!mapping_weak.expired() && !residency_weak.expired(), "inventory keeps actual mapping and HIP consumer owners alive");
        verify_binding(*mapping_weak.lock(), residency_weak.lock()->binding(), mode);
        inventory = {};
        expect(mapping_weak.expired() && residency_weak.expired(), "inventory releases residency before the final mapping lease");
    }
    // A borrowed API retains its original lifetime contract; it cannot claim
    // stronger mapping ownership just because a snapshot is requested later.
    strix::AllocationGenerationSource generations;
    auto mapped = std::make_shared<strix::MappedModelFile>(
        strix::MappedModelFile::open_read_only(path, "borrowed-inventory-negative", generations));
    strix::HipModelResidencyPolicy policy; policy.mode = strix::HipModelResidencyMode::resident_copy;
    auto borrowed = std::make_shared<strix::HipModelResidency>(strix::HipModelResidency::create(*mapped, manifest, generations, policy));
    bool rejected = false;
    try { static_cast<void>(borrowed->owner_inventory(borrowed, mapped)); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "borrowed residency cannot manufacture a strong mapping lease");
    policy.mode = static_cast<strix::HipModelResidencyMode>(99);
    rejected = false;
    try { static_cast<void>(strix::plan_hip_model_residency_reservation(*mapped, manifest, policy)); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "invalid residency mode cannot silently allocate a copied model");
}

}  // namespace

int main() try {
    const auto path =
        fixture_path();
    write_fixture(path);

    try {
        strix::AllocationGenerationSource
            generations;
        auto mapped =
            strix::MappedModelFile::
                open_read_only(
                    path,
                    "residency-fixture",
                    generations);
        const auto manifest =
            fixture_manifest();

        {
            strix::HipModelResidencyPolicy
                policy;
            policy.mode =
                strix::HipModelResidencyMode::
                    resident_copy;
            auto residency =
                strix::HipModelResidency::create(
                    mapped,
                    manifest,
                    generations,
                    policy);

            expect(
                residency.copied_bytes() ==
                    2048,
                "resident-copy mode copies only promoted tensor bytes");
            expect(
                residency.registered_bytes() ==
                    0,
                "resident-copy mode registers no host pages");
            verify_binding(
                mapped,
                residency.binding(),
                policy.mode);
        }
        shared_owner_inventory(path, manifest);

        {
            strix::HipModelResidencyPolicy
                policy;
            policy.mode =
                strix::HipModelResidencyMode::
                    registered_file_mapping;
            auto residency =
                strix::HipModelResidency::create(
                    mapped,
                    manifest,
                    generations,
                    policy);

            expect(
                residency.registered_bytes() >=
                    8192,
                "registered mode accounts page-granular promoted ranges");
            expect(
                residency.copied_bytes() ==
                    0,
                "registered mode copies no model bytes");
            verify_binding(
                mapped,
                residency.binding(),
                policy.mode);
        }
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }

    std::filesystem::remove(path);

    if (failures == 0) {
        std::cout
            << "HIP model residency tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP model residency tests: "
        << error.what() << '\n';
    return 2;
}
