#include "strix/hip_model_residency.hpp"

#include "strix/model_manifest.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_model_residency_diagnostic.hpp"
#include <algorithm>
#include <array>

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
#include <utility>
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
    const std::filesystem::path& path,
    std::size_t bytes = 16U * 1024U) {
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
            tensor.dimensions = {256};
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


#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
strix::ModelManifest selective_manifest() {
    auto manifest = fixture_manifest();
    manifest.tensors[0].role = strix::TensorRole::normalization;
    manifest.tensors[2].role = strix::TensorRole::output_head;
    // Keep valid F32 rows while exercising nonzero page-relative offsets.
    manifest.tensors[0].absolute_offset += 12U;
    manifest.tensors[2].absolute_offset += 28U;
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
    return manifest;
}
strix::HipModelResidencyPolicy selective_policy(strix::HipModelCopyTier tier) {
    strix::HipModelResidencyPolicy policy;
    policy.copy_tier = tier;
    policy.copy_payload_bytes = tier == strix::HipModelCopyTier::normalization ? 1024U : 2048U;
    policy.reservation_limits = {65536U, 65536U};
    return policy;
}
void selective_admission(const std::shared_ptr<const strix::MappedModelFile>& mapped,
    const strix::ModelManifest& manifest) {
    namespace diag = strix::model_residency_diagnostic;
    const auto base = selective_policy(strix::HipModelCopyTier::hot_nonexpert);
    std::vector<strix::HipModelResidencyPolicy> invalid;
    auto value = base; value.copy_payload_bytes.reset(); invalid.push_back(value);
    value = base; value.reservation_limits.retained_bytes.reset(); invalid.push_back(value);
    value = base; value.reservation_limits.setup_peak_bytes.reset(); invalid.push_back(value);
    value = base; value.copy_payload_bytes = 2049U; invalid.push_back(value);
    value = base; value.copy_tier = static_cast<strix::HipModelCopyTier>(99); invalid.push_back(value);
    value = base; value.mode = strix::HipModelResidencyMode::resident_copy; invalid.push_back(value);
    value = base; value.promote_file_backed_candidates = true; invalid.push_back(value);
    const auto planned = strix::plan_hip_model_residency_reservation(*mapped, manifest, base);
    value = base; value.reservation_limits.retained_bytes = planned.retained_reserved_bytes - 1U; invalid.push_back(value);
    value = base; value.reservation_limits.setup_peak_bytes = planned.setup_peak_reserved_bytes - 1U; invalid.push_back(value);
    value = {}; value.copy_payload_bytes = 1U; invalid.push_back(value);
    for (const auto& policy : invalid) {
        diag::set_fault_plan(); strix::AllocationGenerationSource generation(100000);
        bool rejected = false;
        try { static_cast<void>(strix::HipModelResidency::create(mapped, manifest, generation, policy)); }
        catch (const std::exception&) { rejected = true; }
        const auto calls = diag::stats();
        expect(rejected && generation.next() == 100000 && calls.registrations == 0 && calls.allocations == 0 &&
            calls.alias_queries == 0 && calls.copies == 0, "selective invalid metadata/budgets reject before generation and HIP mutation");
    }
    diag::set_fault_plan(); strix::AllocationGenerationSource generation(100000);
    bool rejected = false;
    try { static_cast<void>(strix::HipModelResidency::create(*mapped, manifest, generation, base)); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected && generation.next() == 100000 && diag::stats().registrations == 0,
        "borrowed selective source cannot manufacture a strong mapping pin");
}
void selective_owned_and_faults(const std::filesystem::path& path) {
    namespace diag = strix::model_residency_diagnostic;
    const auto manifest = selective_manifest();
    strix::AllocationGenerationSource generations;
    std::shared_ptr<const strix::MappedModelFile> mapped = std::make_shared<strix::MappedModelFile>(
        strix::MappedModelFile::open_read_only(path, "selective-source", generations));
    selective_admission(mapped, manifest);
    for (const auto tier : {strix::HipModelCopyTier::normalization, strix::HipModelCopyTier::hot_nonexpert}) {
        diag::set_fault_plan(); const auto policy = selective_policy(tier);
        const auto plan = strix::build_selective_copy_plan(manifest, policy, mapped->bytes(), 4096U);
        const auto reservation = strix::plan_hip_model_residency_reservation(*mapped, manifest, policy);
        auto owner = std::make_shared<strix::HipModelResidency>(
            strix::HipModelResidency::create(mapped, manifest, generations, policy));
        auto inventory = owner->owner_inventory(owner);
        expect(reservation.setup_staging_reserved_bytes == 0 && diag::stats().staging_allocations == 0 &&
            reservation.setup_peak_reserved_bytes == reservation.retained_reserved_bytes,
            "normalization and hot tiers keep direct copies without temporary pinned backing");
        expect(owner->copy_tier() == tier && owner->mode() == strix::HipModelResidencyMode::registered_file_mapping &&
            owner->copied_bytes() == plan.arena_bytes && reservation.copy_payload_bytes == *policy.copy_payload_bytes &&
            inventory.known_allocated_storage_bytes == plan.arena_bytes && inventory.known_backings.size() == 2U &&
            inventory.unique_backing_reserved_bytes == reservation.mapped_reserved_bytes + plan.arena_bytes,
            "selective one-arena backing is distinct from unchanged registered aliases");
        {
            auto fake_mapping = std::shared_ptr<const strix::MappedModelFile>(mapped.get(), [](const auto*) {});
            auto fake_owner = std::shared_ptr<const strix::HipModelResidency>(owner.get(), [](const auto*) {});
            bool rejected_mapping = false, rejected_owner = false;
            try { static_cast<void>(owner->owner_inventory(owner, fake_mapping)); }
            catch (const std::invalid_argument&) { rejected_mapping = true; }
            try { static_cast<void>(owner->owner_inventory(fake_owner)); }
            catch (const std::invalid_argument&) { rejected_owner = true; }
            expect(rejected_mapping && rejected_owner, "selective inventory rejects borrowed fake control-block pins");
        }
        std::uintptr_t arena = 0;
        for (const auto& view : inventory.pinned_views) if (view.category == "weights.copy") {
            const auto current = view.address - view.view_offset;
            if (!arena) arena = current;
            expect(arena == current && view.allocation_bytes == plan.arena_bytes,
                "all selective range views share exactly one real arena lease");
        }
        for (const auto& range : plan.ranges) {
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(range.file_end - range.file_begin));
            check(hipMemcpy(bytes.data(), reinterpret_cast<const void*>(arena + range.arena_offset), bytes.size(),
                hipMemcpyDeviceToHost), "read selective arena including page padding");
            bool equal = true;
            for (std::size_t j = 0; j < bytes.size(); ++j) equal = equal && bytes[j] == expected_byte(range.file_begin + j);
            expect(equal, "every copied payload and actual file padding byte remains exact");
        }
        for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
            const auto& tensor = manifest.tensors[i]; const auto& binding = owner->binding().tensors[i];
            const auto selected = std::find(plan.tensors.begin(), plan.tensors.end(), static_cast<std::uint32_t>(i)) != plan.tensors.end();
            expect(binding.tensor_index == i && binding.bytes == tensor.byte_size &&
                binding.owner_node == tensor.placement.node,
                "relocation preserves tensor index, byte extent and placement metadata");
            expect(binding.address % 4096U == tensor.absolute_offset % 4096U,
                "all local bindings preserve their source page-relative alignment");
            expect(binding.allocation_generation == (i == 1U ? mapped->allocation_generation() : owner->generation()),
                "binding retains the actual mapping or residency generation");
            if (selected) expect(binding.storage == strix::RuntimeTensorStorage::resident &&
                binding.address % 4096U == tensor.absolute_offset % 4096U && owner->covers_device_range(binding.address, binding.bytes),
                "selected tensors point into actual owned page-congruent arena");
            else if (i == 1U) expect(binding.storage == strix::RuntimeTensorStorage::file_mapped &&
                binding.address == mapped->base_address() + tensor.absolute_offset,
                "excluded PLE table retains original host mapping and storage");
            else {
                void* alias = nullptr;
                check(hipHostGetDevicePointer(&alias, reinterpret_cast<void*>(mapped->base_address() + tensor.absolute_offset), 0),
                    "query unchanged registered tensor alias");
                expect(binding.storage == strix::RuntimeTensorStorage::registered_file_mapped &&
                    binding.address == reinterpret_cast<std::uintptr_t>(alias), "nonselected tensor remains the actual registered alias");
            }
        }
        const auto& first = owner->binding().tensors[0];
        const auto first_offset = manifest.tensors[0].absolute_offset;
        const auto mutation = static_cast<std::uint8_t>(expected_byte(first_offset) ^ 0xffU);
        check(hipMemset(reinterpret_cast<void*>(first.address), mutation, 1U), "mutate synthetic device copy only");
        std::uint8_t observed = 0;
        check(hipMemcpy(&observed, reinterpret_cast<const void*>(owner->binding().tensors[0].address), 1U, hipMemcpyDeviceToHost),
            "consume actual synthetic selected binding");
        expect(observed == mutation && *reinterpret_cast<const std::uint8_t*>(mapped->base_address() + first_offset) == expected_byte(first_offset),
            "consumer reads actual copy while immutable source stays unchanged");
        inventory = {}; const auto old_generation = owner->generation();
        auto moved = std::make_shared<strix::HipModelResidency>(std::move(*owner));
        expect(owner->generation() == 0 && owner->copied_bytes() == 0 && moved->generation() == old_generation && moved->copy_tier() == tier,
            "selective move transfers its one arena and clears moved-from ownership");
        owner = std::move(moved);
        auto assigned = std::make_shared<strix::HipModelResidency>();
        *assigned = std::move(*owner);
        expect(owner->generation() == 0 && assigned->generation() == old_generation && assigned->copy_tier() == tier,
            "selective move assignment transfers its same arena without another allocation");
        owner = std::move(assigned); inventory = owner->owner_inventory(owner); owner.reset();
        expect(diag::stats().frees == 0, "inventory pins keep selective arena alive");
        inventory = {};
        const auto done = diag::stats();
        expect(done.acquired_allocations == 1U && done.released_allocations == 1U && done.frees == 1U &&
            done.acquired_registrations == done.released_registrations, "selective release frees arena once and retires every alias");
    }
    const std::array faults{
        diag::FaultPlan{diag::Fault::allocation, 1}, diag::FaultPlan{diag::Fault::device_alias, 1},
        diag::FaultPlan{diag::Fault::device_alias, 2}, diag::FaultPlan{diag::Fault::copy, 1},
        diag::FaultPlan{diag::Fault::copy, 2}, diag::FaultPlan{diag::Fault::arena_alignment, 1},
        diag::FaultPlan{diag::Fault::alias_alignment, 1}};
    const auto policy = selective_policy(strix::HipModelCopyTier::hot_nonexpert);
    for (const auto fault : faults) {
        diag::set_fault_plan(fault); bool rejected = false;
        try { static_cast<void>(strix::HipModelResidency::create(mapped, manifest, generations, policy)); }
        catch (const std::exception&) { rejected = true; }
        const auto calls = diag::stats();
        expect(rejected && calls.acquired_allocations == calls.released_allocations &&
            calls.acquired_registrations == calls.released_registrations,
            "partial selective setup failure retires all acquired aliases and its arena");
        diag::set_fault_plan();
        { auto retry = strix::HipModelResidency::create(mapped, manifest, generations, policy);
          expect(retry.copy_tier() == policy.copy_tier, "fresh selective owner succeeds after failed setup"); }
    }
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "selective capture fixture stream");
    check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "selective capture fixture begin");
    diag::set_fault_plan(); strix::AllocationGenerationSource before_capture(100000); bool rejected = false;
    {
        strix::HipLaunchContext context; context.stream = stream;
        const strix::HipLaunchScope scope(context);
        try { static_cast<void>(strix::HipModelResidency::create(mapped, manifest, before_capture, policy)); }
        catch (const std::invalid_argument&) { rejected = true; }
    }
    hipGraph_t graph = nullptr; check(hipStreamEndCapture(stream, &graph), "selective capture fixture end");
    std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "selective capture fixture nodes");
    expect(rejected && nodes == 0 && before_capture.next() == 100000 && diag::stats().registrations == 0 &&
        diag::stats().allocations == 0, "active owned capture rejects before registration/allocation with zero graph nodes");
    check(hipGraphDestroy(graph), "selective fixture graph destroy"); check(hipStreamDestroy(stream), "selective fixture stream destroy");
    diag::set_fault_plan({diag::Fault::unregister, 1});
    auto quarantine_owner = std::make_shared<strix::HipModelResidency>(
        strix::HipModelResidency::create(mapped, manifest, generations, policy));
    std::weak_ptr<const strix::MappedModelFile> weak = mapped;
    mapped.reset(); quarantine_owner.reset();
    expect(!weak.expired() && diag::stats().quarantined_mappings == 1 &&
        diag::stats().acquired_allocations == diag::stats().released_allocations,
        "failed unregister conservatively retains actual mapping but retires independent arena");
    diag::retire_quarantined_mappings();
    expect(weak.expired(), "test-only quarantine retirement releases mapping only after actual unregister");
    diag::set_fault_plan();
}

strix::ModelManifest experts_manifest() {
    strix::ModelManifest manifest;
    manifest.architecture = "experts-residency-fixture";
    const auto add = [&](const char* name, std::uint64_t offset, std::uint64_t bytes,
                         strix::TensorRole role, strix::ResidencyPolicy residency) {
        strix::ManifestTensor tensor;
        tensor.name = name; tensor.ggml_type = 0; tensor.dimensions = {bytes / 4U};
        tensor.byte_size = bytes; tensor.absolute_offset = offset;
        tensor.role = role; tensor.residency = residency;
        manifest.total_tensor_bytes += bytes;
        if (residency == strix::ResidencyPolicy::file_backed_candidate)
            manifest.file_backed_candidate_bytes += bytes;
        manifest.tensors.push_back(std::move(tensor));
    };
    using Role = strix::TensorRole;
    using Residency = strix::ResidencyPolicy;
    add("norm", 4108U, 1024U, Role::normalization, Residency::resident);
    // The four small dense tensors share precisely the first/last source page
    // of the two experts. Each expert still has several unregistered pages.
    add("before.a", 8192U, 64U, Role::dense_projection, Residency::resident);
    add("expert.a", 8256U, 16384U, Role::moe_expert, Residency::resident);
    add("after.a", 24640U, 1024U, Role::dense_projection, Residency::resident);
    add("before.b", 40960U, 192U, Role::dense_projection, Residency::resident);
    add("expert.b", 41152U, 12288U, Role::moe_expert, Residency::resident);
    add("after.b", 53440U, 1024U, Role::dense_projection, Residency::resident);
    add("table", 61440U, 1024U, Role::ngram_table, Residency::file_backed_candidate);
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
    return manifest;
}

strix::HipModelResidencyPolicy experts_policy() {
    strix::HipModelResidencyPolicy policy;
    policy.copy_tier = strix::HipModelCopyTier::experts;
    policy.copy_payload_bytes = 28672U;
    policy.coalesce_gap_bytes = 0;
    policy.reservation_limits = {102400U, 122880U};
    return policy;
}

void experts_owned_and_faults(const std::filesystem::path& original_path) {
    namespace diag = strix::model_residency_diagnostic;
    diag::set_fault_plan();
    auto path = original_path; path += ".experts";
    write_fixture(path, 65536U);
    try {
        const auto manifest = experts_manifest();
        const auto policy = experts_policy();
        strix::AllocationGenerationSource generations;
        std::shared_ptr<const strix::MappedModelFile> mapped = std::make_shared<strix::MappedModelFile>(
            strix::MappedModelFile::open_read_only(path, "experts-source", generations));
        const auto plan = strix::build_selective_copy_plan(manifest, policy, mapped->bytes(), 4096U);
        const auto reservation = strix::plan_hip_model_residency_reservation(*mapped, manifest, policy);
        expect(plan.tensors == std::vector<std::uint32_t>{2U, 5U} && plan.ranges.size() == 2U &&
            plan.payload_bytes == 28672U && plan.arena_bytes == 36864U &&
            reservation.registered_alias_bytes == 20480U && reservation.retained_reserved_bytes == 102400U &&
            reservation.setup_staging_reserved_bytes == 20480U && reservation.setup_peak_reserved_bytes == 122880U,
            "experts fixture has two padded copies and only four nonexpert registration ranges");

        std::vector<strix::HipModelResidencyPolicy> invalid;
        auto bad = policy; bad.copy_payload_bytes = 28673U; invalid.push_back(bad);
        bad = policy; bad.copy_payload_bytes.reset(); invalid.push_back(bad);
        bad = policy; bad.reservation_limits.retained_bytes.reset(); invalid.push_back(bad);
        bad = policy; bad.reservation_limits.setup_peak_bytes.reset(); invalid.push_back(bad);
        bad = policy; bad.reservation_limits.retained_bytes = 102399U; invalid.push_back(bad);
        bad = policy; bad.reservation_limits.setup_peak_bytes = 122879U; invalid.push_back(bad);
        bad = policy; bad.reservation_limits.setup_peak_bytes = 102400U; invalid.push_back(bad);
        bad = policy; bad.coalesce_gap_bytes = 4096U; invalid.push_back(bad);
        bad = policy; bad.mode = strix::HipModelResidencyMode::resident_copy; invalid.push_back(bad);
        bad = policy; bad.promote_file_backed_candidates = true; invalid.push_back(bad);
        const auto reject_before_hip = [&](const strix::ModelManifest& input,
                                            const strix::HipModelResidencyPolicy& input_policy,
                                            bool borrowed = false, std::string_view expected_error = {}) {
            diag::set_fault_plan(); strix::AllocationGenerationSource before(100000);
            bool rejected = false;
            try {
                if (borrowed) static_cast<void>(strix::HipModelResidency::create(*mapped, input, before, input_policy));
                else static_cast<void>(strix::HipModelResidency::create(mapped, input, before, input_policy));
            } catch (const std::invalid_argument& error) {
                rejected = expected_error.empty() || std::string_view(error.what()) == expected_error;
            }
            const auto calls = diag::stats();
            expect(rejected && before.next() == 100000 && calls.registrations == 0 && calls.alias_queries == 0 &&
                calls.allocations == 0 && calls.copies == 0 && calls.frees == 0 && calls.staging_allocations == 0,
                "experts invalid geometry/ownership/budgets reject before generation or HIP mutation");
        };
        for (const auto& input_policy : invalid) reject_before_hip(manifest, input_policy);
        reject_before_hip(manifest, policy, true);
        auto short_expert = manifest;
        short_expert.tensors[2].byte_size = 1024U; short_expert.tensors[2].dimensions = {256U};
        short_expert.total_tensor_bytes -= 15360U;
        short_expert.layout_identity = strix::model_manifest_layout_identity(short_expert);
        bad = policy; bad.copy_payload_bytes = 13312U;
        reject_before_hip(short_expert, bad, false,
            "expert source is fully covered by registered boundary pages");
        auto interior_alias = manifest;
        // Remove both expert.a boundary aliases. Its sole overlap below is an
        // interior page, so a weak "at most two overlap pages" check would pass.
        interior_alias.tensors[1].absolute_offset = 0U;
        interior_alias.tensors[3].absolute_offset = 32768U;
        auto intruder = manifest.tensors[1]; intruder.name = "interior-alias";
        intruder.absolute_offset = 16384U; intruder.byte_size = 4U; intruder.dimensions = {1U};
        interior_alias.tensors.push_back(std::move(intruder)); interior_alias.total_tensor_bytes += 4U;
        interior_alias.layout_identity = strix::model_manifest_layout_identity(interior_alias);
        bad = policy; bad.reservation_limits = {131072U, 131072U};
        reject_before_hip(interior_alias, bad, false,
            "expert source interior overlaps registered model storage");

        // Tier none still registers the experts and owns no copied backing.
        diag::set_fault_plan();
        {
            strix::HipModelResidencyPolicy none;
            const auto old_plan = strix::plan_hip_model_residency_reservation(*mapped, manifest, none);
            auto baseline = strix::HipModelResidency::create(mapped, manifest, generations, none);
            expect(old_plan.registered_alias_bytes == 40960U && baseline.registered_bytes() == 40960U &&
                old_plan.setup_staging_reserved_bytes == 0 && diag::stats().staging_allocations == 0 &&
                baseline.copied_bytes() == 0 && baseline.binding().tensors[2].storage ==
                    strix::RuntimeTensorStorage::registered_file_mapped,
                "tier none preserves expert source registration without copied backing");
        }
        diag::set_fault_plan();
        auto owner = std::make_shared<strix::HipModelResidency>(
            strix::HipModelResidency::create(mapped, manifest, generations, policy));
        auto inventory = owner->owner_inventory(owner);
        expect(diag::stats().registrations == 4U && diag::stats().allocations == 1U && diag::stats().copies == 6U &&
            diag::stats().staging_allocations == 1U && diag::stats().acquired_staging_allocations == 1U &&
            diag::stats().staging_frees == 1U && diag::stats().released_staging_allocations == 1U &&
            diag::stats().staging_copy_chunks == 6U && diag::stats().maximum_staging_copy_bytes == 12288U &&
            owner->registered_bytes() == 20480U && owner->copied_bytes() == 36864U &&
            inventory.known_backings.size() == 2U && inventory.known_allocated_storage_bytes == 36864U &&
            inventory.unique_backing_reserved_bytes == 102400U,
            "experts source interiors are excluded from registration and all copies share one arena lease");
        std::uintptr_t arena = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> registered;
        std::size_t copy_views = 0;
        for (const auto& view : inventory.pinned_views) {
            if (view.category == "weights.copy") {
                ++copy_views;
                const auto base = view.address - view.view_offset;
                if (!arena) arena = base;
                expect(base == arena && view.allocation_bytes == 36864U,
                    "experts inventory exports two range views on exactly one arena backing");
            } else if (view.category == "weights.registered-host")
                registered.emplace_back(view.view_offset, view.view_offset + view.view_bytes);
        }
        expect(copy_views == 2U && registered.size() == 4U && arena % 4096U == 0,
            "expert arena and nonexpert aliases have the expected page layout");
        for (const auto& range : plan.ranges) {
            std::uint64_t overlap = 0;
            for (const auto& alias : registered) {
                const auto begin = std::max(range.file_begin, alias.first);
                const auto end = std::min(range.file_end, alias.second);
                if (end > begin) overlap += end - begin;
            }
            expect(overlap == 8192U && overlap < range.file_end - range.file_begin,
                "only two boundary pages of each expert remain registered");
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(range.file_end - range.file_begin));
            check(hipMemcpy(bytes.data(), reinterpret_cast<const void*>(arena + range.arena_offset), bytes.size(),
                hipMemcpyDeviceToHost), "read synthetic experts and padding");
            bool equal = true;
            for (std::size_t i = 0; i < bytes.size(); ++i) equal = equal && bytes[i] == expected_byte(range.file_begin + i);
            expect(equal, "every expert stored byte and boundary padding byte matches the mapped source");
        }
        for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
            const auto& tensor = manifest.tensors[i]; const auto& binding = owner->binding().tensors[i];
            expect(binding.tensor_index == i && binding.bytes == tensor.byte_size && binding.owner_node == tensor.placement.node &&
                binding.address % 4096U == tensor.absolute_offset % 4096U,
                "all expert experiment bindings preserve metadata and page-relative address congruence");
            if (tensor.role == strix::TensorRole::moe_expert) {
                expect(binding.storage == strix::RuntimeTensorStorage::resident &&
                    binding.allocation_generation == owner->generation() && binding.address >= arena &&
                    binding.address - arena < plan.arena_bytes && owner->covers_device_range(binding.address, binding.bytes),
                    "expert binding gets the arena and explicit residency generation without a registered source alias");
            } else if (tensor.residency == strix::ResidencyPolicy::file_backed_candidate) {
                expect(binding.storage == strix::RuntimeTensorStorage::file_mapped &&
                    binding.address == mapped->base_address() + tensor.absolute_offset &&
                    binding.allocation_generation == mapped->allocation_generation(),
                    "excluded table keeps its actual host mapping lease");
            } else {
                void* alias = nullptr;
                check(hipHostGetDevicePointer(&alias, reinterpret_cast<void*>(mapped->base_address() + tensor.absolute_offset), 0),
                    "query synthetic nonexpert alias");
                expect(binding.storage == strix::RuntimeTensorStorage::registered_file_mapped &&
                    binding.address == reinterpret_cast<std::uintptr_t>(alias) && binding.allocation_generation == owner->generation(),
                    "nonexpert binding remains the actual registered alias");
            }
        }
        {
            auto fake_mapping = std::shared_ptr<const strix::MappedModelFile>(mapped.get(), [](const auto*) {});
            auto fake_owner = std::shared_ptr<const strix::HipModelResidency>(owner.get(), [](const auto*) {});
            bool rejected_mapping = false, rejected_owner = false;
            try { static_cast<void>(owner->owner_inventory(owner, fake_mapping)); }
            catch (const std::invalid_argument&) { rejected_mapping = true; }
            try { static_cast<void>(owner->owner_inventory(fake_owner)); }
            catch (const std::invalid_argument&) { rejected_owner = true; }
            expect(rejected_mapping && rejected_owner, "experts inventory rejects fake control-block pins");
        }
        const auto mutation = static_cast<std::uint8_t>(expected_byte(8256U) ^ 0xffU);
        check(hipMemset(reinterpret_cast<void*>(owner->binding().tensors[2].address), mutation, 1U), "mutate synthetic expert arena");
        std::uint8_t observed = 0;
        check(hipMemcpy(&observed, reinterpret_cast<const void*>(owner->binding().tensors[2].address), 1U,
            hipMemcpyDeviceToHost), "consume synthetic expert binding");
        expect(observed == mutation && *reinterpret_cast<const std::uint8_t*>(mapped->base_address() + 8256U) == expected_byte(8256U),
            "expert consumer reads the separate arena while source bytes remain unchanged");
        inventory = {}; const auto generation = owner->generation();
        auto moved = std::make_shared<strix::HipModelResidency>(std::move(*owner));
        expect(owner->generation() == 0 && owner->copied_bytes() == 0 && moved->generation() == generation,
            "experts move construction transfers the arena and clears its source");
        owner = std::move(moved);
        auto assigned = std::make_shared<strix::HipModelResidency>(); *assigned = std::move(*owner);
        expect(owner->generation() == 0 && assigned->generation() == generation && assigned->copy_tier() == policy.copy_tier,
            "experts move assignment preserves its allocation generation and one arena");
        owner = std::move(assigned); inventory = owner->owner_inventory(owner);
        std::weak_ptr<const strix::MappedModelFile> mapping_weak = mapped;
        std::weak_ptr<const strix::HipModelResidency> owner_weak = owner;
        mapped.reset(); owner.reset();
        expect(!mapping_weak.expired() && !owner_weak.expired() && diag::stats().frees == 0,
            "experts inventory retains both actual owners through consumer lifetime");
        inventory = {};
        expect(mapping_weak.expired() && owner_weak.expired() && diag::stats().frees == 1U &&
            diag::stats().acquired_allocations == diag::stats().released_allocations &&
            diag::stats().acquired_registrations == diag::stats().released_registrations,
            "experts pins retire aliases and one arena before releasing the source mapping");

        mapped = std::make_shared<strix::MappedModelFile>(
            strix::MappedModelFile::open_read_only(path, "experts-faults", generations));
        const std::array faults{
            diag::FaultPlan{diag::Fault::device_alias, 1U}, diag::FaultPlan{diag::Fault::device_alias, 4U},
            diag::FaultPlan{diag::Fault::alias_alignment, 1U}, diag::FaultPlan{diag::Fault::allocation, 1U},
            diag::FaultPlan{diag::Fault::arena_alignment, 1U}, diag::FaultPlan{diag::Fault::copy, 1U},
            diag::FaultPlan{diag::Fault::copy, 2U}, diag::FaultPlan{diag::Fault::copy, 6U},
            diag::FaultPlan{diag::Fault::staging_allocation, 1U}, diag::FaultPlan{diag::Fault::staging_post_allocate, 1U}};
        for (const auto fault : faults) {
            diag::set_fault_plan(fault); bool rejected = false;
            try { static_cast<void>(strix::HipModelResidency::create(mapped, manifest, generations, policy)); }
            catch (const std::exception&) { rejected = true; }
            const auto calls = diag::stats();
            expect(rejected && calls.acquired_allocations <= 1U && calls.frees == calls.acquired_allocations &&
                calls.acquired_allocations == calls.released_allocations &&
                calls.acquired_registrations == calls.released_registrations &&
                calls.acquired_staging_allocations == calls.released_staging_allocations &&
                calls.staging_frees == calls.acquired_staging_allocations && calls.staging_frees <= 1U,
                "experts partial setup retires aliases, one arena and only an actually acquired pinned source");
            if (fault.fault == diag::Fault::staging_allocation)
                expect(calls.staging_allocations == 1U && calls.acquired_staging_allocations == 0 && calls.staging_frees == 0,
                    "failed pinned allocation does not manufacture a lease or free");
            if (fault.fault == diag::Fault::staging_post_allocate)
                expect(calls.staging_allocations == 1U && calls.acquired_staging_allocations == 1U && calls.staging_frees == 1U,
                    "an acquired pinned source survives guard failure and is retired once");
            diag::set_fault_plan();
            { auto retry = strix::HipModelResidency::create(mapped, manifest, generations, policy);
              expect(retry.copy_tier() == policy.copy_tier, "experts fresh creation succeeds after each injected failure"); }
        }
        // This cap is compiled only into the diagnostic executable; production
        // still uses min(32 MiB, largest logical range). It forces a real tail.
        diag::set_fault_plan({diag::Fault::none, 1U, 8192U});
        const auto chunk_plan = strix::plan_hip_model_residency_reservation(*mapped,manifest,policy);
        expect(chunk_plan.setup_staging_reserved_bytes == 8192U &&
            chunk_plan.setup_peak_reserved_bytes == 110592U && chunk_plan.retained_reserved_bytes == 102400U,
            "diagnostic chunk capacity is included in pre-HIP temporary budgeting");
        {
            auto chunked = strix::HipModelResidency::create(mapped,manifest,generations,policy);
            const auto calls = diag::stats();
            expect(calls.copies == 7U && calls.staging_copy_chunks == 7U &&
                calls.maximum_staging_copy_bytes == 8192U && calls.staging_allocations == 1U &&
                calls.staging_frees == 1U && calls.acquired_staging_allocations == calls.released_staging_allocations,
                "one retired staging lease serves every real bounded chunk and short tail");
            const auto expert_address = chunked.binding().tensors[2].address;
            const auto base = expert_address - (manifest.tensors[2].absolute_offset - plan.ranges[0].file_begin);
            for (const auto& range : plan.ranges) {
                std::vector<std::byte> bytes(static_cast<std::size_t>(range.file_end-range.file_begin));
                check(hipMemcpy(bytes.data(),reinterpret_cast<void*>(base+range.arena_offset),bytes.size(),hipMemcpyDeviceToHost),
                    "read every chunked expert arena byte");
                const auto* expected = reinterpret_cast<const std::byte*>(mapped->base_address()+range.file_begin);
                expect(std::equal(bytes.begin(),bytes.end(),expected),"chunk boundaries preserve complete source payload and padding");
            }
        }
        for (const auto occurrence : {1U,4U,7U}) {
            diag::set_fault_plan({diag::Fault::copy,occurrence,8192U});
            bool rejected = false;
            try { (void)strix::HipModelResidency::create(mapped,manifest,generations,policy); }
            catch (const std::exception&) { rejected = true; }
            const auto calls = diag::stats();
            expect(rejected && calls.copies == occurrence && calls.frees == 1U &&
                calls.acquired_allocations == calls.released_allocations &&
                calls.acquired_registrations == calls.released_registrations &&
                calls.staging_frees == 1U && calls.acquired_staging_allocations == calls.released_staging_allocations,
                "first, middle and last chunk failures drain and retire both actual allocations");
            // Retry with the same generation source object; never rewind or
            // substitute a fake source to escape partial-setup accounting.
            diag::set_fault_plan({diag::Fault::none,1U,8192U});
            { auto retry = strix::HipModelResidency::create(mapped,manifest,generations,policy);
              expect(retry.copy_tier()==policy.copy_tier && diag::stats().copies==7U,
                  "same generation source supports fresh chunked creation after each failure"); }
        }
        diag::set_fault_plan();
        hipStream_t stream = nullptr;
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "experts capture stream");
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "experts capture begin");
        diag::set_fault_plan(); strix::AllocationGenerationSource before_capture(100000); bool rejected = false;
        {
            strix::HipLaunchContext context; context.stream = stream;
            const strix::HipLaunchScope scope(context);
            try { static_cast<void>(strix::HipModelResidency::create(mapped, manifest, before_capture, policy)); }
            catch (const std::invalid_argument&) { rejected = true; }
        }
        hipGraph_t graph = nullptr; check(hipStreamEndCapture(stream, &graph), "experts capture end");
        std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "experts capture nodes");
        expect(rejected && nodes == 0 && before_capture.next() == 100000 && diag::stats().registrations == 0 &&
            diag::stats().allocations == 0 && diag::stats().copies == 0 && diag::stats().staging_allocations == 0,
            "experts owned capture rejects before generation, registration, allocation or graph nodes");
        check(hipGraphDestroy(graph), "experts capture graph destroy");
        check(hipStreamDestroy(stream), "experts capture stream destroy");
        diag::set_fault_plan({diag::Fault::unregister, 1U});
        auto quarantined = std::make_shared<strix::HipModelResidency>(
            strix::HipModelResidency::create(mapped, manifest, generations, policy));
        mapping_weak = mapped; mapped.reset(); quarantined.reset();
        const auto calls = diag::stats();
        expect(!mapping_weak.expired() && calls.quarantined_mappings == 1U && calls.frees == 1U &&
            calls.acquired_allocations == calls.released_allocations &&
            calls.acquired_registrations == calls.released_registrations + 1U,
            "experts failed unregister keeps the source lease while freeing its independent arena once");
        diag::retire_quarantined_mappings();
        expect(mapping_weak.expired(), "experts quarantine releases mapping only after actual failed registration retirement");
        diag::set_fault_plan();
    } catch (...) { std::filesystem::remove(path); throw; }
    std::filesystem::remove(path);
}
#endif

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
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
        selective_owned_and_faults(path);
        experts_owned_and_faults(path);
#endif

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
