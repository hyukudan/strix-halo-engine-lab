#include "strix/hip_model_residency.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>

namespace {
void expect(bool value, std::source_location at = std::source_location::current()) {
    if (!value) throw std::runtime_error("selective copy layout failed at line " + std::to_string(at.line()));
}
template<class Action> void rejects(Action action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    expect(failed);
}
strix::ManifestTensor tensor(strix::TensorRole role, std::uint64_t offset, std::uint64_t bytes) {
    strix::ManifestTensor result;
    result.name = "fixture." + std::to_string(offset);
    result.role = role;
    result.ggml_type = 24; // One byte per element; synthetic byte-layout fixture.
    result.absolute_offset = offset;
    result.byte_size = bytes;
    result.dimensions = {bytes};
    return result;
}
void seal(strix::ModelManifest& manifest) {
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
}
strix::HipModelResidencyPolicy policy(strix::HipModelCopyTier tier, std::uint64_t payload) {
    strix::HipModelResidencyPolicy result;
    result.copy_tier = tier;
    result.copy_payload_bytes = payload;
    result.reservation_limits.retained_bytes = std::numeric_limits<std::uint64_t>::max();
    result.reservation_limits.setup_peak_bytes = std::numeric_limits<std::uint64_t>::max();
    return result;
}
}

int main() try {
    using namespace strix;
    ModelManifest manifest;
    manifest.tensors = {tensor(TensorRole::normalization, 4103, 33),
        tensor(TensorRole::output_head, 8190, 10), tensor(TensorRole::dense_projection, 20003, 7),
        tensor(TensorRole::moe_expert, 32768, 4096), tensor(TensorRole::embedding, 49152, 1024)};
    seal(manifest);
    const auto identity = manifest.layout_identity;
    for (const auto mode : {HipModelResidencyMode::registered_file_mapping, HipModelResidencyMode::resident_copy}) {
        HipModelResidencyPolicy defaults; defaults.mode = mode;
        const auto empty = build_selective_copy_plan(manifest, defaults, 65536, 4096);
        expect(empty.tensors.empty() && empty.ranges.empty() && empty.payload_bytes == 0 && empty.arena_bytes == 0);
    }
    auto norms = policy(HipModelCopyTier::normalization, 33);
    const auto a = build_selective_copy_plan(manifest, norms, 65536, 4096);
    expect(a.payload_bytes == 33 && a.arena_bytes == 4096);
    expect(a.tensors.size() == 1 && a.tensors.front() == 0 && a.ranges.size() == 1);
    expect(a.ranges[0].file_begin == 4096 && a.ranges[0].file_end == 8192 && a.ranges[0].arena_offset == 0);
    auto hot = policy(HipModelCopyTier::hot_nonexpert, 50);
    const auto b = build_selective_copy_plan(manifest, hot, 65536, 4096);
    expect(b.payload_bytes == 50 && b.arena_bytes == 12288 && b.tensors.size() == 3 && b.ranges.size() == 2);
    expect(b.ranges[0].file_begin == 4096 && b.ranges[0].file_end == 12288 && b.ranges[0].arena_offset == 0);
    expect(b.ranges[1].file_begin == 16384 && b.ranges[1].file_end == 20480 && b.ranges[1].arena_offset == 8192);
    for (const auto index : b.tensors) {
        const auto& t = manifest.tensors.at(index);
        const auto range = std::find_if(b.ranges.begin(), b.ranges.end(), [&](const auto& r) {
            return t.absolute_offset >= r.file_begin && t.absolute_offset < r.file_end &&
                t.byte_size <= r.file_end - t.absolute_offset;
        });
        expect(range != b.ranges.end());
        const auto relative = range->arena_offset + t.absolute_offset - range->file_begin;
        expect(relative % 4096 == t.absolute_offset % 4096);
        expect(relative <= b.arena_bytes && t.byte_size <= b.arena_bytes - relative);
    }
    expect(manifest.layout_identity == identity && model_manifest_layout_identity(manifest) == identity);
    // Registration coalescing must not pull the unselected gap into the arena.
    hot.coalesce_gap_bytes = 65536;
    const auto coalesced = build_selective_copy_plan(manifest, hot, 65536, 4096);
    expect(coalesced.ranges.size() == b.ranges.size() && coalesced.arena_bytes == b.arena_bytes);
    for (const auto role : {TensorRole::embedding, TensorRole::moe_expert, TensorRole::ngram_table,
                           TensorRole::unknown, TensorRole::mtp})
        expect(!selective_copy_eligible(tensor(role, 0, 4), HipModelCopyTier::hot_nonexpert));
    for (const auto role : {TensorRole::output_head, TensorRole::normalization, TensorRole::attention_projection,
                           TensorRole::dense_projection, TensorRole::moe_router, TensorRole::recurrent_state,
                           TensorRole::sparse_index, TensorRole::ple_parameter})
        expect(selective_copy_eligible(tensor(role, 0, 4), HipModelCopyTier::hot_nonexpert));
    auto excluded = tensor(TensorRole::normalization, 0, 4);
    excluded.placement.kind = PlacementKind::remote;
    expect(!selective_copy_eligible(excluded, HipModelCopyTier::normalization));
    excluded.placement.kind = PlacementKind::local;
    for (const auto residency : {ResidencyPolicy::file_backed_candidate, ResidencyPolicy::bounded_cache_candidate}) {
        excluded.residency = residency;
        expect(!selective_copy_eligible(excluded, HipModelCopyTier::hot_nonexpert));
    }
    auto bad = norms; bad.copy_tier = static_cast<HipModelCopyTier>(99);
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    bad = norms; bad.mode = HipModelResidencyMode::resident_copy;
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    bad = norms; bad.promote_file_backed_candidates = true;
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    auto invalid = manifest; invalid.unknown_tensor_count = 1; seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, norms, 65536, 4096); });
    invalid = manifest; invalid.tensors.clear(); seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, norms, 65536, 4096); });
    for (unsigned fault = 0; fault < 3; ++fault) {
        invalid = manifest;
        if (fault == 0) invalid.tensors[0].byte_size = 0;
        if (fault == 1) invalid.tensors[0].dimensions.clear();
        if (fault == 2) invalid.tensors[0].absolute_offset = 65530;
        seal(invalid);
        rejects([&] { (void)build_selective_copy_plan(invalid, norms, 65536, 4096); });
    }
    rejects([&] { (void)build_selective_copy_plan(manifest, norms, 65536, 0); });
    rejects([&] { (void)build_selective_copy_plan(manifest, norms, 65536, 3000); });
    bad = norms; bad.copy_payload_bytes = 32;
    rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    for (unsigned missing = 0; missing < 3; ++missing) {
        bad = norms;
        if (missing == 0) bad.copy_payload_bytes.reset();
        if (missing == 1) bad.reservation_limits.retained_bytes.reset();
        if (missing == 2) bad.reservation_limits.setup_peak_bytes.reset();
        rejects([&] { (void)build_selective_copy_plan(manifest, bad, 65536, 4096); });
    }
    invalid = manifest; invalid.tensors[0].dimensions = {32}; seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, norms, 65536, 4096); });
    // The final range is clipped to real file bytes: no read past mmap EOF.
    invalid = {}; invalid.tensors = {tensor(TensorRole::normalization, 65530, 5)}; seal(invalid);
    const auto tail = build_selective_copy_plan(invalid, policy(HipModelCopyTier::normalization, 5), 65535, 4096);
    expect(tail.ranges.size() == 1 && tail.ranges[0].file_end == 65535 && tail.arena_bytes == 4095);
    // Overflow in tensor extent and in page padding are distinct errors.
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    invalid = {}; invalid.tensors = {tensor(TensorRole::normalization, maximum - 2, 4)}; seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, policy(HipModelCopyTier::normalization, 4), maximum, 4096); });
    invalid.tensors[0].byte_size = 1; invalid.tensors[0].dimensions = {1}; seal(invalid);
    rejects([&] { (void)build_selective_copy_plan(invalid, policy(HipModelCopyTier::normalization, 1), maximum, 4096); });
    std::cout << "selective model copy selection, page congruence and bounded layout PASS\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
