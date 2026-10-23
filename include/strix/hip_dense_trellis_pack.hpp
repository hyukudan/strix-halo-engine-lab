#pragma once

#include "strix/hip_dense_trellis.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include <filesystem>
#include <memory>
#include <string>

namespace strix {
inline constexpr const char* dense_trellis_g1_pack_sha256 = "040e72b2317b9f6b3985d21e8e738d4f98b094bc9b9a87e41418b8855b51b4dc";
struct HipDenseTrellisPack {
    struct Impl;
    std::shared_ptr<Impl> impl;
    std::string mix, sha256;
    std::uint64_t tensors = 0, packed_bytes = 0, replaced_bytes = 0, workspace_bytes = 0;
    ~HipDenseTrellisPack();
};

// Experimental G1 only. Pins both the admitted source checkpoint and the exact
// selected pack. Reserves all workspace before publishing an immutable registry.
// The owner must outlive submitted work; destruction requires quiescent callers.
std::shared_ptr<const HipDenseTrellisPack> load_hip_dense_trellis_pack(const std::filesystem::path& path,
    const MappedModelFile& target, const ModelManifest& manifest, const ModelRuntimeBinding& binding,
    AllocationGenerationSource& generations);

bool hip_dense_trellis_active() noexcept;
// Returns false without enqueueing anything for an unregistered source tensor.
bool project_hip_dense_trellis_registered(const void* original, const HipDenseTrellisRows& views,
    unsigned rows, unsigned inputs, unsigned outputs, void* stream);
} // namespace strix
