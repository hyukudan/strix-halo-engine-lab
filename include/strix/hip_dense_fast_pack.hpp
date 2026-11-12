#pragma once

// Dense FAST pack (--dense-fast-pack): a GGUF of Q4_K / Q5_K / Q6_K copies, under their own
// names, of the target's dense Q8_0 matrices (tools/dense_fast_quantize.py). The pack pins
// the exact target file it was built from. Its copies hold re-quantised values: a run that
// reads them has a FAST numerical identity of its own (docs/engine/kernels/dense-fast-pack.md).

#include "strix/hip_model_residency.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace strix {

struct HipDenseFastPack {
    std::shared_ptr<const MappedModelFile> mapped;
    std::shared_ptr<const HipModelResidency> residency;
    std::string spec, sha256;
    std::uint64_t tensors = 0, packed_bytes = 0, replaced_bytes = 0;
    ~HipDenseFastPack();
};

// Maps the pack, checks every copy's type and shape against its target tensor and
// registers it in strix::dense_fast (cleared when the returned owner is destroyed).
[[nodiscard]] std::shared_ptr<const HipDenseFastPack> load_hip_dense_fast_pack(const std::filesystem::path& path,
    const MappedModelFile& target, const ModelManifest& target_manifest, const ModelRuntimeBinding& target_binding,
    AllocationGenerationSource& generations);

}  // namespace strix
