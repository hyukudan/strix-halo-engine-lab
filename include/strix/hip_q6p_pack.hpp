#pragma once

// Dense Q6P32 pack (--dense-q6p-pack): a GGUF of "<name>.q6p" (I8 bytes, Q6P32 rows)
// and "<name>.bf16" (BF16) copies of the target's dense Q8_0 and F32 router tensors.
// The pack pins the exact target file it was built from; every copy is checked to
// decode to the target tensor's bytes before it is registered with strix::q6p.

#include "strix/hip_model_residency.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace strix {

struct Q6pPackHeader {
    std::string format, source_content, target_content, target_file_sha256;
};
// Header metadata only (no mapping, no hashing).
[[nodiscard]] Q6pPackHeader read_q6p_pack_header(const std::filesystem::path& path);

struct HipQ6pPack {
    std::shared_ptr<const MappedModelFile> mapped;
    std::shared_ptr<const HipModelResidency> residency;
    std::uint64_t q6p_tensors = 0, bf16_tensors = 0, q6p_bytes = 0, replaced_bytes = 0;
    ~HipQ6pPack();
};

// Maps and registers the pack, verifies every copy against the target mapping and
// registers it in strix::q6p (cleared when the returned owner is destroyed).
[[nodiscard]] std::shared_ptr<const HipQ6pPack> load_hip_q6p_pack(const std::filesystem::path& path,
    const MappedModelFile& target, const ModelManifest& target_manifest, const ModelRuntimeBinding& target_binding,
    AllocationGenerationSource& generations);

}  // namespace strix
