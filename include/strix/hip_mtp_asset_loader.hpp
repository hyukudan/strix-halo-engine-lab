#pragma once
#include "strix/compiled_mtp_draft_plan.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include <filesystem>
#include <memory>

namespace strix {
struct HipMtpAssetLoadRequest {
    std::filesystem::path sidecar_path;
    std::string sidecar_logical_name;
    MtpSidecarAssetEvidence expected_sidecar;
    ModelContentIdentity expected_target_file_sha256,expected_target_content;
    // Explicit bounded header read; does not hold the full weight payload.
    std::uint64_t metadata_read_budget_bytes=0;
    // Both retained/setup limits are explicit and checked before HIP creation.
    HipModelResidencyPolicy residency;
};
// Local assets only; no downloads, environment changes or MTP execution. The
// actual target is reused without rehashing, registering or copying it again.
// This owner is nonmovable: shared binding descriptor addresses remain stable.
class HipMtpAssets final {
public:
    [[nodiscard]] static std::shared_ptr<const HipMtpAssets> load(const ModelContract&,
        const HipMtpAssetLoadRequest&,std::shared_ptr<const MappedModelFile> actual_target_mapping,
        const ModelManifest& actual_target_manifest,std::shared_ptr<const HipModelResidency> actual_target_residency,
        AllocationGenerationSource&);
    ~HipMtpAssets();
    HipMtpAssets(const HipMtpAssets&)=delete;
    HipMtpAssets& operator=(const HipMtpAssets&)=delete;
    [[nodiscard]] const ModelManifest& sidecar_manifest() const noexcept;
    [[nodiscard]] const ModelManifest& target_manifest() const noexcept;
    [[nodiscard]] std::shared_ptr<const MappedModelFile> sidecar_mapping() const noexcept;
    [[nodiscard]] std::shared_ptr<const HipModelResidency> sidecar_residency() const noexcept;
    [[nodiscard]] std::shared_ptr<const HipModelResidency> target_residency() const noexcept;
    [[nodiscard]] const MtpSidecarSourceContract& source() const noexcept;
    [[nodiscard]] const MtpSharedTargetBinding& shared_target() const noexcept;
    [[nodiscard]] const MtpDraftRuntimeBinding& draft_binding() const noexcept;
    [[nodiscard]] const CompiledMtpDraftPlan& compiled() const noexcept;
    // Setup-only FD witnesses, not an extra full-file hash and not hot-path I/O.
    void validate_assets_unchanged() const;
private:
    HipMtpAssets();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix