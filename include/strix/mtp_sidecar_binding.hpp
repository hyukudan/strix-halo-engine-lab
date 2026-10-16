#pragma once
#include "strix/model_contract.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/mtp_forward_semantics.hpp"
#include <memory>
#include <optional>
#include <string>

namespace strix {
// The only source-conversion profile currently admitted by this loader. It
// describes interpretation, not an attestation of how a published file was made.
enum class MtpSidecarConversionProfile : std::uint32_t {
    unresolved=0, effective_norm_gain_embedding_first_packed_v1=1
};
struct MtpSidecarAssetEvidence {
    std::string repository,revision,relative_path;
    ModelContentIdentity declared_file_sha256;
    std::optional<ModelContentIdentity> verified_file_sha256;
    // Aggregate identity from the same fully hashed loader/mapping; the file
    // digest and framed runtime content identity are deliberately not aliases.
    ModelContentIdentity verified_model_content_identity;
    std::uint64_t declared_file_bytes=0;
    MtpSidecarConversionProfile conversion=MtpSidecarConversionProfile::unresolved;
    ModelContentIdentity converter_reference_sha256;
    ModelContentIdentity norm_converter_reference_sha256;
    std::string converter_reference_revision;
};
// Metadata inspection can describe a declared remote object from bounded
// headers. It cannot acquire live binding authority without a verified FULL
// file hash. No boolean can substitute for matching the two digest values.
struct MtpSidecarSourceContract {
    MtpSidecarAssetEvidence asset;
    std::uint64_t manifest_layout_identity=0,source_identity=0;
    bool shared_target_tensors=false;
    MtpNormWeightEncoding norm_weight_encoding=MtpNormWeightEncoding::effective_scale;
};
[[nodiscard]] ModelManifest build_mtp_sidecar_manifest(const ModelContract&,
    const gguf::FileInfo&,const MtpSidecarAssetEvidence&);
[[nodiscard]] MtpSidecarSourceContract inspect_mtp_sidecar_source(const ModelContract&,
    const gguf::FileInfo&,const MtpSidecarAssetEvidence&);
void validate_mtp_sidecar_source(const ModelContract&,const ModelManifest&,
    const MtpSidecarSourceContract&,bool require_verified_full_file);

struct MtpSharedTargetBinding {
    MtpSidecarSourceContract source;
    std::uint32_t target_embedding_tensor_index=0,target_output_tensor_index=0;
    std::uint64_t sidecar_manifest_layout_identity=0,target_manifest_layout_identity=0;
    ModelContentIdentity sidecar_model_content_identity,target_model_content_identity;
    std::uint64_t compatibility_identity=0,physical_binding_identity=0;
    // These must be the real model/residency owner control blocks retaining
    // the passed immutable descriptors and allocations, never alias-free tags.
    std::shared_ptr<const void> actual_sidecar_owner,actual_target_owner;
    const ModelManifest* sidecar_manifest=nullptr;
    const ModelRuntimeBinding* sidecar_runtime=nullptr;
    const ModelManifest* target_manifest=nullptr;
    const ModelRuntimeBinding* target_runtime=nullptr;
};
[[nodiscard]] MtpSharedTargetBinding bind_mtp_shared_target_tensors(const ModelContract&,
    const MtpSidecarSourceContract&,const ModelManifest& sidecar,const ModelRuntimeBinding& sidecar_runtime,
    const ModelManifest& target,const ModelRuntimeBinding& target_runtime,
    const ModelContentIdentity& expected_target_content,std::shared_ptr<const void> actual_sidecar_owner,
    std::shared_ptr<const void> actual_target_owner);
void validate_mtp_shared_target_binding(const ModelContract&,const ModelManifest& sidecar,
    const ModelRuntimeBinding& sidecar_runtime,const ModelManifest& target,const ModelRuntimeBinding& target_runtime,
    const ModelContentIdentity& expected_target_content,const MtpSharedTargetBinding&);
} // namespace strix