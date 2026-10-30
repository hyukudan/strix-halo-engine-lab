#pragma once
#include "hip_decode_options.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/hip_mtp_asset_loader.hpp"
namespace strix::decode_cli {
inline ModelContentIdentity mtp_cli_digest(std::string_view text) {
    if(!mtp_sha256_text(text))throw std::invalid_argument("MTP SHA256 pin is invalid");
    const auto digit=[](char ch)->unsigned {if(ch>='0'&&ch<='9')return static_cast<unsigned>(ch-'0');
        if(ch>='a'&&ch<='f')return static_cast<unsigned>(ch-'a')+10U;
        return static_cast<unsigned>(ch-'A')+10U;};
    ModelContentIdentity out;
    for(std::size_t i=0;i<out.bytes.size();++i)out.bytes[i]=static_cast<std::uint8_t>((digit(text[i*2U])<<4U)|digit(text[i*2U+1U]));
    return out;
}
// Accepted hexadecimal pins denote binary SHA-256 digests in either case.
inline void verify_mtp_serving_assets(const MtpCliOptions& m,
    const ModelContentIdentity& sidecar, std::uint64_t sidecar_bytes,
    const ModelContentIdentity& golden, std::uint64_t golden_bytes) {
    if(sidecar_bytes!=m.sidecar_bytes || sidecar!=mtp_cli_digest(m.sidecar_sha256) ||
        golden!=mtp_cli_digest(m.golden_sha256) || golden_bytes>m.golden_max_bytes)
        throw std::invalid_argument("one-slot:asset size or digest differs");
}
inline std::shared_ptr<const HipMtpAssets> load_mtp_cli_assets(const MtpCliOptions& m, const ModelContract& model,
    std::shared_ptr<const MappedModelFile> mapped, const ModelManifest& manifest,
    std::shared_ptr<const HipModelResidency> weights, AllocationGenerationSource& generations) {
    HipMtpAssetLoadRequest load;load.sidecar_path=m.sidecar;load.sidecar_logical_name=m.sidecar_logical_name;
    load.expected_target_file_sha256=mtp_cli_digest(m.target_sha256);load.expected_target_content=mtp_cli_digest(m.target_content);
    auto& evidence=load.expected_sidecar;evidence.repository=m.repository;evidence.revision=m.revision;evidence.relative_path=m.source_path;
    evidence.declared_file_sha256=mtp_cli_digest(m.sidecar_sha256);evidence.declared_file_bytes=m.sidecar_bytes;
    evidence.conversion=MtpSidecarConversionProfile::effective_norm_gain_embedding_first_packed_v1;
    evidence.converter_reference_sha256=mtp_cli_digest(m.converter_sha256);
    evidence.norm_converter_reference_sha256=mtp_cli_digest(m.norm_converter_sha256);evidence.converter_reference_revision=m.converter_revision;
    load.metadata_read_budget_bytes=m.header_bytes;
    load.residency.mode=m.sidecar_residency=="copy"?HipModelResidencyMode::resident_copy:HipModelResidencyMode::registered_file_mapping;
    load.residency.reservation_limits={m.sidecar_retained_bytes,m.sidecar_setup_bytes};
    return HipMtpAssets::load(model,load,std::move(mapped),manifest,std::move(weights),generations);
}
} // namespace strix::decode_cli
