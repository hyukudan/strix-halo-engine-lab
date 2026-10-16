#include "strix/hip_mtp_asset_loader.hpp"
#include <algorithm>
#include <istream>
#include <limits>
#include <stdexcept>
#include <streambuf>
#include <utility>

namespace strix {
namespace {
class HeaderBuffer final:public std::streambuf {
public:
    HeaderBuffer(const MappedModelFile& mapped,std::uint64_t budget) {
        const auto bytes=std::min(mapped.bytes(),budget);
        if(!bytes || bytes>static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
            throw std::invalid_argument("MTP bounded header view exceeds pointer domain");
        // streambuf never writes through its get area. The source mapping is
        // read-only, owned and fstat-checked before/after this metadata parse.
        auto* base=reinterpret_cast<char*>(mapped.base_address());
        setg(base,base,base+static_cast<std::ptrdiff_t>(bytes));
    }
};
void require(bool ok,const char* message) {if(!ok)throw std::invalid_argument(message);}
}
struct HipMtpAssets::Impl {
    ModelContract model;
    std::shared_ptr<const MappedModelFile> sidecar_map,target_map;
    ModelManifest sidecar_manifest,target_manifest;
    std::shared_ptr<const HipModelResidency> sidecar,target;
    MtpSidecarSourceContract source;
    MtpSharedTargetBinding shared;
    MtpDraftRuntimeBinding draft;
    CompiledMtpDraftPlan compiled;
};
HipMtpAssets::HipMtpAssets():impl_(std::make_unique<Impl>()) {}
HipMtpAssets::~HipMtpAssets()=default;
std::shared_ptr<const HipMtpAssets> HipMtpAssets::load(const ModelContract& model,const HipMtpAssetLoadRequest& request,
    std::shared_ptr<const MappedModelFile> target_map,const ModelManifest& target_manifest,
    std::shared_ptr<const HipModelResidency> target,AllocationGenerationSource& generations) {
    require(model.validate().empty() && !request.sidecar_path.empty() && !request.sidecar_logical_name.empty() &&
        request.expected_sidecar.declared_file_sha256.valid() && request.expected_sidecar.declared_file_bytes &&
        request.metadata_read_budget_bytes && request.metadata_read_budget_bytes<=32ULL*1024ULL*1024ULL &&
        request.residency.reservation_limits.retained_bytes && *request.residency.reservation_limits.retained_bytes &&
        request.residency.reservation_limits.setup_peak_bytes && *request.residency.reservation_limits.setup_peak_bytes,
        "MTP asset load requires explicit file/header/residency budgets and a valid model");
    require(target_map && target && request.expected_target_file_sha256.valid() && request.expected_target_content.valid(),
        "MTP asset load requires the two actual target owners and expected file/content pins");
    target_map->validate_unchanged();
    require(target_map->file_sha256()==request.expected_target_file_sha256 && target_map->content_identity()==request.expected_target_content &&
        target->binding().model_content_identity==request.expected_target_content,
        "MTP target mapping/residency differs from the expected verified base file");
    // Strong-create checks the exact actual mapping control block, not just
    // matching byte values. This inventory snapshot allocates no GPU pool.
    (void)target->owner_inventory(target,target_map);
    auto assets=std::shared_ptr<HipMtpAssets>(new HipMtpAssets);
    auto& s=*assets->impl_;s.model=model;s.target_map=std::move(target_map);s.target=std::move(target);
    s.target_manifest=target_manifest;
    s.sidecar_map=std::make_shared<const MappedModelFile>(MappedModelFile::open_read_only(
        request.sidecar_path,request.sidecar_logical_name,generations));
    s.sidecar_map->validate_unchanged();
    require(s.sidecar_map->file_sha256()==request.expected_sidecar.declared_file_sha256 &&
        s.sidecar_map->bytes()==request.expected_sidecar.declared_file_bytes,
        "MTP sidecar local full-file digest/size differs from declared asset");
    auto evidence=request.expected_sidecar;
    if(evidence.verified_file_sha256) require(*evidence.verified_file_sha256==s.sidecar_map->file_sha256(),
        "MTP supplied prior file verification differs from the current actual mapping");
    if(evidence.verified_model_content_identity.valid()) require(evidence.verified_model_content_identity==s.sidecar_map->content_identity(),
        "MTP supplied prior content identity differs from the current actual mapping");
    evidence.verified_file_sha256=s.sidecar_map->file_sha256();
    evidence.verified_model_content_identity=s.sidecar_map->content_identity();
    HeaderBuffer buffer(*s.sidecar_map,request.metadata_read_budget_bytes);std::istream input(&buffer);
    gguf::Limits limits;limits.max_metadata_entries=256;limits.max_tensors=64;
    limits.max_string_bytes=request.metadata_read_budget_bytes;limits.max_total_array_elements=1000000;limits.max_array_depth=2;
    const auto info=gguf::read(input,s.sidecar_map->bytes(),limits);
    s.sidecar_map->validate_unchanged();s.target_map->validate_unchanged();
    s.sidecar_manifest=build_mtp_sidecar_manifest(model,info,evidence);
    s.source=inspect_mtp_sidecar_source(model,info,evidence);
    // The existing planner applies explicit limits BEFORE actual HIP resources.
    (void)plan_hip_model_residency_reservation(*s.sidecar_map,s.sidecar_manifest,request.residency);
    s.sidecar=std::make_shared<const HipModelResidency>(HipModelResidency::create(
        s.sidecar_map,s.sidecar_manifest,generations,request.residency));
    s.sidecar_map->validate_unchanged();s.target_map->validate_unchanged();
    s.shared=bind_mtp_shared_target_tensors(model,s.source,s.sidecar_manifest,s.sidecar->binding(),
        s.target_manifest,s.target->binding(),request.expected_target_content,s.sidecar,s.target);
    s.draft=build_mtp_draft_runtime_binding(s.sidecar_manifest,s.sidecar->binding());
    s.compiled=compile_mtp_draft_plan(model,s.sidecar_manifest,s.draft,&s.shared);
    return assets;
}
const ModelManifest& HipMtpAssets::sidecar_manifest()const noexcept{return impl_->sidecar_manifest;}
const ModelManifest& HipMtpAssets::target_manifest()const noexcept{return impl_->target_manifest;}
std::shared_ptr<const MappedModelFile> HipMtpAssets::sidecar_mapping()const noexcept{return impl_->sidecar_map;}
std::shared_ptr<const HipModelResidency> HipMtpAssets::sidecar_residency()const noexcept{return impl_->sidecar;}
std::shared_ptr<const HipModelResidency> HipMtpAssets::target_residency()const noexcept{return impl_->target;}
const MtpSidecarSourceContract& HipMtpAssets::source()const noexcept{return impl_->source;}
const MtpSharedTargetBinding& HipMtpAssets::shared_target()const noexcept{return impl_->shared;}
const MtpDraftRuntimeBinding& HipMtpAssets::draft_binding()const noexcept{return impl_->draft;}
const CompiledMtpDraftPlan& HipMtpAssets::compiled()const noexcept{return impl_->compiled;}
void HipMtpAssets::validate_assets_unchanged()const {
    impl_->sidecar_map->validate_unchanged();impl_->target_map->validate_unchanged();
    validate_mtp_shared_target_binding(impl_->model,impl_->sidecar_manifest,impl_->sidecar->binding(),
        impl_->target_manifest,impl_->target->binding(),impl_->shared.target_model_content_identity,impl_->shared);
}
} // namespace strix