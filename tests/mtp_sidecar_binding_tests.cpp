#include "strix/mtp_sidecar_binding.hpp"
#include "strix/mtp_output_contract.hpp"
#include "strix/compiled_mtp_draft_plan.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {
int failures=0;
void expect(bool ok,const char* message) {if(!ok){std::cerr<<"FAIL: "<<message<<'\n';++failures;}}
template<class Function>void rejects(Function f,const char* message) {
    bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}expect(rejected,message);
}
strix::ModelContentIdentity id(std::uint8_t byte) {strix::ModelContentIdentity v;v.bytes[0]=byte;return v;}
struct Fixture {
    strix::ModelContract model=strix::ModelContract::qwen38_flash_next();
    strix::gguf::FileInfo info;
    strix::MtpSidecarAssetEvidence evidence;
    std::shared_ptr<const void> sidecar_owner=std::make_shared<int>(1),target_owner=std::make_shared<int>(2);
    strix::ModelManifest sidecar,target;
    strix::ModelRuntimeBinding sidecar_runtime,target_runtime;
    strix::MtpSidecarSourceContract source;
    Fixture() {
        info.version=3;info.file_size=2786568256ULL;info.data_offset=10946624;info.alignment=32;
        strix::gguf::MetadataEntry architecture;architecture.key="general.architecture";
        architecture.type=strix::gguf::ValueType::string;architecture.string_value="qwen4exp";info.metadata.push_back(architecture);
        strix::gguf::MetadataEntry shared;shared.key="qwen4exp.nextn_shared_target_tensors";
        shared.type=strix::gguf::ValueType::boolean;shared.boolean_value=true;info.metadata.push_back(shared);
        const auto number=[&](const char* key,std::uint64_t value){strix::gguf::MetadataEntry item;item.key=key;
            item.type=strix::gguf::ValueType::uint32;item.unsigned_value=value;info.metadata.push_back(item);};
        number("qwen4exp.block_count",49);
        number("qwen4exp.nextn_predict_layers",1);
        number("qwen4exp.embedding_length",2560);
        number("qwen4exp.context_length",262144);
        number("qwen4exp.attention.head_count",24);
        number("qwen4exp.attention.head_count_kv",2);
        number("qwen4exp.attention.key_length",256);
        number("qwen4exp.attention.value_length",256);
        number("qwen4exp.expert_count",512);
        number("qwen4exp.expert_used_count",10);
        number("qwen4exp.hyper_connection.count",4);
        number("qwen4exp.hyper_connection.low_rank",320);
        number("qwen4exp.attention.indexer.head_count",4);
        number("qwen4exp.attention.indexer.key_length",128);
        number("qwen4exp.attention.indexer.top_k",2048);
        // Literal header census observed at the pinned sidecar revision. No
        // payload is allocated/read; identities below are synthetic CPU fixtures.
        const auto add=[&](const char* name,std::uint32_t type,std::vector<std::uint64_t> dims,std::uint64_t offset){
            strix::gguf::TensorInfo item;item.name=name;item.dimensions=std::move(dims);item.ggml_type=type;
            item.relative_offset=offset;item.absolute_offset=info.data_offset+offset;
            const auto geometry=*strix::gguf::tensor_type_geometry(type);
            item.byte_size=item.dimensions[0]/geometry.block_elements*geometry.block_bytes;
            for(std::size_t i=1;i<item.dimensions.size();++i)item.byte_size*=item.dimensions[i];
            item.absolute_end=item.absolute_offset+item.byte_size;info.tensors.push_back(std::move(item));};
        add("blk.48.attn_k.weight",8, {2560,512},0ULL);
        add("blk.48.attn_k_norm.weight",0, {256},1392640ULL);
        add("blk.48.attn_output.weight",8, {6144,2560},1393664ULL);
        add("blk.48.attn_q.weight",8, {2560,12288},18105344ULL);
        add("blk.48.attn_q_norm.weight",0, {256},51528704ULL);
        add("blk.48.attn_v.weight",8, {2560,512},51529728ULL);
        add("blk.48.ffn_down_exps.weight",8, {640,2560,512},52922368ULL);
        add("blk.48.ffn_down_shexp.weight",8, {640,2560},944211968ULL);
        add("blk.48.ffn_gate_exps.weight",8, {2560,640,512},945952768ULL);
        add("blk.48.ffn_gate_inp.weight",0, {2560,512},1837242368ULL);
        add("blk.48.ffn_gate_inp_shexp.weight",0, {2560},1842485248ULL);
        add("blk.48.ffn_gate_shexp.weight",8, {2560,640},1842495488ULL);
        add("blk.48.ffn_up_exps.weight",8, {2560,640,512},1844236288ULL);
        add("blk.48.ffn_up_shexp.weight",8, {2560,640},2735525888ULL);
        add("blk.48.hc_attn_down.weight",8, {10240,320},2737266688ULL);
        add("blk.48.hc_attn_inject.weight",8, {10240,4},2740748288ULL);
        add("blk.48.hc_attn_norm.weight",0, {10240},2740791808ULL);
        add("blk.48.hc_attn_up.weight",8, {320,10240},2740832768ULL);
        add("blk.48.hc_ffn_down.weight",8, {10240,320},2744314368ULL);
        add("blk.48.hc_ffn_inject.weight",8, {10240,4},2747795968ULL);
        add("blk.48.hc_ffn_norm.weight",0, {10240},2747839488ULL);
        add("blk.48.hc_ffn_up.weight",8, {320,10240},2747880448ULL);
        add("blk.48.indexer.k_norm.weight",0, {128},2751362048ULL);
        add("blk.48.indexer.k_proj.weight",30, {2560,128},2751362560ULL);
        add("blk.48.indexer.q_norm.weight",0, {128},2752017920ULL);
        add("blk.48.indexer.q_proj.weight",30, {2560,512},2752018432ULL);
        add("blk.48.nextn.eh_proj.weight",8, {5120,2560},2754639872ULL);
        add("blk.48.nextn.enorm.weight",0, {2560},2768566272ULL);
        add("blk.48.nextn.hc_head_down.weight",8, {10240,320},2768576512ULL);
        add("blk.48.nextn.hc_head_norm.weight",0, {10240},2772058112ULL);
        add("blk.48.nextn.hc_head_up.weight",8, {320,10240},2772099072ULL);
        add("blk.48.nextn.hnorm.weight",0, {10240},2775580672ULL);
        evidence.repository="fixture-source";evidence.revision="fixture-revision";evidence.relative_path="MTP/shared-Q8_0.gguf";
        evidence.declared_file_bytes=info.file_size;evidence.declared_file_sha256=id(1);
        evidence.conversion=strix::MtpSidecarConversionProfile::effective_norm_gain_embedding_first_packed_v1;
        evidence.converter_reference_revision="fixture-reference";evidence.converter_reference_sha256=id(2);evidence.norm_converter_reference_sha256=id(5);
        sidecar=strix::build_mtp_sidecar_manifest(model,info,evidence);
        source=strix::inspect_mtp_sidecar_source(model,info,evidence);
        target.architecture="qwen4exp";
        for(const auto* name:{"token_embd.weight","output.weight"}) {
            strix::ManifestTensor tensor;tensor.name=name;tensor.dimensions={2560,248320};tensor.ggml_type=8;
            tensor.byte_size=675430400;tensor.absolute_offset=target.total_tensor_bytes;
            tensor.role=target.tensors.empty()?strix::TensorRole::embedding:strix::TensorRole::output_head;
            target.total_tensor_bytes+=tensor.byte_size;target.tensors.push_back(std::move(tensor));
        }
        target.layout_identity=strix::model_manifest_layout_identity(target);
        sidecar_runtime=runtime(sidecar,id(3));target_runtime=runtime(target,id(4));
    }
    static strix::ModelRuntimeBinding runtime(const strix::ModelManifest& manifest,strix::ModelContentIdentity content) {
        strix::ModelRuntimeBinding result;result.manifest_layout_identity=manifest.layout_identity;result.model_content_identity=content;
        std::uintptr_t address=0x100000000ULL;
        for(std::size_t i=0;i<manifest.tensors.size();++i) {
            result.tensors.push_back({static_cast<std::uint32_t>(i),strix::RuntimeTensorStorage::resident,address,
                manifest.tensors[i].byte_size,static_cast<std::uint64_t>(i)+1U,0});
            address+=(manifest.tensors[i].byte_size+255U)&~std::uintptr_t{255};
        }
        return result;
    }
    void mark_full_verified() {
        evidence.verified_file_sha256=evidence.declared_file_sha256;
        evidence.verified_model_content_identity=sidecar_runtime.model_content_identity;
        source=strix::inspect_mtp_sidecar_source(model,info,evidence);
    }
    strix::MtpSharedTargetBinding bind() const {
        return strix::bind_mtp_shared_target_tensors(model,source,sidecar,sidecar_runtime,target,target_runtime,id(4),sidecar_owner,target_owner);
    }
};
}
int main() {
    Fixture fixture;
    expect(fixture.sidecar.tensors.size()==32 && fixture.source.shared_target_tensors &&
        fixture.source.norm_weight_encoding==strix::MtpNormWeightEncoding::effective_scale,
        "bounded metadata describes the actual sidecar source profile without inventing full-file verification");
    rejects([&]{(void)fixture.bind();},"declared LFS/header identity cannot authorize live target borrowing");
    fixture.mark_full_verified();auto shared=fixture.bind();
    strix::validate_mtp_shared_target_binding(fixture.model,fixture.sidecar,fixture.sidecar_runtime,
        fixture.target,fixture.target_runtime,id(4),shared);
    const auto draft=strix::build_mtp_draft_runtime_binding(fixture.sidecar,fixture.sidecar_runtime);
    const auto audit=strix::audit_mtp_draft_tensor_families(fixture.sidecar,draft);
    const auto output=strix::bind_mtp_output_head(fixture.model,fixture.sidecar,audit,shared);
    expect(audit.structurally_complete() && output.head_binding_resolved &&
        output.lm_head_source==strix::MtpLmHeadSource::target_shared_explicit &&
        output.lm_head_manifest_layout_identity==fixture.target.layout_identity && output.lm_head_tensor_index==1,
        "explicit shared flag resolves a target-namespace output head without adding synthetic sidecar tensors");
    const auto ordinary=strix::compile_mtp_draft_plan(fixture.model,fixture.sidecar,draft);
    const auto compiled=strix::compile_mtp_draft_plan(fixture.model,fixture.sidecar,draft,&shared);
    expect(!ordinary.output.head_binding_resolved && compiled.output.head_binding_resolved &&
        compiled.plan_identity!=ordinary.plan_identity && !compiled.executable,
        "only explicit verified borrowing closes binding, never semantic/native qualification");
    for(unsigned fault=0;fault<8;++fault) {
        auto info=fixture.info;auto asset=fixture.evidence;
        if(fault==0) info.metadata[1].boolean_value=false;
        if(fault==1) info.metadata.erase(info.metadata.begin()+1);
        if(fault==2) asset.verified_file_sha256=id(9);
        if(fault==3) asset.conversion=strix::MtpSidecarConversionProfile::unresolved;
        if(fault==4) info.tensors.back().dimensions={2560};
        if(fault==5) info.tensors[15].ggml_type=0;
        if(fault==6) info.tensors[1].relative_offset=0;
        if(fault==7) info.tensors.push_back(info.tensors.back());
        rejects([&]{(void)strix::inspect_mtp_sidecar_source(fixture.model,info,asset);},
            "sharing flag/hash/encoding/wide-normalization/census/ranges reject incompatible source metadata");
    }
    rejects([&]{(void)strix::bind_mtp_shared_target_tensors(fixture.model,fixture.source,fixture.sidecar,fixture.sidecar_runtime,
        fixture.target,fixture.target_runtime,id(9),fixture.sidecar_owner,fixture.target_owner);},"foreign target content rejects despite matching geometry");
    const auto original=fixture.target_runtime.tensors[1].allocation_generation;
    fixture.target_runtime.tensors[1].allocation_generation=original+1;
    rejects([&]{strix::validate_mtp_shared_target_binding(fixture.model,fixture.sidecar,fixture.sidecar_runtime,
        fixture.target,fixture.target_runtime,id(4),shared);},"borrowed target allocation generation cannot silently change");
    fixture.target_runtime.tensors[1].allocation_generation=original;
    const auto sidecar_generation=fixture.sidecar_runtime.tensors[0].allocation_generation;
    fixture.sidecar_runtime.tensors[0].allocation_generation=sidecar_generation+1;
    rejects([&]{strix::validate_mtp_shared_target_binding(fixture.model,fixture.sidecar,fixture.sidecar_runtime,
        fixture.target,fixture.target_runtime,id(4),shared);},"sidecar physical relocation also invalidates its live descriptor");
    fixture.sidecar_runtime.tensors[0].allocation_generation=sidecar_generation;
    auto stale=shared;stale.actual_target_owner.reset();
    rejects([&]{strix::validate_mtp_shared_target_binding(fixture.model,fixture.sidecar,fixture.sidecar_runtime,
        fixture.target,fixture.target_runtime,id(4),stale);},"sharing retains both actual model owners");
    return failures?1:0;
}