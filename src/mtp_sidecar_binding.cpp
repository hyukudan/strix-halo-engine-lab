#include "strix/mtp_sidecar_binding.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace strix {
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::invalid_argument(message);}
std::uint64_t mul(std::uint64_t a,std::uint64_t b) {
    if(a && b>UINT64_MAX/a) throw std::overflow_error("MTP sidecar extent overflow");
    return a*b;
}
struct Hash {
    std::uint64_t n=14695981039346656037ULL;
    void word(std::uint64_t v) {for(unsigned i=0;i<64;i+=8) n=(n^((v>>i)&255U))*1099511628211ULL;}
    void bytes(const ModelContentIdentity& id) {for(auto v:id.bytes) n=(n^v)*1099511628211ULL;}
    void text(std::string_view v) {word(v.size());for(char b:v) n=(n^static_cast<unsigned char>(b))*1099511628211ULL;}
    std::uint64_t done() const {return n?n:1U;}
};
void model_scope(const ModelContract& m) {
    require(m.validate().empty() && m.hidden_layers==48 && m.hidden_size==2560 && m.hc_count==4 && m.hc_lowrank==320 &&
        m.attention_heads==24 && m.kv_heads==2 && m.head_dim==256 && m.num_experts==512 && m.num_experts_per_token==10 && m.moe_intermediate_size==640 &&
        m.shared_expert_intermediate_size==640 && m.indexer_n_heads==4 && m.indexer_kv_heads==1 &&
        m.indexer_head_dim==128 && m.indexer_budget==2048,
        "MTP sidecar conversion profile requires the canonical model envelope");
}
struct Spec {std::string name;std::uint32_t type;std::vector<std::uint64_t> dims;};
std::vector<Spec> specs(const ModelContract& m) {
    const auto p="blk."+std::to_string(m.hidden_layers)+".";
    const auto h=m.hidden_size,w=mul(h,m.hc_count),r=m.hc_lowrank,k=m.head_dim,
        q=mul(m.attention_heads,k),kv=mul(m.kv_heads,k),f=m.moe_intermediate_size,s=m.shared_expert_intermediate_size;
    return {
        {p+"attn_k.weight",8,{h,kv}},{p+"attn_k_norm.weight",0,{k}},
        {p+"attn_output.weight",8,{q,h}},{p+"attn_q.weight",8,{h,mul(q,2)}},
        {p+"attn_q_norm.weight",0,{k}},{p+"attn_v.weight",8,{h,kv}},
        {p+"ffn_down_exps.weight",8,{f,h,m.num_experts}},{p+"ffn_down_shexp.weight",8,{s,h}},
        {p+"ffn_gate_exps.weight",8,{h,f,m.num_experts}},{p+"ffn_gate_inp.weight",0,{h,m.num_experts}},
        {p+"ffn_gate_inp_shexp.weight",0,{h}},{p+"ffn_gate_shexp.weight",8,{h,s}},
        {p+"ffn_up_exps.weight",8,{h,f,m.num_experts}},{p+"ffn_up_shexp.weight",8,{h,s}},
        {p+"hc_attn_down.weight",8,{w,r}},{p+"hc_attn_inject.weight",8,{w,m.hc_count}},
        {p+"hc_attn_norm.weight",0,{w}},{p+"hc_attn_up.weight",8,{r,w}},
        {p+"hc_ffn_down.weight",8,{w,r}},{p+"hc_ffn_inject.weight",8,{w,m.hc_count}},
        {p+"hc_ffn_norm.weight",0,{w}},{p+"hc_ffn_up.weight",8,{r,w}},
        {p+"indexer.k_norm.weight",0,{m.indexer_head_dim}},
        {p+"indexer.k_proj.weight",30,{h,m.indexer_head_dim}},
        {p+"indexer.q_norm.weight",0,{m.indexer_head_dim}},
        {p+"indexer.q_proj.weight",30,{h,mul(m.indexer_n_heads,m.indexer_head_dim)}},
        {p+"nextn.eh_proj.weight",8,{mul(h,2),h}},{p+"nextn.enorm.weight",0,{h}},
        {p+"nextn.hc_head_down.weight",8,{w,r}},{p+"nextn.hc_head_norm.weight",0,{w}},
        {p+"nextn.hc_head_up.weight",8,{r,w}},{p+"nextn.hnorm.weight",0,{w}}
    };
}
std::uint64_t tensor_bytes(const Spec& spec) {
    const auto geometry=gguf::tensor_type_geometry(spec.type);
    require(geometry && !spec.dims.empty() && spec.dims[0]%geometry->block_elements==0,
        "MTP sidecar tensor block geometry is unsupported");
    auto bytes=mul(spec.dims[0]/geometry->block_elements,geometry->block_bytes);
    for(std::size_t i=1;i<spec.dims.size();++i) bytes=mul(bytes,spec.dims[i]);
    return bytes;
}
void asset_valid(const MtpSidecarAssetEvidence& a,bool full) {
    require(!a.repository.empty() && a.repository.size()<=512 && !a.revision.empty() && a.revision.size()<=128 &&
        !a.relative_path.empty() && a.relative_path.size()<=1024 && a.declared_file_sha256.valid() && a.declared_file_bytes &&
        a.converter_reference_sha256.valid() && a.norm_converter_reference_sha256.valid() && !a.converter_reference_revision.empty() && a.converter_reference_revision.size()<=128 &&
        a.conversion==MtpSidecarConversionProfile::effective_norm_gain_embedding_first_packed_v1,
        "MTP sidecar needs explicit asset and conversion reference provenance");
    if(a.verified_file_sha256)
        require(a.verified_file_sha256->valid() && *a.verified_file_sha256==a.declared_file_sha256,
            "MTP sidecar verified full-file digest differs from the declared asset");
    if(full) require(a.verified_file_sha256.has_value() && a.verified_model_content_identity.valid(),
        "MTP live sharing requires a fully hashed file and its loader content identity, not a bounded header/LFS declaration");
}
std::uint64_t source_id(const MtpSidecarSourceContract& c) {
    const auto& a=c.asset;Hash hash;hash.text("mtp-sidecar-source-v1");
    hash.text(a.repository);hash.text(a.revision);hash.text(a.relative_path);hash.bytes(a.declared_file_sha256);
    hash.word(a.verified_file_sha256.has_value());if(a.verified_file_sha256) hash.bytes(*a.verified_file_sha256);
    hash.bytes(a.verified_model_content_identity);hash.word(a.declared_file_bytes);hash.word(static_cast<std::uint32_t>(a.conversion));
    hash.bytes(a.converter_reference_sha256);hash.bytes(a.norm_converter_reference_sha256);hash.text(a.converter_reference_revision);
    hash.word(c.manifest_layout_identity);hash.word(c.shared_target_tensors);hash.word(static_cast<std::uint32_t>(c.norm_weight_encoding));
    return hash.done();
}
void manifest_valid(const ModelContract& m,const ModelManifest& manifest) {
    require(manifest.architecture=="qwen4exp" && manifest.layout_identity && manifest.layout_identity==model_manifest_layout_identity(manifest),
        "MTP sidecar manifest identity is stale");
    const auto expected=specs(m);require(manifest.tensors.size()==expected.size(),"MTP sidecar tensor census is not the closed source profile");
    std::set<std::string> seen;
    std::uint64_t total=0;
    for(const auto& tensor:manifest.tensors) {
        require(seen.insert(tensor.name).second,"MTP sidecar tensor name is duplicated");
        const auto spec=std::find_if(expected.begin(),expected.end(),[&](const auto& item){return item.name==tensor.name;});
        require(spec!=expected.end() && tensor.role==TensorRole::mtp && tensor.ggml_type==spec->type && tensor.dimensions==spec->dims &&
            tensor.byte_size==tensor_bytes(*spec) && tensor.residency==ResidencyPolicy::resident &&
            tensor.placement.kind==PlacementKind::local && tensor.placement.node==0 && tensor.layer==m.hidden_layers,
            "MTP source tensor role, geometry, format or placement differs from its actual sidecar profile");
        if(tensor.byte_size>UINT64_MAX-total) throw std::overflow_error("MTP sidecar byte census overflow");
        total+=tensor.byte_size;
    }
    require(total==manifest.total_tensor_bytes && !manifest.file_backed_candidate_bytes && !manifest.bounded_cache_candidate_bytes &&
        !manifest.unknown_tensor_count,"MTP sidecar manifest aggregate census differs");
}
std::uint32_t target_index(const ModelManifest& manifest,std::string_view name,TensorRole role,
    const ModelContract& m) {
    std::optional<std::uint32_t> result;
    for(std::size_t i=0;i<manifest.tensors.size();++i) if(manifest.tensors[i].name==name) {
        require(!result && i<=UINT32_MAX,"MTP shared target tensor name is ambiguous");
        const auto& tensor=manifest.tensors[i];
        require(tensor.role==role && tensor.dimensions==std::vector<std::uint64_t>{m.hidden_size,m.vocab_size} && tensor.ggml_type==8,
            "MTP shared target tensor has unsupported role/shape/type");result=static_cast<std::uint32_t>(i);
    }
    require(result.has_value(),"explicit MTP sharing requires the actual target embedding and output projection");return *result;
}
std::uint64_t shared_id(const MtpSharedTargetBinding& b,bool physical) {
    Hash hash;hash.text("mtp-explicit-shared-target-v1");hash.word(b.source.source_identity);
    hash.word(b.sidecar_manifest_layout_identity);hash.word(b.target_manifest_layout_identity);
    hash.bytes(b.sidecar_model_content_identity);hash.bytes(b.target_model_content_identity);
    hash.word(b.target_embedding_tensor_index);hash.word(b.target_output_tensor_index);
    if(physical) for(const auto& tensor:b.sidecar_runtime->tensors) {
        hash.word(tensor.tensor_index);hash.word(static_cast<std::uint32_t>(tensor.storage));
        hash.word(tensor.address);hash.word(tensor.bytes);hash.word(tensor.allocation_generation);hash.word(tensor.owner_node);
    }
    if(physical) for(const auto index:{b.target_embedding_tensor_index,b.target_output_tensor_index}) {
        const auto* tensor=find_runtime_tensor_binding(*b.target_runtime,index);
        require(tensor!=nullptr,"MTP borrowed target tensor lost its runtime view");
        hash.word(static_cast<std::uint32_t>(tensor->storage));hash.word(tensor->address);hash.word(tensor->bytes);
        hash.word(tensor->allocation_generation);hash.word(tensor->owner_node);
    }
    return hash.done();
}
} // namespace

ModelManifest build_mtp_sidecar_manifest(const ModelContract& m,const gguf::FileInfo& info,const MtpSidecarAssetEvidence& asset) {
    model_scope(m);asset_valid(asset,false);
    require(info.version==3 && info.file_size==asset.declared_file_bytes && info.data_offset<=info.file_size && info.alignment>=16 &&
        !(info.alignment&(info.alignment-1U)),"MTP sidecar header extent/alignment differs from its asset");
    std::set<std::string> metadata_names;
    for(const auto& entry:info.metadata) require(metadata_names.insert(entry.key).second,"MTP sidecar metadata key is duplicated");
    const auto* architecture=info.find_metadata("general.architecture");
    require(architecture && architecture->type==gguf::ValueType::string && architecture->string_value=="qwen4exp",
        "MTP sidecar architecture metadata is missing or foreign");
    const auto* shared=info.find_metadata("qwen4exp.nextn_shared_target_tensors");
    require(shared && shared->type==gguf::ValueType::boolean && shared->boolean_value && *shared->boolean_value,
        "MTP target sharing requires the explicit true sidecar metadata flag");
    const auto number=[&](const char* key,std::uint64_t expected) {
        const auto* entry=info.find_metadata(key);
        require(entry && (entry->type==gguf::ValueType::uint32 || entry->type==gguf::ValueType::uint64) &&
            entry->unsigned_value && *entry->unsigned_value==expected,"MTP sidecar structural metadata differs from target");
    };
    number("qwen4exp.block_count",m.hidden_layers+1U);number("qwen4exp.nextn_predict_layers",1);
    number("qwen4exp.embedding_length",m.hidden_size);number("qwen4exp.context_length",m.max_position_embeddings);
    number("qwen4exp.attention.head_count",m.attention_heads);number("qwen4exp.attention.head_count_kv",m.kv_heads);
    number("qwen4exp.attention.key_length",m.head_dim);number("qwen4exp.attention.value_length",m.head_dim);
    number("qwen4exp.expert_count",m.num_experts);number("qwen4exp.expert_used_count",m.num_experts_per_token);
    number("qwen4exp.hyper_connection.count",m.hc_count);number("qwen4exp.hyper_connection.low_rank",m.hc_lowrank);
    number("qwen4exp.attention.indexer.head_count",m.indexer_n_heads);number("qwen4exp.attention.indexer.key_length",m.indexer_head_dim);
    number("qwen4exp.attention.indexer.top_k",m.indexer_budget);
    if(m.rotary_identity().descriptor().mode()==RotaryMode::yarn) {
        number("qwen4exp.rope.dimension_count",m.rotary_dim());
        const auto* theta=info.find_metadata("qwen4exp.rope.freq_base");
        const auto* sections=info.find_metadata("qwen4exp.rope.dimension_sections");
        // GGUF writers store the M-RoPE sections as INT32 arrays (the shipped target and sidecar both do); accept
        // either integer signedness, never a different value.
        require(theta && theta->floating_value && *theta->floating_value==m.rope_theta && sections &&
            (sections->unsigned_array==std::vector<std::uint64_t>{11,11,10,0} ||
             sections->signed_array==std::vector<std::int64_t>{11,11,10,0}),
            "YaRN MTP sidecar rotary theta/sections differ from shared target contract");
    }
    auto result=build_model_manifest(info);
    std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges;
    for(std::size_t i=0;i<result.tensors.size();++i) {
        auto& tensor=result.tensors[i];const auto& physical=info.tensors[i];
        require(physical.relative_offset<=UINT64_MAX-info.data_offset && physical.absolute_offset==info.data_offset+physical.relative_offset &&
            tensor.absolute_offset==physical.absolute_offset && tensor.absolute_offset%info.alignment==0 &&
            tensor.absolute_offset<=info.file_size && tensor.byte_size<=info.file_size-tensor.absolute_offset,
            "MTP sidecar tensor escapes the actual file range");
        tensor.role=TensorRole::mtp;tensor.residency=ResidencyPolicy::resident;
        ranges.emplace_back(tensor.absolute_offset,tensor.absolute_offset+tensor.byte_size);
    }
    std::sort(ranges.begin(),ranges.end());
    for(std::size_t i=1;i<ranges.size();++i) require(ranges[i-1].second<=ranges[i].first,"MTP sidecar tensor ranges overlap");
    result.file_backed_candidate_bytes=result.bounded_cache_candidate_bytes=result.unknown_tensor_count=0;
    result.layout_identity=model_manifest_layout_identity(result);manifest_valid(m,result);return result;
}
MtpSidecarSourceContract inspect_mtp_sidecar_source(const ModelContract& m,const gguf::FileInfo& info,const MtpSidecarAssetEvidence& asset) {
    const auto manifest=build_mtp_sidecar_manifest(m,info,asset);
    MtpSidecarSourceContract result;result.asset=asset;result.manifest_layout_identity=manifest.layout_identity;
    result.shared_target_tensors=true;result.norm_weight_encoding=MtpNormWeightEncoding::effective_scale;
    result.source_identity=source_id(result);return result;
}
void validate_mtp_sidecar_source(const ModelContract& m,const ModelManifest& manifest,
    const MtpSidecarSourceContract& source,bool full) {
    model_scope(m);asset_valid(source.asset,full);manifest_valid(m,manifest);
    require(source.shared_target_tensors && source.manifest_layout_identity==manifest.layout_identity &&
        source.norm_weight_encoding==MtpNormWeightEncoding::effective_scale && source.source_identity==source_id(source),
        "MTP sidecar source evidence is stale, ambiguous or has an unsupported norm encoding");
}
MtpSharedTargetBinding bind_mtp_shared_target_tensors(const ModelContract& m,const MtpSidecarSourceContract& source,
    const ModelManifest& sidecar,const ModelRuntimeBinding& sidecar_runtime,const ModelManifest& target,
    const ModelRuntimeBinding& target_runtime,const ModelContentIdentity& expected_target,
    std::shared_ptr<const void> sidecar_owner,std::shared_ptr<const void> target_owner) {
    validate_mtp_sidecar_source(m,sidecar,source,true);
    validate_model_runtime_binding(sidecar,sidecar_runtime);validate_model_runtime_binding(target,target_runtime);
    require(sidecar_owner && target_owner && expected_target.valid() && target_runtime.model_content_identity==expected_target &&
        sidecar_runtime.model_content_identity==source.asset.verified_model_content_identity,
        "MTP sharing requires exact expected model content and the two actual retained model owners");
    for(const auto& view:sidecar_runtime.tensors)
        require(runtime_tensor_gpu_visible(view.storage) && view.address%16U==0 && view.owner_node==0,
            "MTP sidecar view is not locally GPU visible and aligned");
    MtpSharedTargetBinding result;result.source=source;
    result.target_embedding_tensor_index=target_index(target,"token_embd.weight",TensorRole::embedding,m);
    result.target_output_tensor_index=target_index(target,"output.weight",TensorRole::output_head,m);
    for(const auto index:{result.target_embedding_tensor_index,result.target_output_tensor_index}) {
        const auto* view=find_runtime_tensor_binding(target_runtime,index);
        require(view && runtime_tensor_gpu_visible(view->storage) && view->address%16U==0 && view->owner_node==0,
            "MTP shared target view is not locally GPU visible and aligned");
    }
    result.sidecar_manifest_layout_identity=sidecar.layout_identity;result.target_manifest_layout_identity=target.layout_identity;
    result.sidecar_model_content_identity=sidecar_runtime.model_content_identity;result.target_model_content_identity=target_runtime.model_content_identity;
    result.actual_sidecar_owner=std::move(sidecar_owner);result.actual_target_owner=std::move(target_owner);
    result.sidecar_manifest=&sidecar;result.sidecar_runtime=&sidecar_runtime;result.target_manifest=&target;result.target_runtime=&target_runtime;
    result.compatibility_identity=shared_id(result,false);result.physical_binding_identity=shared_id(result,true);return result;
}
void validate_mtp_shared_target_binding(const ModelContract& m,const ModelManifest& sidecar,const ModelRuntimeBinding& sidecar_runtime,
    const ModelManifest& target,const ModelRuntimeBinding& target_runtime,const ModelContentIdentity& expected_target,
    const MtpSharedTargetBinding& binding) {
    require(binding.sidecar_manifest==&sidecar && binding.sidecar_runtime==&sidecar_runtime && binding.target_manifest==&target &&
        binding.target_runtime==&target_runtime,"MTP borrowed source descriptors were rebound");
    const auto expected=bind_mtp_shared_target_tensors(m,binding.source,sidecar,sidecar_runtime,target,target_runtime,expected_target,
        binding.actual_sidecar_owner,binding.actual_target_owner);
    require(binding.target_embedding_tensor_index==expected.target_embedding_tensor_index && binding.target_output_tensor_index==expected.target_output_tensor_index &&
        binding.sidecar_manifest_layout_identity==expected.sidecar_manifest_layout_identity && binding.target_manifest_layout_identity==expected.target_manifest_layout_identity &&
        binding.sidecar_model_content_identity==expected.sidecar_model_content_identity && binding.target_model_content_identity==expected.target_model_content_identity &&
        binding.compatibility_identity==expected.compatibility_identity && binding.physical_binding_identity==expected.physical_binding_identity,
        "MTP shared target binding is stale or references a different physical/content generation");
}
} // namespace strix