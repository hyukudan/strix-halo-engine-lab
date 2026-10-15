#include "strix/lane_chain_hyper_output.hpp"
#include <limits>
#include <stdexcept>
#include <string>
namespace strix {
std::uint64_t lane_chain_hyper_output_lds(std::uint64_t streams,std::uint64_t rank) noexcept {
    if(streams<2 || streams>8 || rank==0 || rank>512 || rank%32U)return 0;
    const auto bytes=streams*(34U*rank+128U);
    return bytes<=65536U?bytes:0U;
}
bool lane_chain_hyper_output_eligible(const ManifestTensor& up,std::uint64_t hidden,std::uint64_t streams,std::uint64_t rank,bool registered) noexcept {
    if(!lane_chain_hyper_output_lds(streams,rank) || hidden==0 || hidden%32U || hidden>std::numeric_limits<std::uint32_t>::max()/streams ||
       up.ggml_type!=8 || up.placement.kind!=PlacementKind::local || up.placement.node!=0 ||
       up.dimensions.size()!=2 || up.dimensions[0]!=rank || up.dimensions[1]!=hidden*streams ||
       (registered && up.absolute_offset%16U))return false;
    return up.byte_size==(rank/32U)*34U*hidden*streams;
}
std::vector<std::uint32_t> lane_chain_hyper_output_steps(const ModelContract& model,const ModelManifest& manifest,bool registered) {
    if(!model.validate().empty() || !manifest.layout_identity || manifest.layout_identity!=model_manifest_layout_identity(manifest))
        throw std::invalid_argument("lane-chain HC output requires valid model/manifest identities");
    std::vector<std::uint32_t> eligible;
    for(const auto& step:build_serial_decode_execution_plan(model).steps){
        if(step.op!=SerialOp::attn_hyper_mix && step.op!=SerialOp::mlp_hyper_mix && step.op!=SerialOp::final_hyper_mix)continue;
        const auto name=step.op==SerialOp::final_hyper_mix?std::string("output_hc_up.weight"):
            "blk."+std::to_string(*step.layer)+(step.op==SerialOp::attn_hyper_mix?".hc_attn_up.weight":".hc_ffn_up.weight");
        const ManifestTensor* found=nullptr;
        for(const auto& tensor:manifest.tensors)if(tensor.name==name){if(found)throw std::invalid_argument("lane-chain HC output duplicate up tensor");found=&tensor;}
        if(!found)throw std::invalid_argument("lane-chain HC output missing up tensor");
        if(lane_chain_hyper_output_eligible(*found,model.hidden_size,model.hc_count,model.hc_lowrank,registered))eligible.push_back(step.id);
    }
    if(eligible.empty())throw std::invalid_argument("lane-chain HC output has no eligible steps");
    return eligible;
}
}
