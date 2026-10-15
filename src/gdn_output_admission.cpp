#include "strix/gdn_output_admission.hpp"
#include <limits>
#include <stdexcept>
#include <string>
namespace strix {
void validate_virtual_q8_gdn_output_manifest(const ModelContract& model,const ModelManifest& manifest) {
    if(!model.validate().empty() || model.full_attention_interval==0 || manifest.layout_identity==0 ||
        manifest.layout_identity!=model_manifest_layout_identity(manifest))
        throw std::invalid_argument("virtual Q8 GDN output requires valid model/manifest identities");
    if(model.gdn_value_heads>std::numeric_limits<std::uint64_t>::max()/model.gdn_value_head_dim)
        throw std::overflow_error("GDN output width overflow");
    const auto width=model.gdn_value_heads*model.gdn_value_head_dim;
    if(width==0 || width%32U!=0 || width/32U>std::numeric_limits<std::uint64_t>::max()/34U ||
        width/32U*34U>std::numeric_limits<std::uint64_t>::max()/model.hidden_size)
        throw std::invalid_argument("virtual Q8 GDN output requires bounded aligned widths");
    const auto bytes=width/32U*34U*model.hidden_size;
    std::uint64_t count=0;
    for(std::uint64_t layer=0;layer<model.hidden_layers;++layer) {
        if((layer+1U)%model.full_attention_interval==0) continue;
        const auto name="blk."+std::to_string(layer)+".ssm_out.weight";
        const ManifestTensor* found=nullptr;
        for(const auto& tensor:manifest.tensors) if(tensor.name==name) {
            if(found) throw std::invalid_argument("virtual Q8 GDN output tensor is duplicated");
            found=&tensor;
        }
        if(!found || found->role!=TensorRole::recurrent_state || found->layer!=layer || found->ggml_type!=8U ||
            found->dimensions!=std::vector<std::uint64_t>{width,model.hidden_size} || found->byte_size!=bytes ||
            found->placement.kind!=PlacementKind::local || found->placement.node!=0)
            throw std::invalid_argument("virtual Q8 GDN output requires every original aligned Q8_0 output tensor");
        ++count;
    }
    if(count==0 || count!=model.gdn_layers) throw std::invalid_argument("virtual Q8 GDN output layer count differs");
}
}
