#include "strix/gdn_output_admission.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
namespace {
strix::ModelManifest make(const strix::ModelContract& m) {
    strix::ModelManifest manifest;manifest.architecture="fixture";
    for(std::uint32_t layer=0;layer<m.hidden_layers;++layer) {
        if((layer+1U)%m.full_attention_interval==0) continue;
        strix::ManifestTensor t;t.name="blk."+std::to_string(layer)+".ssm_out.weight";
        t.layer=layer;t.role=strix::TensorRole::recurrent_state;t.ggml_type=8;
        const auto width=m.gdn_value_heads*m.gdn_value_head_dim;t.dimensions={width,m.hidden_size};t.byte_size=width/32U*34U*m.hidden_size;
        manifest.tensors.push_back(t);manifest.total_tensor_bytes+=t.byte_size;
    }
    manifest.layout_identity=strix::model_manifest_layout_identity(manifest);return manifest;
}
}
int main() try {
    const strix::ModelContract model;auto original=make(model);
    strix::validate_virtual_q8_gdn_output_manifest(model,original);
    if(original.tensors.size()!=36) throw std::runtime_error("fixture does not cover all36 outputs");
    for(unsigned mode=0;mode<7;++mode) {
        auto manifest=original;auto& t=manifest.tensors.back();
        if(mode==0)t.ggml_type=0;
        if(mode==1)t.dimensions[0]-=32;
        if(mode==2)--t.byte_size;
        if(mode==3)t.layer=47;
        if(mode==4)t.role=strix::TensorRole::attention_projection;
        if(mode==5)manifest.tensors.pop_back();
        if(mode==6)manifest.tensors.push_back(manifest.tensors.back());
        manifest.layout_identity=strix::model_manifest_layout_identity(manifest);
        bool rejected=false;try {strix::validate_virtual_q8_gdn_output_manifest(model,manifest);}
        catch(const std::invalid_argument& e) {
            const std::string message=e.what();rejected=message==(mode==6?"virtual Q8 GDN output tensor is duplicated":
                "virtual Q8 GDN output requires every original aligned Q8_0 output tensor");
        }
        if(!rejected) throw std::runtime_error("wrong late tensor did not fail exact admission contract");
    }
    std::cout<<"GDN output all-layer admission tests passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
