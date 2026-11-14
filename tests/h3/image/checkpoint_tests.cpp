#include "h3/block/image_policy.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
using namespace strix::h3;
namespace b=strix::h3::block;
void require(bool ok) { if(!ok) throw std::runtime_error("checkpoint host assertion"); }
template<class F> void rejects(F f) { bool caught=false;try { f(); } catch(const std::exception&) { caught=true; }require(caught); }
int main(int argc,char** argv) try {
    if(argc==4 && std::string(argv[1])=="--decode") {
        b::SafeTensors st(argv[2],b::CheckpointNames::image_dit);
        const std::string p="blocks.0.attn.qkv_proj";
        const auto& t=st.at(p+".weight");
        const auto n=std::uint32_t(t.shape.at(0)),k=std::uint32_t(t.shape.at(1))*2;
        auto q=b::detail::read_image_linear(st,p,n,k);
        require(q.operand==gemm::Operand::w4a8 && q.bf16.empty() && q.w.size()==std::size_t(n)*k/2);
        b::QLinear unpacked;unpacked.n=n;unpacked.k=k;unpacked.ws=q.ws;unpacked.w.resize(std::size_t(n)*k);
        for(std::size_t i=0;i<unpacked.w.size();++i) unpacked.w[i]=q.packed.at(q.w.data(),i);
        const auto prepared=gemm::prepare_w4a8(q.packed);
        require(!prepared.pairs.empty());
        for(std::size_t i=0;i<unpacked.w.size();++i) {
            const auto u=std::uint8_t(q.w[i/2]);
            const auto pair=prepared.pairs[unsigned(prepared.scale_ids[i>>prepared.group_shift])*256U+u];
            require(std::uint8_t(unpacked.w[i])==std::uint8_t(pair>>(8U*(i%2))));
        }
        std::ofstream out(argv[3],std::ios::binary);
        out.write(reinterpret_cast<const char*>(unpacked.w.data()),std::streamsize(unpacked.w.size()));
        for(std::size_t i=0;i<unpacked.w.size();++i) {
            const float f=float(unpacked.w[i])*q.ws[i/k];out.write(reinterpret_cast<const char*>(&f),sizeof(f));
        }
        require(bool(out));
        std::vector<std::uint16_t> x(std::size_t(3)*k),a(std::size_t(3)*n),c(a.size());
        for(std::size_t i=0;i<x.size();++i) x[i]=gemm::f32_to_bf16(float(int(i%61)-30)/16.0F);
        b::HostLinear op;op.run(q,x.data(),3,a.data());op.run(unpacked,x.data(),3,c.data());
        require(a==c);return 0;
    }
    if(argc!=2) throw std::invalid_argument("checkpoint_tests FIXTURE_DIR or --decode FILE OUT");
    const std::string root=argv[1];
    const std::vector<std::string> names={"adaln_t_table","rope.inv_freq","blocks.0.attn.qkv_proj.weight",
        "blocks.49.mlp.fc2.weight","blocks.0.attn.qkv_proj.weight_s_rel","blocks.0.attn.qkv_proj.weight_s_channel",
        "blocks.0.attn.qkv_proj.weight_codebook","blocks.0.attn.qkv_proj.comfy_quant","blocks.0.adaln_proj.linear.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight","condition_proj.weight","final_layer.video_out.weight"};
    for(const auto& name:names) {
        require(b::checkpoint_name(name,b::CheckpointNames::image_dit)==name);
        for(const auto* prefix:{"model.diffusion_model.","diffusion_model.","transformer."})
            require(b::checkpoint_name(std::string(prefix)+name,b::CheckpointNames::image_dit)==name);
        require(b::checkpoint_name("transformer."+name,b::CheckpointNames::unchanged)=="transformer."+name);
    }
    for(const auto* prefix:{"first_stage_model.","vae."})
        require(b::checkpoint_name(std::string(prefix)+"decoder.transformer_blocks.23.ff.w2.weight",b::CheckpointNames::image_vae)=="decoder.transformer_blocks.23.ff.w2.weight");
    b::SafeTensors mapped(root+"/mapped.safetensors",b::CheckpointNames::image_dit);
    require(mapped.read_vec<float>("rope.inv_freq",b::StDtype::f32)==std::vector<float>{1,2});
    b::SafeTensors shards(root+"/shards",b::CheckpointNames::image_dit);
    require(shards.read_vec<float>("rope.inv_freq",b::StDtype::f32)==std::vector<float>{1,2});
    rejects([&] { b::SafeTensors bad(root+"/collision.safetensors",b::CheckpointNames::image_dit); });
    b::SafeTensors vae(root+"/vae.safetensors",b::CheckpointNames::image_vae);
    const auto vf=b::detail::read_as_f32(vae,"latents_mean",false);
    require(vf==std::vector<float>({1,-2,0.5F,65504}));
    b::SafeTensors i8(root+"/int8.safetensors");
    auto old=b::detail::read_qlinear(i8,"blocks.0.attn.qkv_proj",2,256);
    auto image=b::detail::read_image_linear(i8,"blocks.0.attn.qkv_proj",2,256);
    require(old.w==image.w && old.ws==image.ws && image.operand==gemm::Operand::int8_convrot);
    require(b::ImageConfig::weights_policy(i8).image_operands && !b::LoadPolicy{}.image_operands);
    b::SafeTensors curve(root+"/curve.safetensors");
    require(!b::ImageConfig::weights_policy(curve).direct_timestep);
    b::GlobalWeights g;g.audio_enabled=false;g.basis=2;g.curve_grid=2;g.adaln_t_table={1,2,3,4};
    g.final_adaln={1,2,{1,2},{0}};
    const auto cfg=b::ImageConfig::load(root+"/fixed-image-policy");
    b::ImageBlockSource source(curve,g,cfg);
    require(g.basis==7 && !g.direct_timestep && g.discrete_times.size()==7);
    require(g.final_adaln.w.size()==7 && g.final_adaln.w.front()==5);
    rejects([&] { (void)b::curve_t_emb(g,{0.123456F}); });
    gemm::PackedW4A8 bad;bad.scales_f32={1};rejects([&] {bad.validate(1,256,128);});
    require(gemm::e4m3fn(0x7e)==448 && gemm::e4m3fn(1)==1.0F/512 && std::isnan(gemm::e4m3fn(0x7f)));
    gemm::PackedW4A8 many;
    for(unsigned i=0;i<257;++i) many.scales_f32.push_back(float(i));
    require(gemm::prepare_w4a8(many).pairs.empty());
    many.group_size=48; require(gemm::prepare_w4a8(many).pairs.empty());
    std::cout<<"checkpoint mapping, FP16, curve policy and legacy INT8 contracts passed\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
