#include "h3/block/image_policy.hpp"
#include "h3/vae/image.hpp"
#include "h3/te/te_vision_host.hpp"
#include <iostream>
#include <cstring>
using namespace strix::h3;
namespace {
int checks=0;
void require(bool b) { ++checks; if(!b) throw std::runtime_error("host assertion "+std::to_string(checks)); }
template<class F> void rejects(F f) { bool caught=false; try { f(); } catch(const std::exception&) { caught=true; } require(caught); }
fixture::Json config() {
    return fixture::JsonParser(R"({"audio_enabled":false,"hidden_size":5376,"num_layers":50,"token_refiner_num_layers":2,
    "num_attention_heads":56,"attention_head_dim":128,"ffn_hidden_size":14336,"latents_dim":24,"patch_size":[1,2,2],
    "text_dim":5120,"timestep_input_dim":256,"time_embed_hidden_size":5376,"time_embed_dim":2688,
    "adaln_out_features":96768,"final_adaln_out_features":10752,"rope_inv_freq_len":16,
    "norm_eps":0.00001,"qk_norm_eps":0.00001,"final_norm_eps":0.00001})").parse();
}
block::ImageConfig image() {
    return block::ImageConfig::parse(config(),fixture::JsonParser(R"({"num_inference_steps":6,
    "sigmas":[1,0.9675236940383911,0.8875485062599182,0.75,0.5327739119529724,0.23200887441635132,0]})").parse());
}
}
int main(int argc,char** argv) try {
    const auto c=image(); require(c.sigmas.size()==7); require(c.sigmas[3]==0.75F);
    rejects([&]{ block::ImageConfig::parse(config(),fixture::JsonParser(R"({"num_inference_steps":6,"sigmas":[1,0.9,0.9,0.7,0.5,0.2,0]})").parse()); });
    rejects([&]{ block::ImageConfig::parse(config(),fixture::JsonParser(R"({"num_inference_steps":5,"sigmas":[1,0]})").parse()); });
    rejects([&]{ block::ImageConfig::parse(fixture::JsonParser("{}").parse(),fixture::JsonParser("{}").parse()); });
    const auto policy=block::ImageConfig::weights_policy(); require(!policy.audio_enabled && !policy.round_params_bf16 && policy.dense_bf16);
    block::LoadPolicy video; require(video.audio_enabled && video.round_params_bf16 && !video.direct_timestep && !video.dense_bf16);
    const std::vector<block::RefBlock> refs={{block::RefBlock::image,1,4,8,0},{block::RefBlock::image,1,2,2,0}};
    const auto L=block::ImageConfig::layout(2,64,32,refs);
    require(L.seq_len==13 && L.audio_pos.empty() && L.img_update.size()==11);
    require(L.img_update[0]==0 && L.img_update[8]==0 && L.img_update[9]==1 && L.img_update[10]==1);
    require(L.position_ids[2*3]==2 && L.position_ids[10*3]==3 && L.position_ids[11*3]==4);
    require(L.find(block::SegKind::video).start==11 && L.find(block::SegKind::video).stop==13);
    // Text, reference and target share the existing three-axis grid; image has a singleton T.
    const auto old=block::PackedLayout::build(2,1,2,4,0,{},refs);
    require(L.position_ids==old.position_ids && L.img_pos==old.img_pos);
    for(auto sigma:c.sigmas) if(sigma>0) {
        const auto p=block::ForwardPlan::build(L,sigma,{},block::ImageConfig::plan_policy());
        require(p.sigma_v==p.sigma_a && p.t_v==p.t_a);
        const auto mr=block::mod_row_of(p.mod_segments,L.seq_len);
        require(mr[0]%3==1 && mr.back()%3==0 && mr[2]%3==0);
    }
    rejects([]{ block::ImageConfig::layout(1,65,64); });
    rejects([]{ block::ImageConfig::layout(1,0,64); });
    rejects([]{ block::ImageConfig::layout(1,64,64,std::vector<block::RefBlock>(17)); });
    rejects([]{ block::ImageConfig::layout(1,64,64,{{block::RefBlock::audio,0,0,0,5}}); });
    require(block::ImageConfig::layout(1,4096,4096).seq_len==16385);
    require(block::ImageConfig::layout(1,6272,2688).seq_len==16465);
    block::GlobalWeights g; g.audio_enabled=false; g.pdd_fused=true; g.pdd_steps=6;
    g.video_out.n=6*96; g.video_out.k=2; g.video_out.w.resize(6*96*2); g.video_out.b.resize(6*96);
    for(int i=0;i<6;++i) {
        std::fill_n(g.video_out.w.data()+i*96*2,96*2,float(i+1)); std::fill_n(g.video_out.b.data()+i*96,96,float(10+i));
    }
    for(std::size_t i=0;i<6;++i) {
        block::ForwardPlan p; p.sigma_v=c.sigmas[i]; block::PddContext pc{c.sigmas,c.sigmas[i],1,1};
        block::F32Linear v,a;
        require(block::select_heads(g,p,&pc,v,a)); require(v.n==96 && v.k==2 && a.w.empty());
        require(v.w.front()==float(i+1) && v.w.back()==float(i+1) && v.b.front()==float(10+i));
    }
    int calls=0;
    const auto sampled=block::sample_image(c,std::vector<float>(24,0),[&](const block::AvState& x,float s) {
        require(x.audio.empty() && s==c.sigmas[calls++]); return block::AvState{std::vector<float>(24,1),{}};
    });
    require(calls==6); require(std::abs(sampled.video[0]+1)<1e-6);
    auto z=vae::ImageVae::posterior(std::vector<float>(48,0),std::vector<float>(24,2),std::vector<float>(24,0),std::vector<float>(24,2));
    require(z==std::vector<float>(24,1));
    rejects([]{ vae::ImageVae::posterior({}, {}, {}, {}); });
    const auto rope=vae::image_rope_table(2,3);
    require(rope.L==11 && rope.c.size()==11*24);
    require(std::abs(rope.c[0]-std::cos(float(2*3.14159265358979323846)*float(-6.0/7)))<1e-6);
    require(rope.c[6*24]==1 && rope.s[6*24]==0);
    for(bool fp32 : {false,true}) {
        const auto r2=vae::image_rope_table(2,3,fp32,2),r3=vae::image_rope_table(2,3,fp32,3);
        require(r2.L==11 && r2.pairs==16 && r2.c.size()==11*16 && r2.s.size()==r2.c.size());
        for(int row=0;row<11;++row) for(int pair=0;pair<16;++pair) {
            require(r2.c[row*16+pair]==r3.c[row*24+8+pair]);
            require(r2.s[row*16+pair]==r3.s[row*24+8+pair]);
        }
    }
    rejects([] { (void)vae::image_rope_table(2,3,false,1); });
    std::vector<float> rows(768); for(int i=0;i<768;++i) rows[i]=float(i);
    const auto pixels=vae::precision::unpatchify<float>(rows,1,1,1,1); require(rows==pixels);
    // Small dense projection and scalar-policy arithmetic without device work.
    block::QLinear q; q.n=1; q.k=2; q.bf16={block::tob(2),block::tob(3)};
    std::uint16_t x[]={block::tob(4),block::tob(5)},y=0; block::HostLinear linear;
    linear.run(q,x,1,&y); require(block::bf(y)==23);
    vae::LinearT<float> f; f.in=2; f.out=1; f.w={2,3}; f.b={1};
    require(vae::precision::linear<float>({4,5},1,f)[0]==24);
    int rh=0,rw=0;
    te::smart_resize(32,32,rh,rw); require(rh==64 && rw==64);
    te::smart_resize(32,32,rh,rw,65536,16777216);require(rh==256 && rw==256);
    te::smart_resize(4096,4096,rh,rw,65536,16777216);require(rh==4096 && rw==4096);
    if(argc>1) {
        const auto fixture=block::read_json_file(argv[1]);
        for(const auto& test:fixture.arr()) {
            std::vector<block::RefBlock> rr;
            for(const auto& r:test.at("refs").arr()) rr.push_back({block::RefBlock::image,1,std::uint32_t(r.arr()[0].num()),std::uint32_t(r.arr()[1].num()),0});
            const auto layout=block::ImageConfig::layout(std::uint32_t(test.at("text").num()),std::uint32_t(test.at("width").num()),std::uint32_t(test.at("height").num()),rr);
            const auto& expected=test.at("positions").arr(); require(expected.size()==layout.position_ids.size());
            for(std::size_t i=0;i<expected.size();++i) require(std::abs(expected[i].num()-layout.position_ids[i])<1e-12);
        }
    }
    if(argc>2) {
        const auto actual=block::ImageConfig::load(argv[2]); require(actual.sigmas==c.sigmas);
        block::SafeTensors heads(std::string(argv[2])+"/pdd_fused_heads.safetensors");
        require(heads.at("video_out.weight").dtype==block::StDtype::f32);
        block::load_image_pdd(g,heads,actual);
        require(g.video_out.w.size()==std::size_t(6)*96*5376 && g.video_out.b.size()==6*96);
        require(std::all_of(g.video_out.w.begin(),g.video_out.w.end(),[](float v){return std::isfinite(v);}));
    }
    std::cout<<"image host tests passed: "<<checks<<" assertions\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
