#include "../apps/qwen_deploy_provider.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace dp = strix::deploy;
static void check(bool b, const char* m) { if (!b) throw std::runtime_error(m); }
static dp::DeviceFacts device(unsigned gib) {
    dp::DeviceFacts d; d.gfx_target="gfx1201"; d.cu_count=64;
    d.mem_total_bytes=d.mem_free_bytes=std::uint64_t{gib}<<30; d.source="fixture"; return d;
}
int main() {
    try {
        dp::Footprint f; f.weights=20ULL<<30; f.kv_total=2ULL<<30; f.workspace=1ULL<<30;
        strix::qwen::deployment::Provider single(f,8192,1,"single","default");
        dp::Spec s; s.context=8192; s.slots=1;
        auto p=dp::resolve(s,{device(32)},single);
        check(!p.refused && p.ranks[0].required_bytes==23ULL<<30,"single owner inventory exactly once");
        check(dp::resolve(s,{device(24)},single).refused,"free bytes/margin arithmetic");
        strix::qwen::deployment::Provider replicas(f,8192,1,"replica","default");
        s.nodes=2;
        auto r=dp::resolve(s,{device(32),device(64)},replicas);
        check(!r.refused && r.ranks.size()==2,"two independent replicas");
        check(r.ranks[0].required_bytes==r.ranks[1].required_bytes,"full replica on each node");
        s.nodes=1;
        check(dp::resolve(s,{device(64)},replicas).refused,"replica binding requires two nodes");
        s.nodes=2;
        strix::qwen::deployment::Provider pipeline(f,8192,1,"pipeline","default");
        check(dp::resolve(s,{device(32),device(64)},pipeline).refused,"pipeline requires link");
        s.link="usb4auto";
        auto pp=dp::resolve(s,{device(32),device(64)},pipeline);
        check(!pp.refused && pp.ranks[0].footprint.weights==f.weights,"pipeline full model continuation");
        check(pp.ranks[0].footprint.staging==32ULL<<20,"pipeline handoff staging");
        s.context=16384;
        check(dp::resolve(s,{device(64),device(64)},pipeline).refused,"context cannot drift from compiler owners");
        s.context=8192; s.slots=2;
        check(dp::resolve(s,{device(64),device(64)},pipeline).refused,"unsupported slots pin refuses");
        s.slots=1; s.profile="quality";
        check(dp::resolve(s,{device(64),device(64)},pipeline).refused,"profile pin refuses");
        s.profile="auto"; s.format.decode="int4";
        check(dp::resolve(s,{device(64),device(64)},pipeline).refused,"cannot claim another GGUF encoding");
        s.format.decode="auto";
        unsigned calls=0;
        const auto echo_peer=[&](std::span<const std::byte> out,std::span<std::byte> in) {
            std::memcpy(in.data(),out.data(),out.size());
            if (!calls++) { std::uint64_t rank=1; std::memcpy(in.data()+8,&rank,8); }
        };
        const auto agreed=dp::resolve_pair(s,pipeline,0,device(32),echo_peer);
        check(!agreed.refused && calls==2,"live facts and plan agreement");
        calls=0; bool mismatch=false;
        try {
            (void)dp::resolve_pair(s,pipeline,0,device(32),[&](auto out,auto in) {
                echo_peer(out,in); if(calls==2) in[0]^=std::byte{1};
            });
        } catch(const std::runtime_error&) { mismatch=true; }
        check(mismatch,"peer plan mismatch refuses");
        check(dp::inspection_devices(dp::Spec{},0,device(32)).size()==1,"no fabricated remote memory");
        std::cout<<"Qwen deployment host checks passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
