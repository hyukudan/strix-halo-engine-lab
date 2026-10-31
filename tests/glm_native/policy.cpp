#include "glm/rtxpf/producer/policy.hpp"
#include "glm/rtxpf/producer/schedule.hpp"
#include "glm/rtxpf/producer/state_slices.hpp"
#include <iostream>
using namespace strix::glm::rtxpf::producer;
void check(bool b){if(!b)throw std::runtime_error("policy check failed");}
int main(){
    for(auto invalid:{"", "0", "0,1", "GPU-123", "MIG-123", "GPU-00000000-0000-0000-0000-000000000000,GPU-11111111-1111-1111-1111-111111111111"}){
        bool rejected=false;try{require_pro_uuid(invalid);}catch(...){rejected=true;}check(rejected);
    }
    check(!valid_schedule(131076,256)&&!valid_schedule(131072,33)&&!valid_schedule(131072,65536));
    check(effective_row_tile(8192,0)==256&&effective_row_tile(32768,0)==2048&&effective_row_tile(131072,256)==256);
    for(unsigned tile:{32u,256u,1024u,2048u})for(unsigned capacity:{8192u,32768u,131072u}){
        check(valid_schedule(capacity,tile));
        // Exercise partial pools, exact/tail graph buckets, B32 carry boundaries
        // and the maximum prefix, without allocating a prefix-sized test buffer.
        for(unsigned prefix:{1u,3u,31u,32u,33u,tile-1,tile,tile+1,capacity-1}){
            unsigned covered=0,pools=0;
            for(unsigned first=0;first<prefix;first+=tile){auto c=row_tile(first,prefix,tile);check(c.first==covered&&c.first%32==0&&c.rows>0&&c.rows<=tile&&c.first+c.rows<=prefix);check(c.bucket==(c.rows==tile?0u:1u));pools+=(c.first+c.rows)/4-c.first/4;covered+=c.rows;}
            check(covered==prefix&&pools==prefix/4);
        }
        // A larger capacity adds only residual/hidden rows and token IDs;
        // changing tile size affects scratch without multiplying by capacity.
        if(capacity<131072)check(activation_bytes(capacity*2,tile)-activation_bytes(capacity,tile)==std::uint64_t(capacity)*(4096*4*5+4));
        check(activation_bytes(capacity,tile)>std::uint64_t(capacity)*4096*4*5);
    }
    check(activation_bytes(131072,256)<11ULL*1024*1024*1024);
    require_pro_uuid("GPU-00000000-0000-0000-0000-000000000000");
    check(targets(8192).device==20&&targets(32768).device==19&&targets(131072).device==14);
    check(layer_limit("0")==0&&layer_limit("43")==43);
    check(!caps_require_file_tier(32768,43,43)&&caps_require_file_tier(32768,43,0));
    check(caps_require_file_tier(32768,19,22)&&!caps_require_file_tier(32768,19,24));
    for(auto bad:{"-1","44","1x","9999999999999999",""}){bool rejected=false;try{layer_limit(bad);}catch(...){rejected=true;}check(rejected);}
    check(!file_tier_covered(0,0,0,false)&&!file_tier_covered(2,0,3,true));
    check(!file_tier_covered(1,1024,1,true)&&!file_tier_covered(2,1024,1,true));
    check(file_tier_covered(2,1024,3,true)&&file_tier_covered(1,1024,1,false));
    check(!admit(1,0,0)&&!admit(1,10,10)&&admit(10,20,10));
    check(host_budget(60ULL<<30,50ULL<<30)==42ULL<<30);
    check(host_budget(60ULL<<30,7ULL<<30)==0);
    check(host_budget(0,~0ULL)==0);
    check(host_budget(60ULL<<30,50ULL<<30,16)==34ULL<<30);
    check(host_budget(60ULL<<30,15ULL<<30,16)==0);
    check(host_budget(60ULL<<30,50ULL<<30,0)==50ULL<<30);
    std::vector<std::byte> conv(64*4608),state(64*65536);
    for(unsigned part=0;part<3;++part)for(unsigned head=0;head<64;++head)
        std::fill_n(conv.begin()+(part*64+head)*1536,1536,std::byte(part*64+head));
    for(unsigned head=0;head<64;++head)std::fill_n(state.begin()+head*65536,65536,std::byte(head));
    for(unsigned rank=0;rank<2;++rank){auto c=kda_tp_conv(conv,rank);auto s=kda_tp_state(state,rank);
        for(unsigned part=0;part<3;++part)for(unsigned head=0;head<32;++head){auto b=c.begin()+(part*32+head)*1536;check(std::all_of(b,b+1536,[&](auto v){return v==std::byte(part*64+rank*32+head);}));}
        for(unsigned head=0;head<32;++head)check(s[head*65536]==std::byte(rank*32+head));
    }
    std::cout<<"PASS 128K row schedule/pool coverage, bounded activation accounting, UUID, tier targets, VRAM reserve, physical/commit admission, TP2 state/convolution slices\n";
}
