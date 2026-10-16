#pragma once
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::transport::visibility_lab {
struct RunOptions {
    std::string campaign,run,device,local,peer,gpu_uuid,kernel,build,link;
    std::uint64_t epoch=0,request=0,bytes=0,seed=0;
    unsigned iterations=1,ring_frames=32;
    int gpu_device=0,runtime_version=0,driver_version=0;
    std::uint16_t port=0;
    bool receiver=false,listener=false,duplex=false,check_only=false;
    std::chrono::milliseconds timeout{10000},producer_delay{0},consumer_delay{0};
};
inline std::uint64_t bounded(std::string_view value,std::uint64_t low,std::uint64_t high) {
    std::uint64_t result=0;auto parsed=std::from_chars(value.data(),value.data()+value.size(),result);
    if(parsed.ec!=std::errc{}||parsed.ptr!=value.data()+value.size()||result<low||result>high)
        throw std::invalid_argument("invalid bounded integer");
    return result;
}
inline bool hex_id(std::string_view s,std::size_t low,std::size_t high) {
    return s.size()>=low&&s.size()<=high&&s.size()%2==0&&
        std::all_of(s.begin(),s.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
inline RunOptions parse(int argc,char** argv) {
    RunOptions o;std::map<std::string,std::string> values;bool opt=false;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--experimental-commissioning") {if(opt)throw std::invalid_argument("duplicate opt-in");opt=true;continue;}
        if(key=="--duplex-pairs") {if(o.duplex)throw std::invalid_argument("duplicate duplex");o.duplex=true;continue;}
        if(key=="--check-config") {if(o.check_only)throw std::invalid_argument("duplicate config check");o.check_only=true;continue;}
        if(i+1>=argc||!values.emplace(key,argv[++i]).second)throw std::invalid_argument("missing/duplicate option");
    }
    const auto take=[&](const char* key,const char* fallback=nullptr) {
        auto it=values.find(key);if(it==values.end()){if(fallback)return std::string(fallback);throw std::invalid_argument(std::string("required option: ")+key);}
        auto v=it->second;values.erase(it);
        if(v.empty()||v.size()>512||std::any_of(v.begin(),v.end(),[](unsigned char c){return c<32||c>126;}))throw std::invalid_argument("invalid option text");
        return v;
    };
    o.campaign=take("--campaign-id");o.run=take("--run-id");o.device=take("--dma-device");
    o.local=take("--local-endpoint");o.peer=take("--peer-endpoint");o.gpu_uuid=take("--gpu-uuid");
    o.kernel=take("--kernel-release");o.build=take("--driver-build-id");o.link=take("--physical-link");
    auto role=take("--role"),mode=take("--control-mode");
    if((role!="sender"&&role!="receiver")||(mode!="listen"&&mode!="connect"))throw std::invalid_argument("invalid role/control-mode");
    o.receiver=role=="receiver";o.listener=mode=="listen";
    o.epoch=bounded(take("--epoch"),1,UINT64_MAX);o.request=bounded(take("--request-id"),1,UINT64_MAX);
    o.bytes=bounded(take("--payload-bytes"),1,4095ULL*4096);o.seed=bounded(take("--seed","0"),0,UINT64_MAX);
    o.ring_frames=static_cast<unsigned>(bounded(take("--ring-frames","32"),32,4096));
    o.iterations=static_cast<unsigned>(bounded(take("--iterations","1"),1,128));
    o.port=static_cast<std::uint16_t>(bounded(take("--control-port"),1,65535));
    o.gpu_device=static_cast<int>(bounded(take("--gpu-device","0"),0,15));
    o.runtime_version=static_cast<int>(bounded(take("--gpu-runtime-version"),1,INT32_MAX));
    o.driver_version=static_cast<int>(bounded(take("--gpu-driver-version"),1,INT32_MAX));
    o.timeout=std::chrono::milliseconds{static_cast<std::int64_t>(bounded(take("--timeout-ms","10000"),1,60000))};
    o.producer_delay=std::chrono::milliseconds{static_cast<std::int64_t>(bounded(take("--producer-delay-ms","0"),0,1000))};
    o.consumer_delay=std::chrono::milliseconds{static_cast<std::int64_t>(bounded(take("--consumer-delay-ms","0"),0,1000))};
    const auto id=[](const std::string& s){return !s.empty()&&s.size()<=116&&std::all_of(s.begin(),s.end(),[](unsigned char c){return(c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.';});};
    if(!values.empty()||!opt||!id(o.campaign)||!id(o.run)||o.device.front()!='/'||
        !hex_id(o.local,32,32)||!hex_id(o.peer,32,32)||!hex_id(o.gpu_uuid,32,32)||!hex_id(o.build,2,128)||
        o.local==o.peer||o.local==std::string(32,'0')||o.peer==std::string(32,'0')||o.gpu_uuid==std::string(32,'0')||
        o.bytes>static_cast<std::uint64_t>(o.ring_frames-1)*4096||o.bytes*o.iterations>128ULL*1024*1024||
        (o.duplex&&(o.iterations<2||o.iterations%2))||o.timeout<=o.producer_delay+o.consumer_delay)
        throw std::invalid_argument("invalid identity, geometry or commissioning budget");
    return o;
}
inline std::string quote(std::string_view s) {
    std::string out="\"";constexpr char hex[]="0123456789abcdef";
    for(char character:s){const auto c=static_cast<unsigned char>(character);if(c=='"'||c=='\\'){out+='\\';out+=static_cast<char>(c);}else if(c<32){out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}else out+=static_cast<char>(c);}
    return out+'"';
}
} // namespace strix::transport::visibility_lab
