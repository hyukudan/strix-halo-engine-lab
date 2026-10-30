#include "strix/hip_mtp_draft_head.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string_view>
namespace strix {
namespace {
void require(bool v,const char* m) { if(!v) throw std::invalid_argument(m); }
}
namespace {
std::vector<std::uint32_t> read_one_token_file(const std::string& path,std::uint32_t vocabulary) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    require(in.good(),"MTP draft vocabulary file cannot be opened");
    const auto bytes=static_cast<std::uint64_t>(in.tellg());
    require(bytes && bytes%4U==0U && bytes/4U<=vocabulary,"MTP draft vocabulary file must hold 1..vocab uint32 ids");
    std::vector<std::uint32_t> ids(bytes/4U);in.seekg(0);
    std::vector<unsigned char> raw(bytes);in.read(reinterpret_cast<char*>(raw.data()),static_cast<std::streamsize>(bytes));
    require(in.good(),"MTP draft vocabulary file read failed");
    for(std::size_t i=0;i<ids.size();++i) {
        ids[i]=0;for(unsigned b=0;b<4U;++b) ids[i]|=static_cast<std::uint32_t>(raw[4*i+b])<<(8U*b);
        require(ids[i]<vocabulary,"MTP draft vocabulary id exceeds the model vocabulary");
    }
    return ids;
}
} // namespace
// A comma-separated list (a core subset plus language packs) is their union in file order, first occurrence kept.
std::string HipMtpDraftHead::resolve_vocabulary_spec(const std::string& spec,std::span<const std::uint32_t> prompt,
    std::uint32_t vocabulary,std::string& chosen) {
    chosen=spec;
    if(spec.rfind("auto:",0)!=0) return spec;
    const auto dir=spec.substr(5);
    require(!dir.empty(),"MTP draft vocabulary auto: needs a directory");
    // STRIX_MTP_DRAFT_VOCAB=2 (vocabulary policy 2): a language pack comes from pack2-<lang>.bin (ranked against
    // core32k, so no row repeats the core; zh holds 32k rows), and a prompt without a pack gets core64k (the rows a
    // pack would have used go to English/code). The language is still chosen with the pack-<lang>16k.bin files.
    // STRIX_MTP_DRAFT_CORE=48k|64k|... forces the core (core<N>.bin, a prefix of the same ranking) in both cases.
    static const bool policy2=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_VOCAB");return v && std::string_view(v)=="2";}();
    static const std::string forced=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_CORE");return std::string(v && *v?v:"");}();
    require(forced.find('/')==std::string::npos,"STRIX_MTP_DRAFT_CORE names a core size, not a path");
    const auto core_file=[&](const std::string& size) {
        const auto path=dir+"/core"+size+".bin";
        require(std::ifstream(path,std::ios::binary).good(),"MTP draft vocabulary core file is missing from the auto: directory");
        return path;
    };
    static constexpr const char* languages[]={"de","es","fr","it","pt","ru","ja","zh"};
    std::string best;std::uint64_t best_count=0;
    for(const auto* language:languages) {
        const auto path=dir+"/pack-"+language+"16k.bin";
        if(!std::ifstream(path,std::ios::binary)) continue;
        std::vector<std::uint8_t> member(vocabulary,0U);
        for(const auto id:read_one_token_file(path,vocabulary)) member[id]=1U;
        std::uint64_t count=0;
        for(const auto token:prompt) if(token<vocabulary&&member[token]) ++count;
        if(count>best_count) {best_count=count;best=language;}
    }
    if(best.empty()||best_count<2U||best_count*20U<prompt.size()) {
        const auto size=!forced.empty()?forced:policy2?std::string("64k"):std::string("32k");
        chosen="core"+size;return core_file(size);
    }
    const auto size=forced.empty()?std::string("32k"):forced;
    const auto core=core_file(size);
    const auto pack2=dir+"/pack2-"+best+".bin";
    if(policy2 && std::ifstream(pack2,std::ios::binary)) {
        chosen="core"+size+"+"+best+" pack2 ("+std::to_string(best_count)+" prompt tokens)";
        return core+","+pack2;
    }
    chosen="core"+size+"+"+best+"16k ("+std::to_string(best_count)+" prompt tokens)";
    return core+","+dir+"/pack-"+best+"16k.bin";
}
std::vector<std::uint32_t> HipMtpDraftHead::read_token_file(const std::string& paths,std::uint32_t vocabulary) {
    std::vector<std::uint32_t> ids;std::vector<std::uint8_t> seen(vocabulary,0U);
    for(std::size_t begin=0;begin<=paths.size();) {
        const auto end=std::min(paths.find(',',begin),paths.size());
        if(end>begin)
            for(const auto id:read_one_token_file(paths.substr(begin,end-begin),vocabulary))
                if(!seen[id]) {seen[id]=1U;ids.push_back(id);}
        begin=end+1U;
    }
    require(!ids.empty(),"MTP draft vocabulary list names no ids");
    return ids;
}
} // namespace strix
