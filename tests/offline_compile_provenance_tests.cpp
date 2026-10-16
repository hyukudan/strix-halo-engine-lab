#include "strix/offline_compile_provenance.hpp"
#include "strix/model_content_hash.hpp"
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool v){if(!v)throw std::runtime_error("compile provenance fixture failed");}
struct Fixture {
    std::filesystem::path dir;
    struct Input {unsigned role;std::filesystem::path path;std::string content;};
    std::vector<Input> inputs;
    Fixture(){
        dir=std::filesystem::temp_directory_path()/("strix-provenance-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(dir));
        for(unsigned role:{1U,2U,3U,4U,6U,7U,8U,9U,10U}){
            Input f{role,dir/("file"+std::to_string(role)),"synthetic-role-"+std::to_string(role)};
            std::ofstream out(f.path,std::ios::binary);out<<f.content;out.close();require(bool(out));inputs.push_back(std::move(f));
        }
    }
    ~Fixture(){std::error_code e;std::filesystem::remove_all(dir,e);}
    const Input& file(unsigned role)const{for(const auto& f:inputs)if(f.role==role)return f;throw std::logic_error("role absent");}
    static std::string hash(const Input& f){return strix::model_content_identity_hex(strix::sha256_file_content(f.path));}
    std::vector<std::byte> encode(unsigned mutation=0)const{
        std::vector<std::byte>b;
        const auto n=[&](std::uint64_t v,unsigned bytes){for(unsigned i=0;i<bytes;++i)b.push_back(static_cast<std::byte>((v>>(8U*i))&255U));};
        const auto text=[&](const std::string& t){n(t.size(),4);for(unsigned char c:t)b.push_back(static_cast<std::byte>(c));};
        for(unsigned char c:std::string("STXTC001"))b.push_back(static_cast<std::byte>(c));
        text("aabb");text(hash(file(10)));n(1,4);n(inputs.size()+(mutation==1?1U:0U),4);
        for(const auto& f:inputs){n(f.role,4);text(f.path.string());n(f.content.size(),8);text(hash(f));}
        if(mutation==1){const auto& f=file(1);n(f.role,4);text(f.path.string());n(f.content.size(),8);text(hash(f));}
        text("strix-hip-decode-token");n(1,4);text("strix-hip-decode-token");text("EXECUTABLE");text(file(10).path.string());n(mutation==2?1U:0U,4);
        if(mutation==2)text("unclosed-target");
        n(1,4);text("flags:-strict");
        n(1,4);text("strix-hip-decode-token");text(file(2).path.string());text(file(4).path.string());text(dir.string());text(file(1).path.string());
        n(2,4);text(file(1).path.string());text("-c");n(2,4);
        text(mutation==3?(dir/"untracked-header").string():file(2).path.string());text(file(3).path.string());
        if(mutation==4)b.push_back(std::byte{0});
        if(mutation==5)b.resize(b.size()-3U);
        return b;
    }
    void save(const std::vector<std::byte>& b)const{std::ofstream out(dir/"manifest",std::ios::binary);out.write(reinterpret_cast<const char*>(b.data()),static_cast<std::streamsize>(b.size()));out.close();require(bool(out));}
};
template<class F>void reject(F&& f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected);}
}
int main()try{
    Fixture f;auto b=f.encode();f.save(b);
    const auto binary=Fixture::hash(f.file(10));
    const auto result=strix::load_offline_compile_provenance(f.dir/"manifest",binary,"aabb");
    require(result.units==1&&result.targets==1&&result.files_checked==9);
    require(result.compiler_flags_sha256==strix::model_content_identity_hex(strix::sha256_bytes(b)));
    reject([&]{static_cast<void>(strix::load_offline_compile_provenance(f.dir/"manifest",std::string(64,'0'),"aabb"));});
    reject([&]{static_cast<void>(strix::load_offline_compile_provenance(f.dir/"manifest",binary,"stale-context"));});
    {std::ofstream out(f.file(3).path);out<<"changed-header";}
    reject([&]{static_cast<void>(strix::load_offline_compile_provenance(f.dir/"manifest",binary,"aabb"));});
    {std::ofstream out(f.file(3).path);out<<f.file(3).content;}
    for(unsigned mutation=1;mutation<=5;++mutation){f.save(f.encode(mutation));reject([&]{static_cast<void>(strix::load_offline_compile_provenance(f.dir/"manifest",binary,"aabb"));});}
    f.save(b);strix::OfflineCompileProvenanceLimits tiny;tiny.max_manifest_bytes=1;
    reject([&]{static_cast<void>(strix::load_offline_compile_provenance(f.dir/"manifest",binary,"aabb",tiny));});
    tiny={};tiny.max_referenced_bytes=1;
    reject([&]{static_cast<void>(strix::load_offline_compile_provenance(f.dir/"manifest",binary,"aabb",tiny));});
    std::cout<<"offline compile provenance source fixtures completed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
