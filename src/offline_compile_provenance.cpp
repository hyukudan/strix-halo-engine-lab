#include "strix/offline_compile_provenance.hpp"
#include "strix/model_content_hash.hpp"
#include <array>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>
#include <utility>
namespace strix { namespace {
[[noreturn]] void bad() { throw std::invalid_argument("offline compile provenance is malformed, stale or incomplete"); }
bool sha(std::string_view s) {
    if(s.size()!=64) return false;
    for(char c:s) if(!((c>='0'&&c<='9')||(c>='a'&&c<='f'))) return false;
    return true;
}
struct Reader {
    std::span<const std::byte> b; std::size_t at=0;
    std::uint64_t number(unsigned n) {
        if(n>b.size()-at) bad();
        std::uint64_t v=0;
        for(unsigned i=0;i<n;++i) v|=std::uint64_t(std::to_integer<unsigned char>(b[at++]))<<(8U*i);
        return v;
    }
    std::uint32_t count(std::uint32_t maximum) {
        const auto n=number(4);
        if(!n||n>maximum) bad();
        return static_cast<std::uint32_t>(n);
    }
    std::string text(std::size_t maximum=16384) {
        const auto n=number(4); if(!n||n>maximum||n>b.size()-at) bad();
        std::string s(reinterpret_cast<const char*>(b.data()+at),static_cast<std::size_t>(n));at+=static_cast<std::size_t>(n);
        if(s.find('\0')!=std::string::npos) bad();
        return s;
    }
};
struct File { std::uint32_t role=0;std::uint64_t bytes=0;std::string digest; };
struct Target {std::vector<std::string> dependencies;};
} // namespace
OfflineCompileProvenance load_offline_compile_provenance(const std::filesystem::path& path,
    std::string_view executable, std::string_view context, OfflineCompileProvenanceLimits limits) {
    if(!sha(executable)||context.empty()||context.size()>65536) bad();
    const auto size=std::filesystem::file_size(path);
    if(!size||size>limits.max_manifest_bytes||size>std::numeric_limits<std::size_t>::max()||
        size>static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) bad();
    std::ifstream in(path,std::ios::binary);if(!in)bad();
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(size));
    if(in.gcount()!=static_cast<std::streamsize>(size)||in.bad())bad();
    if(in.peek()!=std::char_traits<char>::eof()||in.bad())bad();
    Reader r{data};
    for(char c:std::string_view("STXTC001"))if(r.number(1)!=static_cast<unsigned char>(c))bad();
    if(r.text(65536)!=context||r.text(64)!=executable||r.number(4)!=1)bad(); // Ninja POSIX/FileAPIv2/deps export v1
    const auto file_count=r.count(limits.max_files);
    std::map<std::string,File> files;std::array<std::uint32_t,11> roles{};
    std::uint64_t referenced=0;
    for(std::uint32_t i=0;i<file_count;++i){
        File f;f.role=static_cast<std::uint32_t>(r.number(4));
        if(f.role<1||f.role>10)bad();
        auto p=r.text();f.bytes=r.number(8);f.digest=r.text(64);
        if(!sha(f.digest)||!std::filesystem::path(p).is_absolute()||
            f.bytes>limits.max_single_file_bytes||f.bytes>limits.max_referenced_bytes-referenced||!files.emplace(p,f).second)bad();
        referenced+=f.bytes;++roles[f.role];
        if(f.role==10&&f.digest!=executable)bad();
    }
    if(!roles[1]||!roles[2]||!roles[3]||!roles[4]||!roles[6]||roles[7]!=1||roles[8]!=1||roles[9]!=1||roles[10]!=1)bad();
    const auto use=[&](const std::string& p,std::uint32_t role=0){
        const auto it=files.find(p);if(it==files.end()||(role&&it->second.role!=role))bad();
    };
    const auto root_target=r.text();
    if(root_target!="strix-hip-decode-token")bad();
    const auto target_count=r.count(limits.max_targets);
    std::map<std::string,Target> targets;
    for(std::uint32_t i=0;i<target_count;++i){
        auto name=r.text();auto type=r.text();
        if(type!="EXECUTABLE"&&type!="STATIC_LIBRARY"&&type!="OBJECT_LIBRARY"&&type!="SHARED_LIBRARY")bad();
        if(type!="OBJECT_LIBRARY"){
            const auto artifact=r.text();use(artifact);
            if(name==root_target&&(type!="EXECUTABLE"||files.at(artifact).role!=10))bad();
        }
        Target t;const auto deps=r.number(4);if(deps>limits.max_targets)bad();
        for(std::uint64_t j=0;j<deps;++j)t.dependencies.push_back(r.text());
        const auto fragments=r.number(4);if(fragments>65536)bad();
        for(std::uint64_t j=0;j<fragments;++j)static_cast<void>(r.text(65536));
        if(!targets.emplace(std::move(name),std::move(t)).second)bad();
    }
    for(const auto& [name,t]:targets){static_cast<void>(name);for(const auto& dep:t.dependencies)if(!targets.contains(dep))bad();}
    std::set<std::string> reachable,active;
    const std::function<void(const std::string&)> visit=[&](const std::string& name){
        if(active.contains(name)||!targets.contains(name))bad();
        if(reachable.contains(name))return;
        active.insert(name);
        for(const auto& dep:targets.at(name).dependencies)visit(dep);
        active.erase(name);reachable.insert(name);
    };
    visit(root_target);if(reachable.size()!=targets.size())bad();
    const auto unit_count=r.count(limits.max_units);std::set<std::string> objects;
    for(std::uint32_t i=0;i<unit_count;++i){
        const auto target=r.text();if(!targets.contains(target))bad();
        const auto source=r.text();use(source,2);
        const auto object=r.text();use(object,4);if(!objects.insert(object).second)bad();
        if(!std::filesystem::path(r.text()).is_absolute())bad(); // exact working directory
        const auto compiler=r.text();use(compiler,1);
        const auto argc=r.count(16384);if(argc<2)bad();
        const auto compiler_spelling=r.text(65536);
        if(!std::filesystem::path(compiler_spelling).is_absolute()||
            std::filesystem::canonical(compiler_spelling)!=std::filesystem::path(compiler))bad();
        for(std::uint32_t j=1;j<argc;++j)static_cast<void>(r.text(65536));
        const auto deps=r.count(limits.max_files);bool has_source=false;
        for(std::uint32_t j=0;j<deps;++j){const auto dep=r.text();use(dep);has_source=has_source||dep==source;}
        if(!has_source)bad();
    }
    if(r.at!=data.size()||objects.size()!=roles[4])bad();
    // Setup-only reads. Verify every referenced source/header/object/toolchain
    // and metadata file against the SAME parsed manifest; never execute commands.
    // These are declared build inputs, not transitive runtime/driver closure.
    for(const auto& [p,f]:files){
        if(!std::filesystem::is_regular_file(p)||std::filesystem::file_size(p)!=f.bytes)bad();
        // The running process image was hashed once by the caller. Every other
        // read consumes EXACTLY its bounded declared extent, then demands EOF;
        // special/growing inputs cannot turn a helper into an unbounded read.
        if(f.role!=10){
            if(f.bytes>std::numeric_limits<std::size_t>::max()||
                f.bytes>static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))bad();
            std::ifstream file(p,std::ios::binary);if(!file)bad();
            std::vector<std::byte> bytes(static_cast<std::size_t>(f.bytes));
            if(!bytes.empty())file.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
            if((!bytes.empty()&&file.gcount()!=static_cast<std::streamsize>(bytes.size()))||file.bad())bad();
            if(file.peek()!=std::char_traits<char>::eof()||file.bad())bad();
            if(model_content_identity_hex(sha256_bytes(bytes))!=f.digest)bad();
        }
    }
    OfflineCompileProvenance result;
    result.payload_sha256=model_content_identity_hex(sha256_bytes(data));
    result.compiler_flags_sha256=result.payload_sha256;
    result.payload_bytes=size;result.referenced_bytes=referenced;
    result.files_checked=file_count;result.units=unit_count;result.targets=target_count;
    return result;
}
} // namespace strix
