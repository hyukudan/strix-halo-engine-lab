#include "hip_mtp_qualification_reference.hpp"
#include "strix/model_content_hash.hpp"
#include <bit>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace strix::tests {
namespace {
void require(bool pass,const char* message){if(!pass)throw std::invalid_argument(message);}
constexpr std::uint64_t limit=1ULL<<30U;
struct Writer {
    std::vector<std::byte> data;std::uint64_t maximum;
    void word(std::uint64_t value,unsigned bytes){require(bytes<=maximum-data.size(),"qualification file budget exceeded");
        for(unsigned i=0;i<bytes;++i)data.push_back(static_cast<std::byte>((value>>(8U*i))&255U));}
    template<class T>void atom(const T& value){
        if constexpr(std::is_same_v<T,float>)word(std::bit_cast<std::uint32_t>(value),4);
        else if constexpr(std::is_same_v<T,std::byte>)word(std::to_integer<unsigned>(value),1);
        else if constexpr(std::is_same_v<T,bool>)word(value?1U:0U,1);
        else if constexpr(std::is_enum_v<T>)word(static_cast<std::uint32_t>(value),4);
        else {static_assert(std::is_unsigned_v<T>);word(value,static_cast<unsigned>(sizeof(T)));}
    }
    template<class... T>void operator()(const T&... values){(atom(values),...);}
    void digest(const ModelContentIdentity& value){for(auto b:value.bytes)atom(b);}
    template<class T>void values(const std::vector<T>& v){atom(static_cast<std::uint64_t>(v.size()));for(const auto& x:v)atom(x);}
    void text(const std::string& s){require(s.size()<=4096U,"qualification profile string too long");atom(static_cast<std::uint64_t>(s.size()));for(char c:s)word(static_cast<unsigned char>(c),1);}
    std::uint64_t count(std::uint64_t n,std::uint64_t maximum_count){require(n<=maximum_count,"qualification record count too large");atom(n);return n;}
};
struct Reader {
    std::span<const std::byte> data;std::size_t offset=0;
    std::uint64_t word(unsigned bytes){require(bytes<=data.size()-offset,"truncated qualification file");std::uint64_t value=0;
        for(unsigned i=0;i<bytes;++i)value|=std::uint64_t(std::to_integer<unsigned>(data[offset++]))<<(8U*i);return value;}
    template<class T>void atom(T& value){
        if constexpr(std::is_same_v<T,float>)value=std::bit_cast<float>(static_cast<std::uint32_t>(word(4)));
        else if constexpr(std::is_same_v<T,std::byte>)value=static_cast<std::byte>(word(1));
        else if constexpr(std::is_same_v<T,bool>){const auto n=word(1);require(n<=1U,"invalid qualification boolean");value=n!=0U;}
        else if constexpr(std::is_enum_v<T>)value=static_cast<T>(word(4));
        else {static_assert(std::is_unsigned_v<T>);value=static_cast<T>(word(static_cast<unsigned>(sizeof(T))));}
    }
    template<class... T>void operator()(T&... values){(atom(values),...);}
    void digest(ModelContentIdentity& value){for(auto& b:value.bytes)atom(b);}
    template<class T>void values(std::vector<T>& v){const auto n=word(8);require(n<=(data.size()-offset)/sizeof(T),"qualification vector exceeds remaining bytes");
        v.resize(static_cast<std::size_t>(n));for(auto& x:v)atom(x);}
    void text(std::string& s){const auto n=word(8);require(n<=4096U && n<=data.size()-offset,"invalid qualification profile string");
        s.resize(static_cast<std::size_t>(n));for(auto& c:s)c=static_cast<char>(word(1));}
    std::uint64_t count(std::uint64_t,std::uint64_t maximum_count){const auto n=word(8);require(n<=maximum_count && n<=data.size()-offset,"qualification record count too large");return n;}
};
template<class A,class G>void metadata(A& a,G& g){a.digest(g.independent_source_content);a.digest(g.target_content);a.digest(g.sidecar_content);
    a(g.independent_reference_identity,g.proposal_identity,g.plan_identity,g.semantic_identity,g.native_profile_identity);}
template<class A,class G>void cycle_row(A& a,G& r){
    a(r.initial_source_position,r.final_source_position,r.final_committed_tokens,r.consumed_target_rows,r.reconciled_next_source_position,
        r.initial_pending_token,r.final_pending_token,r.maximum_candidates,r.token_budget,r.proposed_candidates,r.accepted_candidates,
        r.target_r2_dispatches,r.target_projection_launches_saved,r.eos,r.terminal_budget,r.attempted_window,r.accepted_window,r.restored_checkpoint);
    a.values(r.eos_tokens);a.values(r.emitted_tokens);a.values(r.final_target_wide_bf16);a.values(r.reconciled_kv_bf16);a.values(r.reconciled_raw_index_bf16);
}
template<class A,class V,class F>void records(A& a,V& v,std::uint64_t bound,F work){const auto n=a.count(v.size(),bound);
    if constexpr(!std::is_const_v<V>)v.resize(static_cast<std::size_t>(n));for(auto& r:v)work(a,r);}
template<class A,class G>void selected(A& a,G& g){
    switch(g.kind){
    case MtpQualificationKind::cycles:case MtpQualificationKind::history_fallback:
        records(a,g.cycles.cycles,256U,[](auto& ar,auto& row){cycle_row(ar,row);});break;
    case MtpQualificationKind::projection:{auto& p=g.projection;a(p.position);a.text(p.hyper_profile);a.text(p.routed_profile);
        for(auto& h:p.hyper){a(h.role);a.values(h.input);a.values(h.output);a.values(h.injection);a.values(h.workspace);}
        a.values(p.routed.input);a.values(p.routed.output);for(auto& x:p.routed.routes)a(x);for(auto& x:p.routed.route_weights)a(x);a.values(p.routed.workspace);break;}
    case MtpQualificationKind::shared:{auto& s=g.shared;a(s.position);a.text(s.numerical_profile);a.values(s.input);a.values(s.output);a.values(s.workspace);break;}
    case MtpQualificationKind::window:case MtpQualificationKind::checkpoint:{auto& w=g.window;a(w.initial_position);
        records(a,w.chunks,256U,[](auto& ar,auto& chunk){ar.values(chunk.inputs);ar.values(chunk.predictions);
            records(ar,chunk.wide_hidden,8U,[](auto& inner,auto& row){inner.values(row);});});break;}
    case MtpQualificationKind::lm_head:a(g.head.position,g.head.head_profile_identity,g.head.recipe);a.values(g.head.input_bf16);a.values(g.head.full_logits);break;
    default:throw std::invalid_argument("unknown qualification reference kind");
    }
}
void bind(MtpQualificationReference& g){
    require(g.independent_source_content.valid() && g.target_content.valid() && g.sidecar_content.valid() &&
        g.independent_reference_identity && g.proposal_identity && g.plan_identity && g.semantic_identity && g.native_profile_identity,
        "qualification reference requires complete provenance and model/profile pins");
    auto& c=g.cycles;c.independent_source_content=g.independent_source_content;c.target_content=g.target_content;c.sidecar_content=g.sidecar_content;
    c.independent_reference_identity=g.independent_reference_identity;c.structural_plan_identity=g.plan_identity;c.semantic_identity=g.semantic_identity;c.numerical_profile_identity=g.native_profile_identity;
    g.projection.independent_reference_content=g.independent_source_content;g.projection.proposal_identity=g.proposal_identity;
    g.shared.independent_reference_content=g.independent_source_content;g.shared.proposal_identity=g.proposal_identity;
    g.window.independent_reference_content=g.independent_source_content;g.window.target_content=g.target_content;
    auto& h=g.head;h.independent_source_content=g.independent_source_content;h.target_content=g.target_content;h.sidecar_content=g.sidecar_content;
    h.independent_reference_identity=g.independent_reference_identity;h.plan_identity=g.plan_identity;
}
}
std::vector<std::byte> encode_mtp_qualification_reference(const MtpQualificationReference& reference,std::uint64_t maximum){
    require(maximum>=160U && maximum<=limit,"qualification file budget is invalid");
    require(reference.independent_source_content.valid() && reference.target_content.valid() && reference.sidecar_content.valid() &&
        reference.independent_reference_identity && reference.proposal_identity && reference.plan_identity && reference.semantic_identity && reference.native_profile_identity,
        "qualification encoding lacks provenance pins");
    Writer a{{},maximum};for(char c:std::string_view("MTPQREF2"))a.word(static_cast<unsigned char>(c),1);
    a(std::uint32_t{2},reference.kind,std::uint64_t{0});metadata(a,reference);selected(a,reference);
    const auto bytes=static_cast<std::uint64_t>(a.data.size());for(unsigned i=0;i<8U;++i)a.data[16U+i]=static_cast<std::byte>((bytes>>(8U*i))&255U);
    return std::move(a.data);
}
MtpQualificationReference read_mtp_qualification_reference(const std::filesystem::path& path,const ModelContentIdentity& sha,std::uint64_t maximum){
    require(sha.valid() && maximum>=160U && maximum<=limit,"qualification file requires digest and bounded budget");
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"cannot open qualification reference");const auto end=f.tellg();
    require(end>=std::streampos{0} && static_cast<std::uint64_t>(end)<=maximum && end>=std::streampos{160},"qualification file exceeds bound or header missing");
    std::vector<std::byte> data(static_cast<std::size_t>(end));f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(data.size()));
    require(bool(f) && f.peek()==std::char_traits<char>::eof(),"qualification file changed or truncated");
    require(sha256_bytes(data)==sha,"qualification complete-file SHA256 differs");Reader a{data};
    for(char c:std::string_view("MTPQREF2"))require(a.word(1)==static_cast<unsigned char>(c),"qualification file magic differs");
    require(a.word(4)==2U,"unsupported qualification schema");MtpQualificationReference g;a(g.kind);
    require(a.word(8)==data.size(),"qualification complete length differs");metadata(a,g);selected(a,g);require(a.offset==data.size(),"qualification file has trailing bytes");bind(g);return g;
}
} // namespace strix::tests
