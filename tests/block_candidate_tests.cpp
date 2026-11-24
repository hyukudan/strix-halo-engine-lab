#include "strix/spec/block_candidate.hpp"
#include "strix/spec/block_draft_policy.hpp"
#include "strix/serve/json.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

namespace sw = strix::weight;
namespace sp = strix::spec;
namespace ref = strix::spec::reference;
namespace sj = strix::serve;
namespace {
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class F> void rejects(F f) { bool caught = false; try { f(); } catch (const std::exception&) { caught = true; } check(caught, "expected refusal"); }
void near(double a, double b) { check(std::abs(a-b) < 1e-10, "numerical mismatch"); }
std::string config_json() {
    return R"({"hidden_size":4,"intermediate_size":6,"num_hidden_layers":1,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"vocab_size":8,
        "sliding_window":16,"rms_norm_eps":0.000001,"attention_bias":false,"is_causal":false,
        "hidden_act":"silu","dtype":"bfloat16","num_target_layers":4,
        "layer_types":["sliding_attention"],"rope_parameters":{"rope_theta":10000,"rope_type":"default"},
        "dflash_config":{"block_size":8,"conv_group_size":2,"conv_kernel_size":2,"selector_rank":2,
        "selector_top_k":3,"mask_token_id":7,"target_layer_ids":[1,4],"final_logit_softcapping":30,
        "selector_softcap_mode":"after_correction"}})";
}
using Weights = std::map<std::string, std::vector<float>>;
Weights weights(const sw::BlockDraftConfig& c) {
    Weights w;
    for (const auto& [name, shape] : sw::block_draft_shapes(c)) {
        std::size_t n = 1; for (auto d : shape) n *= static_cast<std::size_t>(d);
        w[name].resize(n);
        if (name.find("norm.weight") != name.npos) std::fill(w[name].begin(), w[name].end(), 1.0f);
    }
    return w;
}
ref::Weights borrow(const Weights& w) { return [&w](const std::string& name) { return std::span<const float>(w.at(name)); }; }

// The spy refuses any payload access OR implicit source hashing during indexing.
struct HeaderSource : sw::Source {
    std::vector<sw::Byte> bytes;
    std::uint64_t total{};
    mutable std::uint64_t reads{};
    sw::Status read(sw::ByteRange range, std::span<sw::Byte> out) const override {
        if (range.offset > bytes.size() || range.bytes > bytes.size()-range.offset || range.bytes != out.size())
            return {sw::Error::malformed, "test: attempted payload read"};
        std::copy_n(bytes.begin()+static_cast<std::ptrdiff_t>(range.offset), out.size(), out.begin());
        reads += out.size(); return {sw::Error::ok, {}};
    }
    std::uint64_t size() const noexcept override { return total; }
    sw::Digest256 identity() const override { throw std::runtime_error("test: attempted full-file digest"); }
};
std::shared_ptr<HeaderSource> header(const sw::BlockDraftConfig& c, const std::string& bad = {}) {
    auto source = std::make_shared<HeaderSource>();
    auto root = sj::Json::object();
    std::uint64_t offset = 0;
    for (const auto& [name, shape] : sw::block_draft_shapes(c)) {
        auto dimensions = sj::Json::array(); std::uint64_t elements = 1;
        for (auto d : shape) { dimensions.push(sj::Json(d)); elements *= d; }
        auto tensor = sj::Json::object(); tensor.set("dtype", name == bad ? "F16" : "BF16");
        tensor.set("shape", dimensions);
        tensor.set("data_offsets", sj::Json(sj::Json::Array{sj::Json(offset), sj::Json(offset+elements*2)}));
        root.set(name, tensor); offset += elements*2;
    }
    const auto json = root.dump();
    const auto n = std::uint64_t(json.size());
    for (unsigned i = 0; i < 8; ++i) source->bytes.push_back(static_cast<sw::Byte>((n >> (i*8)) & 255));
    for (unsigned char ch : json) source->bytes.push_back(static_cast<sw::Byte>(ch));
    source->total = source->bytes.size()+offset;
    return source;
}
void loader_tests() {
    const auto json = config_json(); const auto c = sw::parse_block_draft_config(json);
    auto source = header(c); sw::BlockDraftCheckpoint checkpoint;
    check(bool(sw::index_block_draft(source, json, checkpoint)), "header admission");
    check(source->reads == source->bytes.size(), "only prefix and header read");
    check(checkpoint.index.tensors.size() == 21, "tensor table");
    check(checkpoint.payload_bytes == source->total-source->bytes.size(), "payload byte count");
    const auto digest = checkpoint.config_digest.bytes;
    check(!sw::index_block_draft(header(c, "fc.weight"), json, checkpoint), "bad dtype rejected");
    check(checkpoint.config_digest.bytes == digest, "failed admission preserves output");
    auto altered = sj::parse_json_unique(json); altered.set("hidden_size", 6);
    check(!sw::index_block_draft(source, altered.dump(), checkpoint), "shape mismatch rejected");
    altered = sj::parse_json_unique(json); altered.set("is_causal", true);
    rejects([&] { static_cast<void>(sw::parse_block_draft_config(altered.dump())); });
    altered = sj::parse_json_unique(json); altered.set("output_multiplier", 2);
    rejects([&] { static_cast<void>(sw::parse_block_draft_config(altered.dump())); });
    rejects([&] { static_cast<void>(sw::parse_block_draft_config("{\"hidden_size\":4,\"hidden_size\":6}")); });
    source->bytes[0] = sw::Byte{0xff}; source->bytes[7] = sw::Byte{0xff};
    check(!sw::index_block_draft(source, json, checkpoint), "oversized header rejected");
    // Synthetic payload loading/digesting only, never the downloaded weights.
    source = header(c);
    source->bytes.resize(static_cast<std::size_t>(source->total), sw::Byte{0});
    auto memory = sw::memory_source(source->bytes);
    check(bool(sw::index_block_draft(memory, json, checkpoint)), "synthetic checkpoint");
    std::vector<float> decoded;
    check(bool(sw::read_block_draft_tensor(checkpoint, "fc.weight", decoded)) && decoded.size() == 32, "BF16 payload read");
    check(std::all_of(decoded.begin(), decoded.end(), [](float v) { return v == 0; }), "BF16 zeros");
    std::map<std::string, sw::Digest256> hashes;
    check(bool(sw::digest_block_draft_payload(checkpoint, hashes)) && hashes.size() == 21, "tensor payload digests");
    const auto old_hash=hashes.at("fc.weight").bytes;
    const auto it=std::find_if(checkpoint.index.tensors.begin(),checkpoint.index.tensors.end(),[](const auto& t){return t.name=="fc.weight";});
    const auto offset=static_cast<std::size_t>(it->payload.range.offset);
    source->bytes[offset]=sw::Byte{0x80};source->bytes[offset+1]=sw::Byte{0x3f};
    memory=sw::memory_source(source->bytes);
    check(bool(sw::index_block_draft(memory,json,checkpoint)),"mutated synthetic payload");
    check(bool(sw::read_block_draft_tensor(checkpoint,"fc.weight",decoded)) && decoded.front()==1.0f,"BF16 little endian one");
    check(bool(sw::digest_block_draft_payload(checkpoint,hashes)) && hashes.at("fc.weight").bytes!=old_hash,"payload digest detects changed byte");
    std::cout << "PASS loader: bounded header-only, shapes, dtype, duplicate JSON, atomic failure, synthetic BF16/digests\n";
}
void math_tests() {
    const auto c = sw::parse_block_draft_config(config_json()); auto w = weights(c);
    ref::Matrix x(3,4); x.data = {1,2,3,4, 2,4,6,8, 3,6,9,12};
    ref::Matrix dynamics(3,8); std::fill(dynamics.data.begin(), dynamics.data.end(), 0.5);
    std::vector<float> base(16, 1);
    const auto conv = ref::convolve(x, dynamics, base, 2, 2, 0);
    near(conv(0,0), 1.5); near(conv(1,3), 18); near(conv(2,2), 22.5);
    ref::Matrix features(2,8), noise(8,4);
    for (std::size_t i=0; i<noise.data.size(); ++i) noise.data[i] = 0.2 + static_cast<double>(i%7);
    const auto identity = ref::forward(c, borrow(w), features, noise);
    const auto expected = ref::rms(noise, w.at("norm.weight"), c.epsilon);
    check(identity.data == expected.data, "zero branches give normalized residual");
    // Analytic nonzero attention: Q=K=0 -> uniform over context + all block
    // rows, V picks channels 0,1 and O picks first head. Conv phases are identity.
    auto& kernel = w.at("layers.0.attention_conv.base_kernel");
    for (std::size_t i=0; i<4; ++i) { kernel[i] = 1; kernel[8+i] = 1; }
    auto& vp = w.at("layers.0.self_attn.v_proj.weight"); vp[0]=1; vp[5]=1;
    auto& op = w.at("layers.0.self_attn.o_proj.weight"); op[0]=1; op[5]=1;
    const auto normalized = ref::rms(noise, w.at("norm.weight"), c.epsilon);
    auto analytic = noise;
    for (std::size_t d=0; d<2; ++d) {
        double mean=0; for (std::size_t r=0; r<noise.rows; ++r) mean += normalized(r,d)/10;
        for (std::size_t r=0; r<noise.rows; ++r) analytic(r,d) += mean;
    }
    analytic = ref::rms(analytic, w.at("norm.weight"), c.epsilon);
    const auto actual = ref::forward(c, borrow(w), features, noise);
    for (std::size_t i=0; i<actual.data.size(); ++i) near(actual.data[i], analytic.data[i]);
    auto shifted = ref::forward(c, borrow(w), features, noise, 11);
    check(actual.data == shifted.data, "zero QK independent of absolute offset");
    // Nonzero Q/K, grouped heads, RoPE: global position shifts preserve relative scores.
    auto& qp=w.at("layers.0.self_attn.q_proj.weight"); for (std::size_t i=0;i<4;++i) qp[5*i]=0.25f;
    auto& kp=w.at("layers.0.self_attn.k_proj.weight"); kp[0]=0.5f; kp[5]=0.75f;
    const auto rotary = ref::forward(c, borrow(w), features, noise);
    shifted = ref::forward(c, borrow(w), features, noise, 11);
    for (std::size_t i=0;i<rotary.data.size();++i) near(rotary.data[i],shifted.data[i]);
    check(rotary.data != actual.data, "nonzero QK changes attention");
    auto short_window=c; short_window.window=1;
    check(ref::forward(short_window,borrow(w),features,noise).data != rotary.data, "sliding mask enforced");
    // Independent one-channel SwiGLU calculation after the residual.
    w=weights(c); auto& mk=w.at("layers.0.mlp_conv.base_kernel"); mk[0]=1; mk[8]=1;
    w.at("layers.0.mlp.gate_proj.weight")[0]=1; w.at("layers.0.mlp.up_proj.weight")[0]=2;
    w.at("layers.0.mlp.down_proj.weight")[0]=3;
    analytic=noise;
    for (std::size_t r=0;r<noise.rows;++r) { const auto z=normalized(r,0); analytic(r,0)+=6*z*z/(1+std::exp(-z)); }
    analytic=ref::rms(analytic,w.at("norm.weight"),c.epsilon);
    const auto mlp=ref::forward(c,borrow(w),features,noise);
    for (std::size_t i=0;i<mlp.data.size();++i) near(mlp.data[i],analytic.data[i]);
    std::cout << "PASS reference: convolution boundary, residual, analytic attention, GQA/RoPE, sliding mask, SwiGLU\n";
}
void selector_tests() {
    auto c=sw::parse_block_draft_config(config_json()); auto w=weights(c);
    ref::Matrix h(2,4); h(0,0)=1; h(1,0)=1;
    ref::Matrix logits(2,6); logits.data={0,1,2,3,4,5, 0,1,2,3,4,5}; // excludes padded 6,7
    w.at("candidate_selector.hidden_projection.weight")[0]=1;
    w.at("candidate_selector.predecessor_codebook")[0]=1;
    w.at("candidate_selector.predecessor_codebook")[8]=-1;
    w.at("candidate_selector.successor_codebook")[8]=2;
    const auto result=ref::select(c,borrow(w),h,logits,0);
    check(result.path == std::vector<std::uint32_t>({4,5}), "selector follows predecessor, not independent argmax");
    const double u[]={0.2,0.8};
    const auto sampled=ref::select(c,borrow(w),h,logits,0,1,u);
    for(const auto& row:sampled.probabilities) near(std::accumulate(row.begin(),row.end(),0.0),1);
    const auto a=std::find(sampled.ids[0].begin(),sampled.ids[0].end(),4U)-sampled.ids[0].begin();
    const auto b=std::find(sampled.ids[0].begin(),sampled.ids[0].end(),5U)-sampled.ids[0].begin();
    near(std::log(sampled.probabilities[0][static_cast<std::size_t>(a)]/sampled.probabilities[0][static_cast<std::size_t>(b)]),
         30*std::tanh(6.0/30)-30*std::tanh(5.0/30));
    auto before=c; before.post_softcap=false;
    check(ref::select(before,borrow(w),h,logits,0,1,u).probabilities != sampled.probabilities, "softcap ordering matters");
    auto saturated=logits; std::fill(saturated.data.begin(),saturated.data.end(),10000);
    saturated(0,5)=20000; before.top_k=1;
    check(ref::select(before,borrow(w),h,saturated,0).path.front()==0,"before-correction top-k breaks saturated ties by id");
    rejects([&]{static_cast<void>(ref::select(c,borrow(w),h,logits,6));});
    std::cout << "PASS selector: conditional path, real vocab, sparse q, post-correction softcap\n";
}
void adapter_tests() {
    const auto c=sw::parse_block_draft_config(config_json()); const auto w=weights(c);
    unsigned captures=0, embeddings=0, heads=0, accepts=0;
    sp::CandidateTargetHooks hooks;
    hooks.features=[&] {++captures;return sp::DraftFeatureWindow{ref::Matrix(2,8),0};};
    hooks.embed=[&](std::span<const std::uint32_t> ids) {
        ++embeddings; check(ids.size()==8 && ids[0]==1 && ids[1]==c.mask,"embedding shares target token ids");
        ref::Matrix m(ids.size(),c.hidden); std::fill(m.data.begin(),m.data.end(),1); return m;
    };
    hooks.raw_logits=[&](const ref::Matrix& hidden) {
        ++heads; check(hidden.rows==7,"head excludes anchor hidden"); ref::Matrix m(hidden.rows,6);
        for(std::size_t r=0;r<m.rows;++r)m(r,2)=1;
        return m;
    };
    hooks.accepted=[&](std::span<const std::uint32_t> emitted) {++accepts;check(emitted.size()==2,"accepted tap frontier hook");};
    sp::CandidateBlockForward forward(c,borrow(w),hooks);
    sp::ParallelBlockDrafter draft(sp::candidate_block_policy(c),forward);draft.start(1,2);
    std::vector<std::uint32_t> out;draft.draft({},7,out);
    check(out==std::vector<std::uint32_t>(7,2) && captures==1 && embeddings==1 && heads==1,"one pass adapter");
    const std::uint32_t emitted[]={2,3}; draft.accept(emitted);
    check(accepts==1 && draft.anchor_pos()==4 && draft.last_confidence().empty(),"adapter acceptance");
    rejects([&]{draft.draft({},7,out);}); // deliberately stale target features
    std::cout << "PASS adapter: target hooks, one pass, selector, accepted feature frontier, stale-feature refusal\n";
}

struct FakeForward : sp::BlockForward {
    std::uint32_t accept_length{}, passes{}; std::vector<std::uint32_t> accepted;
    void forward(const sp::BlockRequest& req, std::vector<std::uint32_t>& out, std::vector<float>&) override {
        ++passes; check(req.ids.size()==8 && req.prefix==0,"one full diffusion block");
        for(std::uint32_t i=0;i<req.k;++i) out.push_back(req.ids[0]+i+1+(i>=accept_length?1000:0));
    }
    void accept(std::span<const std::uint32_t> emitted) override { accepted.assign(emitted.begin(),emitted.end()); }
};
struct FakeTarget : sp::VerifyTarget {
    std::vector<std::uint32_t> kv, staged; std::uint32_t width{}, commits{};
    void verify(std::span<const std::uint32_t> ids, std::vector<std::uint32_t>& out) override {
        width=static_cast<std::uint32_t>(ids.size()); staged.assign(ids.begin(),ids.end()); out.clear();
        for(auto id:ids) out.push_back(id+1); // genuinely conditions on wrong speculative rows
    }
    void commit(std::uint32_t n) override { check(n<=staged.size(),"commit range"); commits=n; kv.insert(kv.end(),staged.begin(),staged.begin()+n); staged.clear(); }
};
void cycle_tests() {
    const auto c=sw::parse_block_draft_config(config_json()); const auto policy=sp::candidate_block_policy(c);
    unsigned cases=0;
    for(std::uint32_t a=0;a<=7;++a) for(std::uint32_t cap=0;cap<=8;++cap)
        for(std::uint32_t budget=0;budget<=8;++budget) for(std::uint32_t stop=0;stop<=8;++stop) {
        FakeForward f; f.accept_length=a; sp::ParallelBlockDrafter p(policy,f); p.start(10,0);
        sp::BlockConfig base; base.copy=false; base.siblings=false; base.fixed_costs=true; base.max_block=8;
        base.clock_ms=[] {return 0.0;};
        auto cfg=sp::parallel_block_config(policy,0.001,base);
        sp::BlockDrafter d(cfg,&p); for(std::uint32_t r=1;r<=8;++r)d.seed_verify_ms(r,10);
        const double survival[7]={1,1,1,1,1,1,1}; d.seed_chain_survival(survival,1000000);
        FakeTarget target; sp::BlockSpeculator cycle(d,target); cycle.start(10);
        sp::StepParams params; params.remaining=budget; params.max_rows=cap;
        if(stop>0)params.eos=10+stop;
        std::vector<std::uint32_t> out; const auto record=cycle.step(out,params);
        const auto expected=std::min({a+1,cap,budget,stop>0?stop:8U});
        check(out.size()==expected,"cycle output length");
        for(std::size_t i=0;i<out.size();++i)check(out[i]==11+i,"same tokens as serial greedy");
        check(target.kv.size()==out.size() && target.staged.empty(),"rollback leaves accepted frontier");
        for(std::size_t i=0;i<target.kv.size();++i)check(target.kv[i]==10+i,"canonical committed state");
        check(p.anchor_pos()==out.size(),"draft frontier");
        if(expected)check(cycle.anchor()==out.back(),"correction/bonus is uncommitted anchor");
        if(cap==8 && budget==8 && stop==0)check(record.accepted==a && record.rows==8,"all accept lengths exercised");
        check(f.passes<=1,"one denoising pass"); ++cases;
    }
    std::cout << "PASS cycle: " << cases << " cases; accept 0..7, EOS, context cap, output budget, KV rollback\n";
}
void policy_tests() {
    namespace pf=strix::profile;
    const auto& engine=strix::inkling2::kMiniProfile;
    for(auto tier:{pf::Name::standard,pf::Name::fast,pf::Name::turbo}) {
        pf::Inputs cli;cli.profile=tier;cli.env=[](const char*)->const char*{return nullptr;};
        check(!sp::block_draft_selection(pf::resolve(engine,cli)).enabled(),"all defaults OFF");
        auto previous=engine;previous.table=engine.table.first(engine.table.size()-1);
        check(pf::resolve(previous,cli).fingerprint()==pf::resolve(engine,cli).fingerprint(),"OFF preserves previous profile fingerprint");
        for(const auto* value:{"off","dflash2:8:greedy","dflash2:8:sampled","dflash2:7:greedy","unknown:8:greedy"}) {
            cli.cli={{std::string(sp::kBlockDraftLeverName),value}};
            strix::inkling2::MiniWeightOptions cli_options, server_options;
            cli_options.tier.profile=tier; cli_options.tier.env=cli.env;
            // CLI token versus server/environment value, through their common
            // actual model parser/resolver (also checks CLI precedence).
            check(strix::inkling2::parse_mini_weight_option(cli_options,"--block-drafter",[&]{return value;},true),"CLI option consumed");
            server_options.tier.profile=tier;
            server_options.tier.env=[&](const char* key)->const char*{
                return std::string_view(key)=="STRIX_BLOCK_DRAFTER"?value:nullptr;
            };
            const auto outcome=[&](const strix::inkling2::MiniWeightOptions& input){try {
                static_cast<void>(strix::inkling2::resolve_mini_profile(input,true)); return std::string("off");
            }catch(const std::exception& e){return std::string(e.what());}};
            const auto lhs=outcome(cli_options),rhs=outcome(server_options);check(lhs==rhs,"CLI/server policy parity");
            if(std::string_view(value)!="off")check(lhs!="off","unbound native cannot run");
        }
    }
    std::cout << "PASS resolver: common row, EXACT greedy only, sampled pending, CLI/server parity, OFF in all tiers\n";
}
std::string read_text(const char* path) {std::ifstream f(path,std::ios::binary);check(bool(f),"cannot open config");std::ostringstream b;b<<f.rdbuf();return b.str();}
}
int main(int argc,char** argv) {
    try {
        if(argc==4 && std::string_view(argv[1])=="--header-only") {
            std::shared_ptr<const sw::Source> source; auto status=sw::open_source(argv[3],source,"block-draft.header-only@1");
            check(bool(status),status.detail.c_str()); sw::BlockDraftCheckpoint c;
            status=sw::index_block_draft(source,read_text(argv[2]),c);check(bool(status),status.detail.c_str());
            std::cout<<"PASS checkpoint HEADER ONLY: "<<c.index.tensors.size()<<" tensors, "<<c.payload_bytes<<" payload bytes; payload NOT read\n";
            return 0;
        }
        loader_tests();math_tests();selector_tests();adapter_tests();cycle_tests();policy_tests();
        std::cout<<"PASS block candidate host suite\n";
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
