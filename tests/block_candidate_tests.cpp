#include "strix/spec/block_candidate.hpp"
#include "strix/spec/block_draft_policy.hpp"
#include "strix/serve/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>

namespace sw = strix::weight;
namespace sp = strix::spec;
namespace ref = strix::spec::reference;
namespace sj = strix::serve;
namespace {
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class F> void rejects(F f) { bool caught = false; try { f(); } catch (const std::exception&) { caught = true; } check(caught, "expected refusal"); }
template<class F> void rejects_at_hook(F f, std::string_view message) {
    try { f(); }
    catch (const std::invalid_argument& e) {
        check(e.what() == message, "expected refusal at the hook boundary");
        return;
    }
    throw std::runtime_error("expected hook refusal");
}
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
std::pair<sj::Json, std::uint64_t> header_json(const sw::BlockDraftConfig& c, const std::string& bad = {}) {
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
    return {std::move(root), offset};
}
std::shared_ptr<HeaderSource> header_source(std::string_view json, std::uint64_t payload_bytes) {
    auto source = std::make_shared<HeaderSource>();
    const auto n = std::uint64_t(json.size());
    for (unsigned i = 0; i < 8; ++i) source->bytes.push_back(static_cast<sw::Byte>((n >> (i*8)) & 255));
    for (unsigned char ch : json) source->bytes.push_back(static_cast<sw::Byte>(ch));
    source->total = source->bytes.size()+payload_bytes;
    return source;
}
std::shared_ptr<HeaderSource> header(const sw::BlockDraftConfig& c, const std::string& bad = {}) {
    const auto [root, payload_bytes] = header_json(c, bad);
    return header_source(root.dump(), payload_bytes);
}
void metadata_limit_tests() {
    const auto json = config_json();
    const auto c = sw::parse_block_draft_config(json);
    const auto [root, payload_bytes] = header_json(c);
    auto source = header(c);
    sw::BlockDraftCheckpoint checkpoint;
    check(bool(sw::index_block_draft(source, json, checkpoint)), "small checkpoint admitted");
    const auto config_source = sw::memory_source({reinterpret_cast<const sw::Byte*>(json.data()), json.size()});
    sw::BlockDraftCheckpoint from_source;
    check(bool(sw::index_block_draft(source, *config_source, from_source)) &&
          from_source.config_digest.bytes == checkpoint.config_digest.bytes &&
          from_source.index.header_identity.bytes == checkpoint.index.header_identity.bytes,
          "bounded config source preserves metadata identities");
    const auto digest = checkpoint.index.header_identity.bytes;

    auto altered = sj::parse_json_unique(json);
    altered.set("description", std::string(16 * 1024, 'x'));
    const auto reads = source->reads;
    const auto refuse = [&](const auto& input, std::string_view config, const char* reason) {
        const auto status = sw::index_block_draft(input, config, checkpoint);
        check(!status && status.detail.find(reason) != std::string::npos, reason);
        check(checkpoint.index.header_identity.bytes == digest, "refusal preserves the admitted checkpoint");
    };
    refuse(source, altered.dump(), "JSON byte limit");
    check(source->reads == reads, "oversized config rejected before header I/O");
    altered.set("description", sj::Json::Array(256, sj::Json(0)));
    refuse(source, altered.dump(), "too many values");
    check(source->reads == reads, "over-budget config rejected before header I/O");
    HeaderSource oversized_config;
    oversized_config.total = 16U * 1024U + 1;
    const auto status = sw::index_block_draft(source, oversized_config, checkpoint);
    check(!status && status.detail.find("JSON byte limit") != std::string::npos &&
          source->reads == reads && oversized_config.reads == 0, "oversized config source rejected before reading");

    source = header_source(root.dump() + std::string(64 * 1024, ' '), payload_bytes);
    refuse(source, json, "invalid safetensors header size");
    check(source->reads == 8, "oversized header rejected after reading only its prefix");
    const auto tensor = [](sj::Json::Array shape, std::uint64_t first, std::uint64_t last) {
        auto result = sj::Json::object();
        result.set("dtype", "BF16");
        result.set("shape", std::move(shape));
        result.set("data_offsets", sj::Json::Array{sj::Json(first), sj::Json(last)});
        return result;
    };
    altered = sj::parse_json_unique(root.dump());
    altered.set("fc.weight", tensor(sj::Json::Array(256, sj::Json(0)), 0, 0));
    refuse(header_source(altered.dump(), payload_bytes), json, "too many values");
    altered = sj::parse_json_unique(root.dump());
    altered.set("extra.weight", tensor({sj::Json(0)}, payload_bytes, payload_bytes));
    refuse(header_source(altered.dump(), payload_bytes), json, "too many safetensors tensors");
    altered = sj::parse_json_unique(root.dump());
    auto metadata = sj::Json::object();
    for (unsigned i = 0; i < 16; ++i) metadata.set(std::to_string(i), "value");
    altered.set("__metadata__", metadata);
    sw::BlockDraftCheckpoint with_metadata;
    check(bool(sw::index_block_draft(header_source(altered.dump(), payload_bytes), json, with_metadata)) &&
          with_metadata.index.metadata.size() == 16, "metadata at the limit is admitted");
    metadata.set("extra", "value");
    altered.set("__metadata__", metadata);
    refuse(header_source(altered.dump(), payload_bytes), json, "too many safetensors metadata entries");

    auto duplicate = root.dump();
    duplicate.pop_back();
    duplicate += ",\"fc.weight\":" + root.find("fc.weight")->dump() + "}";
    refuse(header_source(duplicate, payload_bytes), json, "duplicate object key");
    altered = sj::parse_json_unique(root.dump());
    altered.set("fc.weight", tensor({sj::Json(4), sj::Json(8)}, 0, 64));
    refuse(header_source(altered.dump(), payload_bytes), json, "overlap or unclaimed hole");
    altered.set("fc.weight", tensor({sj::Json(UINT64_C(9223372036854775807)), sj::Json(3)}, 0, 64));
    refuse(header_source(altered.dump(), payload_bytes), json, "shape overflow");
    std::cout << "PASS metadata limits: config and header bytes/values, tensor and metadata counts, duplicate, overlap, overflow\n";
}
void compatibility_tests() {
    const auto json = config_json();
    const auto base = sw::parse_block_draft_config(json);
    sw::BlockDraftCheckpoint checkpoint;
    check(bool(sw::index_block_draft(header(base), json, checkpoint)), "candidate geometry admitted");
    const auto refuse = [&](const auto& config, const auto& parsed) {
        auto source = header(config);
        const auto status = sw::index_block_draft(source, parsed.dump(), checkpoint);
        check(!status && source->reads == 0, "unsupported candidate geometry rejected before header I/O");
        check(checkpoint.config.block == base.block && checkpoint.config.layers == base.layers,
              "geometry refusal preserves previously admitted config");
        rejects([&] { static_cast<void>(sp::candidate_block_policy(config)); });
    };
    auto c = base;
    c.layers = 9;
    auto parsed = sj::parse_json_unique(json);
    parsed.set("num_hidden_layers", 9);
    parsed.set("layer_types", sj::Json::Array(9, sj::Json("sliding_attention")));
    refuse(c, parsed);
    for (const int rows : {9, 18, 32}) {
        c = base;
        c.block = static_cast<std::uint32_t>(rows);
        parsed = sj::parse_json_unique(json);
        auto candidate = *parsed.find("dflash_config");
        candidate.set("block_size", rows);
        parsed.set("dflash_config", candidate);
        refuse(c, parsed);
    }
    c = base;
    c.target_layers = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    parsed = sj::parse_json_unique(json);
    parsed.set("num_target_layers", 8);
    auto candidate = *parsed.find("dflash_config");
    candidate.set("target_layer_ids", sj::Json::Array{0, 1, 2, 3, 4, 5, 6, 7, 8});
    parsed.set("dflash_config", candidate);
    refuse(c, parsed);
    c.target_layers = {1, 65536};
    parsed.set("num_target_layers", 65536);
    candidate.set("target_layer_ids", sj::Json::Array{1, 65536});
    parsed.set("dflash_config", candidate);
    refuse(c, parsed);
    std::cout << "PASS candidate geometry: layer/tap count, tap id and block width limits\n";
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
    sp::CandidateTargetHooks hooks; hooks.live_vocab=6;
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

void hook_boundary_tests() {
    const auto c = sw::parse_block_draft_config(config_json());
    const auto w = weights(c);
    std::vector<std::uint32_t> ids(c.block, c.mask), out{42};
    ids.front() = 1;
    std::vector<float> confidence{0.5f};
    sp::CandidateTargetHooks hooks; hooks.live_vocab = c.vocab;
    hooks.features = [] { return sp::DraftFeatureWindow{ref::Matrix(2, 8), 0}; };
    hooks.accepted = [](auto) {};
    hooks.raw_logits = [&](const auto&) -> ref::Matrix { throw std::runtime_error("unexpected head call"); };
    const auto no_weights = [](const auto&) -> std::span<const float> {
        // Catch premature oracle entry without executing the unchecked row copy.
        throw std::runtime_error("embedding rows must be checked before the oracle reads weights");
    };
    for (const std::size_t rows : {1U, 9U}) {
        hooks.embed = [&](auto) { return ref::Matrix(rows, c.hidden); };
        sp::CandidateBlockForward forward(c, no_weights, hooks);
        rejects_at_hook([&] { forward.forward({ids, 2, 0, 7}, out, confidence); },
                        "candidate block: embedding hook returned wrong row count");
    }
    hooks.embed = [](auto) -> ref::Matrix { throw std::runtime_error("unexpected embedding call"); };
    sp::CandidateBlockForward incomplete(c, no_weights, hooks);
    rejects_at_hook([&] { incomplete.forward({std::span(ids).first(7), 2, 0, 6}, out, confidence); },
                    "candidate block: invalid block request");
    hooks.embed = [&](auto input) { return ref::Matrix(input.size(), c.hidden); };
    for (const std::size_t rows : {5U, 7U}) {
        hooks.raw_logits = [&, rows](const auto& hidden) {
            check(hidden.rows == 6, "head receives exactly the requested proposal rows");
            return ref::Matrix(rows, c.vocab);
        };
        sp::CandidateBlockForward forward(c, borrow(w), hooks);
        rejects_at_hook([&] { forward.forward({ids, 2, 0, 6}, out, confidence); },
                        "candidate block: raw-logit hook returned wrong row count");
    }
    check(out == std::vector<std::uint32_t>{42} && confidence == std::vector<float>{0.5f},
          "refused requests leave outputs untouched");
    // Live head width (6) is narrower than the padded storage count (8): the hook must return exactly 6 columns.
    hooks.live_vocab = c.vocab - 2U;
    for (const std::size_t cols : {std::size_t{c.vocab - 3U}, std::size_t{c.vocab - 1U}, std::size_t{c.vocab}}) {
        hooks.raw_logits = [&, cols](const auto&) { return ref::Matrix(6, cols); };
        sp::CandidateBlockForward forward(c, borrow(w), hooks);
        rejects_at_hook([&] { forward.forward({ids, 2, 0, 6}, out, confidence); },
                        "candidate block: raw-logit hook returned wrong width");
    }
    check(out == std::vector<std::uint32_t>{42}, "width refusals leave outputs untouched");
    hooks.raw_logits = [&](const auto&) { return ref::Matrix(6, c.vocab - 2U); };
    sp::CandidateBlockForward exact(c, borrow(w), hooks);
    exact.forward({ids, 2, 0, 6}, out, confidence);
    check(out.size() == 6 && confidence.empty(), "exact live width is accepted");
    for (const std::uint32_t id : {c.vocab - 2U, c.vocab - 1U}) {
        auto anchored = ids; anchored.front() = id;
        rejects_at_hook([&] { exact.forward({anchored, 2, 0, 6}, out, confidence); },
                        "candidate block: anchor outside the live head width");
    }
    for (const std::uint32_t live : {0U, c.top_k - 1U, c.vocab + 1U}) {
        auto bad = hooks; bad.live_vocab = live;
        rejects_at_hook([&] { sp::CandidateBlockForward refused(c, borrow(w), bad); },
                        "candidate block: invalid live head width");
    }
    std::cout << "PASS hook boundaries: full block, short/long embeddings and raw logits\n";
}

// The shared reader gained a limit-taking overload; the incumbent two-argument entry must be unchanged.
void incumbent_indexing_tests() {
    auto root = sj::Json::object();
    auto metadata = sj::Json::object();
    metadata.set("format", "pt"); metadata.set("note", "caf\xC3\xA9 \\ \"q\""); metadata.set("n", "3");
    root.set("__metadata__", metadata);
    std::uint64_t offset = 0;
    const struct { const char* name; const char* dtype; std::uint64_t width; std::vector<std::uint64_t> shape; } specs[] = {
        {"a.weight", "BF16", 2, {3, 4}}, {"b.bias", "F32", 4, {5}}, {"c.ids", "I32", 4, {2, 2, 2}}, {"d.scalar", "U8", 1, {1}}};
    for (const auto& s : specs) {
        auto dims = sj::Json::array(); std::uint64_t elements = 1;
        for (auto d : s.shape) { dims.push(sj::Json(d)); elements *= d; }
        auto tensor = sj::Json::object(); tensor.set("dtype", s.dtype); tensor.set("shape", dims);
        tensor.set("data_offsets", sj::Json(sj::Json::Array{sj::Json(offset), sj::Json(offset + elements * s.width)}));
        root.set(s.name, tensor); offset += elements * s.width;
    }
    // The incumbent entry hashes the whole source, so use a fully materialised in-memory file.
    auto file = header_source(root.dump(), 0)->bytes; file.resize(file.size() + offset);
    const auto source = sw::memory_source(file);
    sw::TensorIndex two, three;
    { const auto st = sw::index_safetensors(source, two); check(bool(st), ("two-argument indexing: " + st.detail).c_str()); }
    check(bool(sw::index_safetensors(source, three, sw::SafetensorsIndexLimits{})), "default-limits indexing");
    check(two.container == three.container && two.schema == three.schema && two.owner == three.owner,
          "container, schema and owner agree");
    check(two.header_identity.bytes == three.header_identity.bytes, "header identity agrees");
    check(two.metadata == three.metadata && two.metadata.size() == 3, "metadata agrees");
    check(two.metadata_blobs.size() == three.metadata_blobs.size(), "metadata blobs agree");
    check(two.tensors.size() == 4 && two.tensors.size() == three.tensors.size(), "tensor count agrees");
    for (std::size_t i = 0; i < two.tensors.size(); ++i) {
        const auto &x = two.tensors[i], &y = three.tensors[i];
        check(x.name == y.name && x.id == y.id && x.wire_dtype == y.wire_dtype && x.wire_type == y.wire_type &&
              x.extents == y.extents && x.shape.rank == y.shape.rank && x.shape.dims == y.shape.dims &&
              x.payload.source == y.payload.source && x.payload.range.offset == y.payload.range.offset &&
              x.payload.range.bytes == y.payload.range.bytes, "tensor record agrees");
    }
    const auto text = root.dump();
    const auto limited = sj::parse_json_unique(text, SIZE_MAX, SIZE_MAX);
    check(sj::parse_json_unique(text).dump() == limited.dump() && limited.dump() == text, "unbounded JSON overload agrees");
    rejects([&] { static_cast<void>(sj::parse_json_unique(text, text.size() - 1U, SIZE_MAX)); });
    rejects([&] { static_cast<void>(sj::parse_json_unique(text, SIZE_MAX, 3)); });
    std::cout << "PASS incumbent safetensors indexing: two-argument entry equals default-limits entry\n";
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
}
int main(int argc,char** argv) {
    try {
        if(argc==4 && std::string_view(argv[1])=="--header-only") {
            std::shared_ptr<const sw::Source> source, config;
            auto status=sw::open_source(argv[2],config);
            check(bool(status),status.detail.c_str());
            status=sw::open_source(argv[3],source,"block-draft.header-only@1");
            check(bool(status),status.detail.c_str()); sw::BlockDraftCheckpoint c;
            status=sw::index_block_draft(source,*config,c);check(bool(status),status.detail.c_str());
            std::cout<<"PASS checkpoint HEADER ONLY: "<<c.index.tensors.size()<<" tensors, "<<c.payload_bytes<<" payload bytes; payload NOT read\n";
            return 0;
        }
        loader_tests();metadata_limit_tests();compatibility_tests();math_tests();selector_tests();
        adapter_tests();hook_boundary_tests();incumbent_indexing_tests();cycle_tests();
        std::cout<<"PASS block candidate host suite\n";
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
