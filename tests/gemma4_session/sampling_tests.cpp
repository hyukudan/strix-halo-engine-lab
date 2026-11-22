#include "gemma4/engine/api.hpp"
#include "strix/serve/engine.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
namespace g = strix::gemma4;
namespace e = g::engine;
namespace sv = strix::serve;

void check(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
}

class Stream final : public e::StreamState {
    class Txn final : public g::KvTxn {
    public:
        explicit Txn(std::uint64_t& cursor) : cursor_(cursor) {}
        std::uint64_t frontier() const override { return cursor_; }
        void reserve_rows(std::size_t) override {}
        void commit(std::size_t count) override { cursor_ += count; }
    private:
        std::uint64_t& cursor_;
    };
public:
    std::unique_ptr<g::KvTxn> begin() override { return std::make_unique<Txn>(cursor_); }
    std::uint64_t frontier() const override { return cursor_; }
    std::string digest() const override { return std::to_string(cursor_); }
private:
    std::uint64_t cursor_ = 0;
};

constexpr std::uint32_t winner = 2065;
class Backend final : public e::Backend {
public:
    Backend() {
        config_.vocab = 262144;
        config_.hidden = 1;
        config_.max_positions = 96;
        capped.resize(config_.vocab,-12.0F);
        capped[winner] = capped[winner+1] = 2.0F;
        capped[winner+2] = 1.8F;
        capped[winner+3] = 1.1F;
    }
    const g::ModelConfig& config() const override { return config_; }
    sv::Json identity() const override { return sv::Json(sv::Json::Object{{"identity","sampling-test"}}); }
    std::unique_ptr<e::StreamState> create_stream() override { return std::make_unique<Stream>(); }
    g::ForwardResult forward_rows(std::span<const g::RowPlan> plans) override {
        g::ForwardResult out;
        out.rows = plans.size();
        out.vocab = config_.vocab;
        out.tap_width = 5*config_.hidden;
        out.device_checked = true;
        out.selected.assign(out.rows,winner);
        if (std::any_of(plans.begin(),plans.end(),[](const auto& p) { return p.export_logits; }))
            out.logits.resize(out.rows*out.vocab);
        for (const auto& p : plans) {
            if (p.export_logits) {
                ++exports;
                std::copy(capped.begin(),capped.end(),out.logits.data()+p.output_row*out.vocab);
            }
        }
        return out;
    }
    std::vector<float> capped;
    std::size_t exports = 0;
private:
    g::ModelConfig config_;
};

class Writer final : public sv::HttpResponseWriter {
    using Headers = std::vector<std::pair<std::string,std::string>>;
public:
    void send(int code, std::string_view, std::string_view body, const Headers&) override {
        status = code;
        bytes = body;
    }
    bool begin_stream(int, std::string_view, const Headers&) override { throw std::logic_error("unexpected stream"); }
    bool write(std::string_view) override { throw std::logic_error("unexpected stream"); }
    void end_stream() override {}
    bool client_gone() const override { return false; }
    int status = 0;
    std::string bytes;
};

Writer request(e::Api& api, const char* path, const sv::Json& body, const char* method = "POST") {
    sv::HttpRequest r;
    r.path = path;
    r.method = method;
    r.body = body.dump();
    Writer writer;
    api.handle(r,writer);
    return writer;
}

std::vector<std::uint32_t> ids(const sv::Json& receipt) {
    std::vector<std::uint32_t> out;
    for (const auto& id : receipt.find("generated_ids")->as_array()) out.push_back(std::uint32_t(id.as_int()));
    return out;
}

void verify(const char* tokenizer_path) {
    Backend backend;
    const auto drafts = []() -> std::unique_ptr<e::DraftSession> { throw std::runtime_error("sampled request opened a drafter"); };
    auto tokenizer = g::serve::Tokenizer::load(tokenizer_path);
    e::Api api(backend,tokenizer,drafts,e::DraftMode::off,"model");

    const auto caps_reply = request(api,"/v1/capabilities",sv::Json::object(),"GET");
    check(caps_reply.status == 200,"capabilities request failed");
    const auto caps = sv::parse_json_unique(caps_reply.bytes);
    check(caps.find("sampling")->find("sample")->as_bool(),"API sampling capability missing");
    check(caps.find("session_sampling")->find("min_p")->as_bool(),"session min_p capability missing");

    const auto run = [&](const sv::Json& body) {
        const auto reply = request(api,"/v1/completions",body);
        if (reply.status != 200) throw std::runtime_error(reply.bytes);
        const auto envelope = sv::parse_json_unique(reply.bytes);
        const auto* receipt = envelope.find("strix");
        check(receipt != nullptr,"missing generation receipt");
        return *receipt;
    };
    constexpr std::uint32_t steps = 12;
    const auto prompt = sv::parse_json("[2,2065,2066]");
    sv::Json body(sv::Json::Object{{"prompt",prompt},{"max_tokens",std::uint64_t(steps)}});
    const auto greedy = ids(run(body));
    body.set("temperature",0.0);
    body.set("top_p",0.8);
    body.set("top_k",3);
    body.set("min_p",0.3);
    body.set("seed",19);
    check(greedy == std::vector<std::uint32_t>(steps,winner) && ids(run(body)) == greedy,"temperature 0 changed greedy IDs");
    check(backend.exports == 0,"greedy requested logits export");
    std::cout << "temperature 0 IDs match greedy; zero logits exports\n";

    sv::GenerationJob job;
    job.sampling = {0.83,0.96,4,0.4,19,true};
    const auto adapter = e::sampling_from_request(job.sampling,7);
    auto sampler = adapter.make_sampler();
    check(sampler.has_value() && adapter.draws == 7,"sampling adapter lost the sampler or progress cursor");
    const auto& policy = sampler->policy();
    check(policy.temperature == float(job.sampling.temperature) && policy.top_p == float(job.sampling.top_p) &&
          policy.top_k == job.sampling.top_k && policy.min_p == float(job.sampling.min_p) && policy.seed == job.sampling.seed,
          "sampling adapter lost request parameters");
    body.set("temperature",job.sampling.temperature);
    body.set("top_p",job.sampling.top_p);
    body.set("top_k",std::uint64_t(job.sampling.top_k));
    body.set("min_p",job.sampling.min_p);
    body.set("seed",job.sampling.seed);
    body.set("dflash","on");
    std::vector<std::uint32_t> expected;
    for (std::uint64_t p = prompt.size(); p < prompt.size()+steps; ++p)
        expected.push_back(sampler->decide(backend.capped,p,nullptr).token);
    const auto sampled = run(body);
    check(sampled.find("sampling")->as_string() == "sampling-v1","sampling receipt identity missing");
    check(ids(sampled) == expected && ids(run(body)) == expected,"HTTP sampling differs from shared absolute-position decisions");
    body.set("seed",37);
    const auto different = ids(run(body));
    check(different != expected,"different seeds produced identical IDs");

    e::Request lane;
    lane.prompt = {2,winner,winner+1};
    lane.max_tokens = steps;
    lane.dflash = e::DraftMode::on;
    lane.sampling = adapter;
    e::Coordinator mixed(backend,drafts);
    const auto first = mixed.submit(lane);
    lane.sampling.seed = 37;
    const auto second = mixed.submit(lane);
    lane.sampling = {};
    lane.dflash = e::DraftMode::off;
    const auto plain = mixed.submit(lane);
    mixed.run();
    check(mixed.result(first).ids == expected && mixed.result(second).ids == different && mixed.result(plain).ids == greedy,
          "mixed cohort changed token IDs");
    check(mixed.result(first).rng_draws == 7+steps && mixed.result(plain).rng_draws == 0,"session progress cursor changed");
    check(backend.exports == 5*steps,"sampled decode did not export exactly one capped row per token");
    std::cout << "sampling-v1 matched the shared policy and absolute positions; same seed IDs match, different seeds differ, drafts off\n";

    body.set("logit_bias",sv::parse_json("{\"2065\": 1}"));
    const auto invalid = request(api,"/v1/completions",body);
    check(invalid.status == 400 && invalid.bytes.find("logit_bias") != std::string::npos,"unsupported field lacked a named HTTP 400");
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 2,"tokenizer fixture path required");
        verify(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
