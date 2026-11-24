// Real API parser + queue + shared sampler against the real command-line parser, sampler and decode loop,
// entirely in process (no sockets, no model).
#include "strix/serve/api.hpp"
#include "strix/serve/job_sampling.hpp"
#include "strix/serve/serial_generation_cli.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>

namespace {
namespace sv = strix::serve;
struct Tokenizer final : sv::ApiTokenizer {
    std::vector<std::uint32_t> encode(std::string_view, bool) const override { return {0, 1}; }
    std::string decode(std::span<const std::uint32_t> ids, bool) const override {
        std::string out;
        for (auto id : ids) out += token_bytes(id);
        return out;
    }
    std::string token_bytes(std::uint32_t id) const override { return std::string(1, static_cast<char>('a' + id)); }
    std::size_t vocab_size() const override { return 4; }
    std::uint32_t find(std::string_view) const override { return 3; }
    std::uint32_t eos_token() const override { return 3; }
};
constexpr std::array logits{0.1F, 0.2F, 0.3F, -100.0F};
std::vector<std::uint32_t> draw(strix::TokenSampler& sampler, std::size_t prompt, unsigned count) {
    std::vector<std::uint32_t> ids;
    for (unsigned i = 0; i < count; ++i) ids.push_back(sampler.decide(logits, prompt + i, nullptr).token);
    return ids;
}
struct Engine final : sv::GenerationEngine {
    std::uint64_t context_tokens() const override { return 256; }
    bool supports_sampling() const override { return true; }
    sv::GenerationStats generate(sv::GenerationJob& job) override {
        auto sampler = sv::make_job_sampler(job.sampling);
        const std::vector<std::uint32_t> ids = sampler ? draw(*sampler, job.prompt.size(), job.max_tokens)
                                                       : std::vector<std::uint32_t>(job.max_tokens, 2U);  // greedy: first max
        (void)job.on_tokens(ids);
        {
            const std::scoped_lock lock(mutex);
            served.emplace_back(job.sampling.seed, job.sampling.temperature, ids);
        }
        sv::GenerationStats stats;
        stats.prompt_tokens = job.prompt.size(); stats.generated_tokens = ids.size();
        if (sampler) stats.sampling = "sampling-v1";
        return stats;
    }
    std::mutex mutex;
    std::vector<std::tuple<std::uint64_t, double, std::vector<std::uint32_t>>> served;
};
// A decoder over the fixed logits: the real command-line parser, sampler and decode loop run on it.
struct Decoder final : sv::SerialGenerationDecoder {
    std::uint32_t prefill(std::span<const std::uint32_t>, std::span<float> out) override { return fill(out); }
    std::uint32_t decode(std::uint32_t, std::span<float> out) override { return fill(out); }
    static std::uint32_t fill(std::span<float> out) {
        std::copy_n(logits.begin(), std::min(out.size(), logits.size()), out.begin());
        return 2U;
    }
};
sv::SerialGenerationFamily test_family(bool with_decoder) {
    return sv::SerialGenerationFamily{"strix-test", "test", "CKPT", 4U, 4U, 256U, 3U,
        with_decoder ? std::function<std::unique_ptr<sv::SerialGenerationDecoder>(const sv::SerialGenerationLoad&)>(
            [](const sv::SerialGenerationLoad&) -> std::unique_ptr<sv::SerialGenerationDecoder> { return std::make_unique<Decoder>(); })
                     : nullptr};
}
// The command line's 64 generated ids for the given options.
std::vector<std::uint32_t> cli_ids(const std::vector<std::string>& options) {
    const auto family = test_family(true);
    std::vector<std::string> words{"strix-test", "checkpoint", "0", "1", "--generate", "64"};
    words.insert(words.end(), options.begin(), options.end());
    std::vector<char*> argv;
    for (auto& w : words) argv.push_back(w.data());
    std::ostringstream out, error;
    if (sv::run_serial_generation_cli(static_cast<int>(argv.size()), argv.data(), family, out, error) != 0)
        throw std::runtime_error("CLI failed: " + error.str());
    std::istringstream in(out.str());
    std::string line;
    std::vector<std::uint32_t> ids;
    while (std::getline(in, line)) {  // the "generated I I I ..." line holds the decided id first, then the continuation
        std::istringstream fields(line);
        std::string word;
        fields >> word;
        if (word != "generated") continue;
        while (fields >> word) ids.push_back(static_cast<std::uint32_t>(std::stoul(word)));
    }
    return ids;
}
struct Writer final : sv::HttpResponseWriter {
    int status = 0;
    std::string body;
    void send(int s, std::string_view, std::string_view b,
              const std::vector<std::pair<std::string, std::string>>&) override { status = s; body = b; }
    bool begin_stream(int s, std::string_view,
                      const std::vector<std::pair<std::string, std::string>>&) override { status = s; return true; }
    bool write(std::string_view b) override { body += b; return true; }
    void end_stream() override {}
    bool client_gone() const override { return false; }
};
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void set_environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
sv::SerialGenerationOptions parse_cli(std::vector<std::string> words) {
    std::vector<char*> argv;
    for (auto& w : words) argv.push_back(w.data());
    return sv::parse_serial_generation_cli(static_cast<int>(argv.size()), argv.data(), test_family(false));
}
}  // namespace

int main() try {
    // The startup functions, not only the helper: the retired counter mode is refused by the shared startup check and
    // by the command-line parser before any request, and a good value passes.
    set_environment("STRIX_SAMPLING_REQUEST_IDENTITY", "counter");
    bool refused = false;
    try { strix::validate_sampling_startup_environment(); } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "startup check accepted the retired counter identity");
    refused = false;
    try {
        (void)parse_cli({"strix-test", "checkpoint", "0", "1"});
    } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "command-line parser accepted the retired counter identity");
    set_environment("STRIX_SAMPLING_REQUEST_IDENTITY", "single");
    strix::validate_sampling_startup_environment();

    using strix::SampledAcceptance;
    bool environment_ok = true;
    for (const auto& [value, expected] : std::vector<std::pair<const char*, SampledAcceptance>>{
        {"", SampledAcceptance::coupled}, {"coupled", SampledAcceptance::coupled}, {"rejection", SampledAcceptance::rejection}}) {
        set_environment("STRIX_MTP_SAMPLED_ACCEPT", value);
        if (strix::token_sampler_options_from_environment(SampledAcceptance::coupled).acceptance != expected) {
            std::cerr << "wrong acceptance with coupled default and STRIX_MTP_SAMPLED_ACCEPT=" << value << '\n';
            environment_ok = false;
        }
    }
    set_environment("STRIX_MTP_SAMPLED_ACCEPT", "invalid");
    refused = false;
    try { (void)strix::token_sampler_options_from_environment(SampledAcceptance::coupled); }
    catch (const std::invalid_argument&) { refused = true; }
    require(refused, "invalid explicit acceptance was admitted");
    set_environment("STRIX_MTP_SAMPLED_ACCEPT", "");
    set_environment("STRIX_MTP_SAMPLED_FORCE_REJECT", "0");
    refused = false;
    try { (void)parse_cli({"strix-test", "checkpoint", "0", "1", "--temperature", "1"}); }
    catch (const std::invalid_argument& e) { refused = std::string_view(e.what()).find("requires coupled acceptance") != std::string_view::npos; }
    if (!refused) std::cerr << "CLI parser admitted forced rejection with its rejection default\n";
    require(environment_ok && refused, "sampling environment regressions");
    set_environment("STRIX_MTP_SAMPLED_ACCEPT", "coupled");
    const auto resolved = parse_cli({"strix-test", "checkpoint", "0", "1", "--temperature", "1"});
    set_environment("STRIX_MTP_SAMPLED_FORCE_REJECT", "");
    set_environment("STRIX_MTP_SAMPLED_ACCEPT", "");
    strix::TokenSampler bound(resolved.sampling, resolved.sampler_options);
    require(bound.coupled() && bound.options().force_reject_row == 0U,
        "CLI execution did not retain the admitted coupled forced-rejection options");
    strix::TokenSampler plain(resolved.sampling, {});
    for (std::uint64_t pos = 2; pos < 66; ++pos) {
        const std::array draft{bound.draft(logits, {}, pos)};
        const auto window = bound.decide_rows(1, pos, draft, {}, [](std::uint32_t) { return std::span<const float>(logits); });
        require(window.size() == 1U && window[0].drafted && !window[0].accepted &&
            window[0].token == plain.decide(logits, pos, nullptr).token,
            "a coupled forced rejection must discard the draft while keeping the target draw");
    }
    require(bound.stats().forced > 0U, "drafted-row fixture did not exercise forced rejection");

    Engine engine;
    sv::GenerationQueue queue;
    sv::ApiConfig config;
    config.model_id = "host-test";
    sv::ApiService api(config, sv::ApiTokenizerBinding(std::make_shared<Tokenizer>()), queue, 256, true);
    std::jthread worker([&](std::stop_token stop) {
        const std::stop_callback shutdown(stop, [&] { queue.shutdown(); });
        queue.run(engine);
    });
    const auto post = [&](const std::string& fields) {
        sv::HttpRequest r;
        r.method = "POST"; r.path = r.target = "/v1/completions";
        r.body = std::string(R"({"prompt":[0,1],"max_tokens":64)") + fields + "}";
        Writer writer;
        api.handle(r, writer);
        if (writer.status != 200) throw std::runtime_error(writer.body);
    };
    // Each case: the request fields and the command-line options that mean the same levers.
    struct Case { const char* name; std::string fields; std::vector<std::string> cli; };
    const std::vector<Case> cases{
        {"omitted seed", R"(,"temperature":1)", {"--temperature", "1"}},
        {"explicit zero", R"(,"temperature":1,"seed":0)", {"--temperature", "1", "--seed", "0"}},
        {"null seed", R"(,"temperature":1,"seed":null)", {"--temperature", "1"}},
        {"explicit 37", R"(,"temperature":1,"seed":37)", {"--temperature", "1", "--seed", "37"}},
        {"filter only", R"(,"top_k":2)", {"--top-k", "2"}},
        {"filter only, seed 5", R"(,"top_p":0.5,"seed":5)", {"--top-p", "0.5", "--seed", "5"}},
        {"rounds to no filter", R"(,"top_p":0.999999999)", {"--top-p", "0.999999999"}},
        {"greedy", R"(,"temperature":0)", {"--temperature", "0"}},
        {"greedy with filters", R"(,"temperature":0,"top_k":2)", {"--temperature", "0", "--top-k", "2"}},
    };
    std::map<std::string, std::vector<std::uint32_t>> expected;
    for (const auto& c : cases) expected[c.name] = cli_ids(c.cli);
    require(expected["explicit 37"] != expected["omitted seed"], "seed 37 fixture equals seed 0");
    require(expected["rounds to no filter"] == std::vector<std::uint32_t>(64, 2U), "CLI top_p 0.999999999 must stay greedy");
    require(expected["greedy"] == expected["rounds to no filter"], "CLI greedy differs");
    require(expected["greedy"] == expected["greedy with filters"], "CLI filters changed greedy tokens");
    // Per case, alone: the request equals the command line of the same levers exactly.
    for (const auto& c : cases) {
        const auto before = engine.served.size();
        post(c.fields);
        const auto& served = std::get<2>(engine.served.at(before));
        if (served != expected[c.name]) {
            std::string detail;
            for (std::size_t i = 0; i < std::min<std::size_t>(12, served.size()); ++i) detail += std::to_string(served[i]);
            detail += " vs ";
            for (std::size_t i = 0; i < std::min<std::size_t>(12, expected[c.name].size()); ++i) detail += std::to_string(expected[c.name][i]);
            throw std::runtime_error(std::string("request differs from the command line: ") + c.name + " sizes " +
                std::to_string(served.size()) + "/" + std::to_string(expected[c.name].size()) + " " + detail);
        }
    }
    // Interleave: several client threads submit every case, twice, in a different rotation each while one worker serves
    // the queue. Each stream the command line produces must be served exactly as often as its cases were submitted.
    constexpr unsigned clients = 4;
    {
        std::vector<std::jthread> threads;
        for (unsigned client = 0; client < clients; ++client)
            threads.emplace_back([&, client] {
                for (std::size_t round = 0; round < 2; ++round)
                    for (std::size_t k = 0; k < cases.size(); ++k) post(cases[(k + client * 3U + round) % cases.size()].fields);
            });
    }
    std::map<std::vector<std::uint32_t>, std::size_t> want, got;
    for (const auto& c : cases) want[expected[c.name]] += clients * 2U;
    for (std::size_t i = cases.size(); i < engine.served.size(); ++i) ++got[std::get<2>(engine.served[i])];
    require(got == want, "interleaved requests differ from the command-line streams");
    const auto matched = engine.served.size() - cases.size();
    std::cout << "PASS API/CLI parity through the real CLI parser and sampler: " << cases.size()
              << " lever sets (seed 0/37, filter-only, top_p rounding to no filter, greedy); " << matched
              << " interleaved requests; counter identity refused at startup\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
}
