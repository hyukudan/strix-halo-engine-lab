// Real API parser + queue + shared sampler, entirely in process (no sockets/model).
#include "strix/serve/api.hpp"
#include "strix/serve/job_sampling.hpp"

#include <array>
#include <iostream>
#include <thread>

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
        seed = job.sampling.seed;
        auto sampler = sv::make_job_sampler(job.sampling);
        ids = draw(*sampler, job.prompt.size(), job.max_tokens);
        (void)job.on_tokens(ids);
        sv::GenerationStats stats;
        stats.prompt_tokens = job.prompt.size(); stats.generated_tokens = ids.size();
        stats.sampling = "sampling-v1";
        return stats;
    }
    std::uint64_t seed = UINT64_MAX;
    std::vector<std::uint32_t> ids;
};
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
}

int main() try {
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
        return engine.ids; // API completion synchronizes with the queue worker.
    };
    const auto request = [&](const char* seed) { return post(std::string(R"(,"temperature":1)") + seed); };
    strix::TokenSamplingPolicy policy;
    policy.temperature = 1.0F;
    strix::TokenSampler cli(policy);
    const auto expected = draw(cli, 2, 64);
    require(request("") == expected && engine.seed == policy.seed, "omitted HTTP seed differs from CLI");
    require(request(",\"seed\":37") != expected, "explicit seed did not change fixture tokens");
    for (unsigned repeat = 0; repeat < 4; ++repeat)
        require(request("") == expected, "intervening request changed omitted-seed tokens");
    require(request(",\"seed\":0") == expected, "explicit zero differs from omitted seed");
    require(request(",\"seed\":null") == expected, "null seed differs from omitted seed");
    // A filter without a temperature: the CLI resolves the omitted temperature through the shared rule (1 with a
    // filter); the HTTP request must draw the same 64 IDs, and no filter stays greedy on both.
    strix::TokenSamplingPolicy filtered;
    filtered.top_k = 2;
    filtered.temperature = strix::omitted_token_sampling_temperature(filtered.top_k, filtered.top_p, filtered.min_p);
    require(filtered.temperature == 1.0F, "a filter without temperature resolves to temperature 1");
    strix::TokenSampler cli_filtered(filtered);
    require(post(R"(,"top_k":2)") == draw(cli_filtered, 2, 64), "filter-only HTTP request differs from the CLI");
    require(strix::omitted_token_sampling_temperature(0, 1.0, 0.0) == 0.0F, "no filter resolves to greedy");
    std::cout << "PASS API/CLI seed parity: omitted/null/zero, repeat and interleaved explicit seed; 64 complete IDs\n";
    std::cout << "PASS API/CLI omitted temperature: filter-only request samples as the CLI; 64 complete IDs\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
}
