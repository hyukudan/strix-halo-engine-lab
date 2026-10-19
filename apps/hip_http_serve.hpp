#pragma once
// --http-serve: the OpenAI/Anthropic-compatible HTTP front end (strix_serve) over the MTP bridge's
// request runner. The bridge thread executes the generation queue; HTTP connections run on their
// own threads and only exchange tokens with it through tickets (docs/serving.md).
#include "hip_decode_options.hpp"
#include "hip_mtp_decode_bridge.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::decode_cli {

// One engine for the queue: a request is one bridge run. A failure inside the runner leaves the
// target in an unknown state, so it fails every later request and stops the server.
class MtpBridgeEngine final : public serve::GenerationEngine {
public:
    MtpBridgeEngine(const MtpServeRunner& run, std::uint64_t context_tokens) : run_(run), context_(context_tokens) {}
    serve::GenerationStats generate(serve::GenerationJob& job) override {
        if (failed_.load()) throw std::runtime_error("the engine failed during an earlier request; restart the server");
        if (job.prompt.empty() || job.prompt.size() >= context_) throw std::invalid_argument("prompt is empty or fills the context");
        MtpServeRequest request;
        request.prompt = job.prompt;
        request.max_tokens = job.max_tokens;
        request.stop_tokens = job.stop_tokens;
        request.on_tokens = job.on_tokens;
        MtpServeResult result;
        try {
            result = run_(request);
        } catch (...) {
            failed_.store(true);
            throw;
        }
        serve::GenerationStats stats;
        stats.prompt_tokens = job.prompt.size();
        stats.cached_tokens = result.restored_tokens;
        stats.generated_tokens = result.generated_tokens;
        stats.cycles = result.cycles;
        stats.proposed = result.proposed;
        stats.accepted = result.accepted;
        stats.prefill_ms = result.prompt_ms;
        stats.decode_ms = result.decode_ms;
        stats.finish = result.finish;
        stats.sampling = "greedy";
        return stats;
    }
    [[nodiscard]] std::uint64_t context_tokens() const override { return context_; }
    [[nodiscard]] bool failed() const noexcept { return failed_.load(); }
private:
    const MtpServeRunner& run_;
    std::uint64_t context_;
    std::atomic<bool> failed_{false};
};

inline std::atomic<bool>& http_serve_stop_requested() {
    static std::atomic<bool> requested{false};
    return requested;
}
extern "C" inline void http_serve_signal(int) { http_serve_stop_requested().store(true); }

inline std::string read_http_api_key(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read --http-api-key-file");
    std::string key((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' ')) key.pop_back();
    if (key.empty()) throw std::invalid_argument("--http-api-key-file is empty");
    return key;
}

// Serves until SIGINT/SIGTERM or an engine failure (rethrown after the server stops).
inline MtpDecodeServe make_http_serve(const Options& options, std::string engine_description) {
    return [&options, engine_description = std::move(engine_description)](const MtpServeRunner& run) {
        const auto tokenizer = serve::BpeTokenizer::load_gguf(options.model_path);
        serve::ApiConfig config;
        config.model_id = options.http_model_id;
        config.default_max_tokens = options.http_default_max_tokens;
        config.api_key = read_http_api_key(options.http_api_key_file);
        config.sampling_policy = options.http_sampling;
        config.engine_description = engine_description;
        config.stop_tokens.push_back(tokenizer->eos_token());
        if (const auto end = tokenizer->find("<|endoftext|>"); end != UINT32_MAX) config.stop_tokens.push_back(end);
        if (options.eos_token) config.stop_tokens.push_back(*options.eos_token);
        std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
        config.stop_tokens.erase(std::unique(config.stop_tokens.begin(), config.stop_tokens.end()), config.stop_tokens.end());

        MtpBridgeEngine engine(run, options.context_tokens);
        serve::GenerationQueue queue(options.http_max_queue);
        serve::ApiService api(config, tokenizer, queue, options.context_tokens, false);
        serve::HttpServerConfig http;
        http.bind_address = options.http_bind;
        http.port = static_cast<std::uint16_t>(options.http_port);
        http.max_connections = options.http_max_connections;
        serve::HttpServer server(http, [&api](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
            api.handle(request, response);
        });
        server.listen();
        api.set_state("ready");
        std::cerr << "strix serve: listening on " << options.http_bind << ':' << server.port() << " model " << config.model_id << std::endl;

        http_serve_stop_requested().store(false);
        const auto previous_int = std::signal(SIGINT, http_serve_signal);
        const auto previous_term = std::signal(SIGTERM, http_serve_signal);
        std::atomic<bool> finished{false};
        std::thread http_thread([&server] { server.serve(); });
        std::thread watcher([&] {
            while (!finished.load() && !http_serve_stop_requested().load() && !engine.failed())
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (engine.failed()) api.set_state("failed");
            server.stop();
            queue.shutdown();
        });
        queue.run(engine);
        finished.store(true);
        watcher.join();
        http_thread.join();
        std::signal(SIGINT, previous_int);
        std::signal(SIGTERM, previous_term);
        if (engine.failed()) throw std::runtime_error("strix serve stopped after an engine failure");
        std::cerr << "strix serve: stopped" << std::endl;
    };
}

// --http-serve --http-batch N: the same front end over the batched runner. The bridge thread admits
// queued requests into free slots between decode cycles (each prompt runs as a single request's would)
// and decodes every active request together; a request leaves when it ends, without stopping the others.
// With --http-sampling sample, a request with temperature > 0 samples (sampling-v1) with its own policy
// and seed.
inline MtpDecodeBatchServe make_http_batch_serve(const Options& options, std::string engine_description) {
    return [&options, engine_description = std::move(engine_description)](MtpBatchRunner& runner) {
        const auto tokenizer = serve::BpeTokenizer::load_gguf(options.model_path);
        serve::ApiConfig config;
        config.model_id = options.http_model_id;
        config.default_max_tokens = options.http_default_max_tokens;
        config.api_key = read_http_api_key(options.http_api_key_file);
        config.sampling_policy = options.http_sampling;
        config.engine_description = engine_description;
        config.stop_tokens.push_back(tokenizer->eos_token());
        if (const auto end = tokenizer->find("<|endoftext|>"); end != UINT32_MAX) config.stop_tokens.push_back(end);
        if (options.eos_token) config.stop_tokens.push_back(*options.eos_token);
        std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
        config.stop_tokens.erase(std::unique(config.stop_tokens.begin(), config.stop_tokens.end()), config.stop_tokens.end());
        const bool samples = options.http_sampling == "sample";

        serve::GenerationQueue queue(options.http_max_queue);
        serve::ApiService api(config, tokenizer, queue, options.context_tokens, samples);
        serve::HttpServerConfig http;
        http.bind_address = options.http_bind;
        http.port = static_cast<std::uint16_t>(options.http_port);
        http.max_connections = options.http_max_connections;
        serve::HttpServer server(http, [&api](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
            api.handle(request, response);
        });
        server.listen();
        api.set_state("ready");
        std::cerr << "strix serve: listening on " << options.http_bind << ':' << server.port() << " model " << config.model_id
                  << " batch " << runner.slots() << std::endl;

        http_serve_stop_requested().store(false);
        const auto previous_int = std::signal(SIGINT, http_serve_signal);
        const auto previous_term = std::signal(SIGTERM, http_serve_signal);
        std::atomic<bool> finished{false}, failed{false};
        std::thread http_thread([&server] { server.serve(); });
        std::thread watcher([&] {
            while (!finished.load() && !http_serve_stop_requested().load() && !failed.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (failed.load()) api.set_state("failed");
            server.stop();
            queue.shutdown();
        });
        // Jobs in the slots, finished by the runner (on_finish) or, after an engine failure, here.
        std::vector<std::shared_ptr<serve::TakenJob>> live;
        const auto release = [&](const std::shared_ptr<serve::TakenJob>& taken) {
            live.erase(std::remove(live.begin(), live.end(), taken), live.end());
        };
        std::random_device entropy;
        const auto admit = [&](serve::TakenJob taken_job) {
            auto taken = std::make_shared<serve::TakenJob>(std::move(taken_job));
            auto& job = taken->job;
            if (job.prompt.empty() || job.prompt.size() >= options.context_tokens) {
                queue.finish(*taken, {}, "prompt is empty or fills the context");
                return;
            }
            MtpBatchRequest request;
            request.prompt = job.prompt;
            request.max_tokens = job.max_tokens;
            request.stop_tokens = job.stop_tokens;
            request.on_tokens = job.on_tokens;
            std::string sampling = "greedy";
            if (samples && job.sampling.temperature > 0.0) {
                TokenSamplingPolicy policy;
                policy.temperature = static_cast<float>(job.sampling.temperature);
                policy.top_k = job.sampling.top_k;
                policy.top_p = static_cast<float>(job.sampling.top_p);
                policy.min_p = static_cast<float>(job.sampling.min_p);
                policy.seed = job.sampling.seeded ? job.sampling.seed
                                                  : (static_cast<std::uint64_t>(entropy()) << 32U) | entropy();
                validate_token_sampling_policy(policy);
                request.sampler = std::make_shared<TokenSampler>(policy);
                sampling = "sampling-v1";
            }
            const auto prompt_tokens = job.prompt.size();
            request.on_finish = [&queue, &release, taken, prompt_tokens, sampling](const MtpServeResult& result) {
                serve::GenerationStats stats;
                stats.prompt_tokens = prompt_tokens;
                stats.cached_tokens = result.restored_tokens;
                stats.generated_tokens = result.generated_tokens;
                stats.cycles = result.cycles;
                stats.proposed = result.proposed;
                stats.accepted = result.accepted;
                stats.prefill_ms = result.prompt_ms;
                stats.decode_ms = result.decode_ms;
                stats.finish = result.finish;
                stats.sampling = sampling;
                queue.finish(*taken, std::move(stats));
                release(taken);
            };
            live.push_back(taken);
            runner.admit(std::move(request));
        };
        std::string failure;
        try {
            for (;;) {
                while (runner.has_free_slot()) {
                    auto job = queue.take(runner.active() == 0U);
                    if (!job) break;
                    admit(std::move(*job));
                }
                if (!runner.active()) {
                    if (queue.stopping()) break;
                    continue;
                }
                runner.step();
            }
        } catch (const std::exception& e) {
            failure = e.what();
        } catch (...) {
            failure = "engine failure";
        }
        if (!failure.empty()) {
            // The target state of every slot is unknown: fail the requests in flight and stop.
            failed.store(true);
            for (auto& taken : std::vector<std::shared_ptr<serve::TakenJob>>(live)) queue.finish(*taken, {}, failure);
            live.clear();
        }
        finished.store(true);
        watcher.join();
        http_thread.join();
        std::signal(SIGINT, previous_int);
        std::signal(SIGTERM, previous_term);
        if (!failure.empty()) throw std::runtime_error("strix serve stopped after an engine failure: " + failure);
        std::cerr << "strix serve: stopped" << std::endl;
    };
}

}  // namespace strix::decode_cli
