#pragma once

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strix::serve {

struct ApiConfig {
    std::string model_id = "qwen3.8-flash-next";
    std::uint32_t default_max_tokens = 4096;
    bool default_thinking = true;            // the template's default (enable_thinking unset)
    std::string api_key;                     // empty: no authentication
    // Requests with temperature > 0 on an engine without sampling: "greedy" runs them greedy and
    // says so in the response, "reject" answers 400.
    std::string sampling_policy = "greedy";
    std::vector<std::uint32_t> stop_tokens;  // generation ends at any of these (not emitted)
    std::size_t recent_requests = 50;
    std::string engine_description;          // free text for /api/status
};

// OpenAI-compatible (/v1/chat/completions, /v1/completions, /v1/models) and Anthropic-compatible
// (/v1/messages) endpoints over one generation queue, plus /health, /metrics (Prometheus text),
// /api/status (JSON) and /tokenize, /detokenize.
class ApiService {
public:
    ApiService(ApiConfig config, std::shared_ptr<const BpeTokenizer> tokenizer, GenerationQueue& queue,
        std::uint64_t context_tokens, bool engine_samples);
    ~ApiService();
    void handle(const HttpRequest& request, HttpResponseWriter& response);
    // Prompt tokens of a chat request body exactly as the chat route builds them (for tests and tools).
    [[nodiscard]] std::vector<std::uint32_t> chat_prompt(const Json& body) const;
    void set_state(std::string state);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::serve
