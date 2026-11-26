#pragma once

#include "strix/serve/chat_template.hpp"

#include <cstdint>
#include <memory>

namespace strix::serve {

class BpeTokenizer;
struct ApiConfig;

struct ErnieChatOptions {
    Json system_settings, agent_storage, examples, options;
};

struct ErniePrompt {
    struct Span { std::string text; bool control; bool assistant; };
    std::vector<Span> spans;
    bool thinking = true;

    [[nodiscard]] std::string text() const;
    [[nodiscard]] std::vector<std::uint32_t> encode(const BpeTokenizer& tokenizer) const;
};

[[nodiscard]] ErniePrompt render_ernie_chat(const ChatTemplateInput& input, const ErnieChatOptions& options = {});
[[nodiscard]] std::shared_ptr<const BpeTokenizer> load_ernie_tokenizer(const std::string& tokenizer_json);

// Bind the host chat protocol to an engine's ApiConfig. The engine supplies generation and the
// same tokenizer wrapped in BpeApiTokenizer. The API owns SSE, IDs, queue admission and stops.
void bind_ernie_chat(ApiConfig& config, std::shared_ptr<const BpeTokenizer> tokenizer);

}  // namespace strix::serve
