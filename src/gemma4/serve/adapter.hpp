#pragma once

#include "chat.hpp"
#include "tokenizer.hpp"
#include "strix/serve/engine.hpp"

namespace strix::gemma4::serve {

// Family-side seam for a future GenerationEngine implementation. GenerationJob
// and GenerationQueue remain shared. No executor or HTTP endpoint is installed.
struct PreparedPrompt {
    std::string rendered;
    std::vector<std::uint32_t> ids, stop_ids;
    bool starts_in_reasoning = false;
    [[nodiscard]] strix::serve::GenerationJob job(std::uint32_t max_tokens) const;
};
class Adapter {
public:
    explicit Adapter(std::shared_ptr<const Tokenizer> tokenizer);
    [[nodiscard]] PreparedPrompt prepare(const ChatInput& input) const;
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return *tokenizer_; }
private:
    std::shared_ptr<const Tokenizer> tokenizer_;
};

// The shared API uses {name, arguments} internally and creates wire IDs itself.
[[nodiscard]] strix::serve::AssistantOutput shared_output(const Output& output);

// Integration must provide a polymorphic tokenizer seam to ApiService (which
// currently accepts concrete BpeTokenizer), and request-local parser state for
// reasoning channels. An engine implements GenerationEngine::generate and
// consumes PreparedPrompt::job through the existing queue. Budget continuation
// is an explicit second job; advertise no engine budget capability until wired.

} // namespace strix::gemma4::serve
