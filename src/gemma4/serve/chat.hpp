#pragma once

#include "strix/serve/chat_template.hpp"
#include "parse.hpp"

namespace strix::gemma4::serve {

using strix::serve::Json;
struct ChatInput {
    Json messages = Json::array();
    Json tools;
    bool add_generation_prompt = true;
    bool enable_thinking = false; // official asset's default
    bool preserve_thinking = false;
    bool conversational_mode = false;
};

// Text-only native implementation of the official Nemotron Gemma 4 template.
// Unknown kwargs, non-text parts and unsupported tool policies throw ChatTemplateError.
[[nodiscard]] ChatInput chat_input(Json messages, const Json& kwargs = Json::object());
[[nodiscard]] std::string render_chat(const ChatInput& input);
// Gemma argument syntax; keys are unquoted in calls/results, quoted in schema enums.
[[nodiscard]] std::string format_argument(const Json& value, bool quote_keys = true);

struct BudgetContinuation {
    std::string prompt;
    std::uint32_t remaining_tokens = 0;
};
// Runtime two-step budget contract from the model card: first run at most budget
// output tokens, then resume the partial assistant with a closed reasoning channel.
// Call only on length cutoff, with no unfinished tool frame. Natural stops need no continuation.
[[nodiscard]] BudgetContinuation continue_after_budget(const ChatInput& original,
    const Output& partial, std::uint32_t used_tokens,
    std::uint32_t thinking_budget, std::uint32_t max_tokens);

} // namespace strix::gemma4::serve
