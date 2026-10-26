#pragma once

#include "strix/serve/json.hpp"

#include <optional>
#include <string>

namespace strix::minimax::serve {

// Template variables of MiniMax M3.1's chat_template.jinja besides `messages` / `tools`.
// An unset optional is an undefined Jinja variable.
struct MinimaxChatOptions {
    bool add_generation_prompt = true;
    std::optional<std::string> thinking_mode;      // "enabled" | "disabled" | "adaptive"
    std::optional<std::string> template_model_id;  // "\n\n<model_id>...</model_id>" in the system turn
    std::optional<std::string> reasoning_effort;   // "\n\n<effort>...</effort>" (wins over `effort`)
    std::optional<std::string> effort;
    std::optional<std::string> model_identity;     // developer turn text when no system/developer message
};

class MinimaxChatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Renders `messages` (an OpenAI-style JSON array) exactly like chat_template.jinja run through
// transformers' apply_chat_template(tokenize=False). The output is meant for
// MinimaxTokenizer::encode(text, parse_special=true).
//
// Supported, as in the template: roles root (first message: high-priority system turn; may be followed by
// one system/developer message), system / developer (first: developer turn; later: mid-conversation
// developer turns), user, assistant (content, reasoning_content, think-tag splitting of the content,
// tool_calls with {function:{name, arguments}} or {name, arguments}), tool (consecutive tool messages share
// one turn); content as a string, null, or a list of parts ({type:text,text}, {type:image|video|frame},
// plain strings); system-message current_date / current_location; `tools` ({type:function,function:{...}})
// rendered with Python json.dumps(ensure_ascii=False) formatting; tool-call arguments rendered with the
// template's to_xml macro (None values skipped, booleans as true/false, numbers in Python str() form).
// Messages with other roles are skipped, as in the template.
//
// Deviations (throw MinimaxChatError instead of producing Python reprs): values the template would print
// with Python's repr (a dict/list as message content, a dict tool-result part without output/text).
// Extension: tool-call `arguments` given as a JSON string (OpenAI wire format) is parsed into an object first
// (the Jinja template itself would fail on a string).
[[nodiscard]] std::string render_minimax_chat(const strix::serve::Json& messages, const strix::serve::Json* tools = nullptr,
    const MinimaxChatOptions& options = {});

// Python json.dumps(value, ensure_ascii=False) (", " / ": " separators, key order kept, numbers in
// Python form: 1.0 stays 1.0, 1e5 becomes 100000.0, integers stay exact).
[[nodiscard]] std::string python_json_dumps(const strix::serve::Json& value);
// Python str() of a JSON scalar as Jinja prints it ({{ value }}): None/True/False, ints, float repr, strings raw.
[[nodiscard]] std::string python_str(const strix::serve::Json& value);

}  // namespace strix::minimax::serve
