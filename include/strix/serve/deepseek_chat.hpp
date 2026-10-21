#pragma once

#include "strix/serve/chat_template.hpp"

#include <string>

// DeepSeek V4.1 chat encoding (the model ships no Jinja template; this is its reference encoder, encoding.py of the
// model release, for text conversations): <｜begin▁of▁sentence｜>, <｜System｜>/<｜User｜>/<｜Assistant｜> turns, chat
// mode ("</think>" after the assistant header) or thinking mode ("<think>", numeric reasoning effort, earlier
// reasoning dropped unless preserve_thinking or tools), tools in the system prompt, tool calls as DSML blocks and tool
// results merged into user turns as <tool_result> blocks.
namespace strix::serve {

// Request-level tools go into the first system message (a system message with empty content is added when the
// conversation has none). reasoning_effort: low | medium | high | xhigh | max | an integer 1..100 (default high).
[[nodiscard]] std::string render_deepseek_v41_chat(const ChatTemplateInput& input);

// The generated text of one assistant turn (without the end-of-sentence token): reasoning before </think> when
// thinking, then the content, then the DSML tool calls (arguments typed by their string="true|false" flag).
[[nodiscard]] AssistantOutput parse_deepseek_v41_output(const std::string& text, bool thinking, const Json& tools);

// The marker that starts the tool-call block in the content (streaming keeps everything from it for the final parse).
inline constexpr const char* deepseek_v41_tool_marker = "<\xEF\xBD\x9C" "DSML\xEF\xBD\x9C calls>";

}  // namespace strix::serve
