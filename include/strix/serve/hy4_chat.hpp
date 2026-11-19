#pragma once

#include "strix/serve/chat_template.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strix::serve {

// The template's data: its HYTK suffix of every special-token text and the reasoning efforts it accepts. The official
// release spells ":opensource" and accepts max | high | no_think; a pack may carry its own template (the Q4_K_M GGUF:
// ":6124c78e", ids unchanged, and also "low"). Read it from the pack's tokenizer.chat_template with hy4_chat_dialect.
struct Hy4ChatDialect {
    std::string suffix = ":opensource";
    std::vector<std::string> efforts = {"max", "high", "no_think"};
};
// The dialect of a Hy4 chat_template.jinja text (its HYTK assignment and its reasoning_effort list); throws
// std::invalid_argument for a template without them (an unknown template version is refused, never guessed).
[[nodiscard]] Hy4ChatDialect hy4_chat_dialect(std::string_view chat_template);

// Hy4 chat template (the model's chat_template.jinja, rendered natively for text content). Every turn is
// <｜hy_start:opensource｜>ROLE<｜hy_middle:opensource｜>...<｜hy_end:opensource｜>. The leading system turn (synthesised when
// absent) carries the tools prompt and ends with <｜reasoning_mode:opensource｜>reasoning_effort:EFFORT. Assistant turns
// write <think:opensource>reasoning</think:opensource> (kept after the last user message, or always with preserved
// thinking) and their tool calls as <tool_calls:opensource><tool_call:opensource>name<arg_key:...>k</arg_key:...>
// <arg_value:...>v</arg_value:...>...</tool_call:opensource></tool_calls:opensource>. Consecutive tool messages are one
// tool turn of <tool_response:opensource> entries, in the preceding assistant's call order when every id matches once.
//
// Inputs: reasoning_effort one of the dialect's efforts (unset = high; enable_thinking=false = no_think; anything else is
// the template's error); preserve_thinking = the template's preserved_thinking (unset = true with tools, false without).
// The generation prompt opens <think:opensource>, closed at once for no_think. Image and video parts are rejected (no
// vision input); content the template would print as a Python repr is rejected. A tool call's `arguments` may be an
// object or a JSON string (parsed here, as the template requires a mapping).
// Shown with the official spellings; every text uses the dialect's suffix.
[[nodiscard]] std::string render_hy4_chat(const ChatTemplateInput& input, const Hy4ChatDialect& dialect = {});

// The generated text of one assistant turn (without the stop token): with `thinking`, reasoning before
// </think:opensource>; then the content up to the tool-call block, then the calls (values typed by the request tools).
[[nodiscard]] AssistantOutput parse_hy4_output(const std::string& text, bool thinking, const Json& tools,
                                               const Hy4ChatDialect& dialect = {});

// The marker that starts a tool call in the content (streaming keeps everything from it for the final parse);
// it prefixes both <tool_calls...> and <tool_call...> in every dialect.
inline constexpr const char* hy4_tool_marker = "<tool_call";

// generation_config.json eos_token_id: <｜hy_end:opensource｜>.
inline constexpr std::array<std::uint32_t, 1> hy4_stop_tokens = {120025U};

}  // namespace strix::serve
