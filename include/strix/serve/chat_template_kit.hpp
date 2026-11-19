#pragma once

#include "strix/serve/chat_template.hpp"
#include "strix/serve/json.hpp"

#include <string>
#include <string_view>

// Pieces shared by the natively rendered chat templates (the Jinja semantics HF transformers applies) and the
// key/value tool-call grammar that several templates use with their own spellings:
//   CALL_START name KEY_START k KEY_END VALUE_START v VALUE_END ... CALL_END
// GLM-5 writes it with <tool_call>/<arg_key>/<arg_value>, another dialect with the ":opensource" forms inside an outer
// <tool_calls:opensource> block.
namespace strix::serve::chat_kit {

[[noreturn]] void raise(const std::string& message);  // throws ChatTemplateError

// Python truthiness of a value (absent = Jinja undefined = false).
[[nodiscard]] bool truthy(const Json* v);
// {{ value }}: Python str() of a scalar ("None", "True", the number's source text); undefined prints nothing.
// An object or array would print as a Python repr: rejected with `what` in the message.
[[nodiscard]] std::string py_str(const Json* v, const char* what);
// A tool as the template sees it: its `function` member when it has one.
[[nodiscard]] const Json& tool_spec(const Json& tool);
// The tool's members in order without defer_loading and strict, as '"k": ' + tojson(v, ensure_ascii=False) joined by ", ".
[[nodiscard]] std::string tool_to_json(const Json& tool);
// A tool call's arguments as an object: an object, or a JSON string holding one (empty text = {}).
[[nodiscard]] Json call_arguments(const Json& call, const std::string& name);

// The spellings of one template's key/value tool-call grammar.
struct KvToolGrammar {
    std::string_view think_end;    // ends the reasoning block
    std::string_view calls_start;  // optional outer block opener ("" = none); content also ends at it
    std::string_view call_start, call_end, key_start, key_end, value_start, value_end;
};

// The generated text of one assistant turn. reasoning_open: generation starts inside the reasoning block (the text
// before think_end is the reasoning; without think_end all of it is). Otherwise all text is content. The content
// ends at the first calls_start / call_start; each call's values are typed by its request tool's parameter schema
// (JSON-parsed when the schema type is not string and the value parses, else kept as text).
[[nodiscard]] AssistantOutput parse_kv_tool_output(const std::string& text, bool reasoning_open, const KvToolGrammar& grammar,
                                                   const Json& tools);

}  // namespace strix::serve::chat_kit
