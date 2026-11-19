#include "strix/serve/hy4_chat.hpp"

#include "strix/serve/chat_template_kit.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Native rendering of the Hy4 chat_template.jinja (its macros render_content, render_tools_prompt,
// render_tool_response, merge_tool_responses and render_tool_responses) as HF transformers renders it.
// The shared pieces (Python str(), truthiness, tool_to_json, call arguments, the key/value tool-call parser)
// come from chat_template_kit; only the spellings and the turn structure are Hy4's.
namespace strix::serve {
namespace {

using chat_kit::py_str;
using chat_kit::raise;
using chat_kit::truthy;

// The template's special-token texts for one HYTK suffix.
struct Spell {
    explicit Spell(const std::string& suffix)
        : start(kBar + "hy_start" + suffix + kBarClose), middle(kBar + "hy_middle" + suffix + kBarClose),
          end(kBar + "hy_end" + suffix + kBarClose), think_begin("<think" + suffix + ">"),
          think_end("</think" + suffix + ">"), calls_begin("<tool_calls" + suffix + ">"), calls_end("</tool_calls" + suffix + ">"),
          call_begin("<tool_call" + suffix + ">"), call_end("</tool_call" + suffix + ">"), key_begin("<arg_key" + suffix + ">"),
          key_end("</arg_key" + suffix + ">"), value_begin("<arg_value" + suffix + ">"), value_end("</arg_value" + suffix + ">"),
          response_begin("<tool_response" + suffix + ">"), response_end("</tool_response" + suffix + ">"),
          reasoning_mode(kBar + "reasoning_mode" + suffix + kBarClose) {}
    // "<" + U+FF5C FULLWIDTH VERTICAL LINE, and U+FF5C + ">"
    static inline const std::string kBar = "<\xef\xbd\x9c", kBarClose = "\xef\xbd\x9c>";
    std::string start, middle, end, think_begin, think_end, calls_begin, calls_end, call_begin, call_end, key_begin, key_end,
        value_begin, value_end, response_begin, response_end, reasoning_mode;
};

bool has_type(const Json& item, std::string_view type) {
    const auto* t = item.find("type");
    return t && t->is_string() && t->as_string() == type;
}
bool is_media(const Json& item) {
    for (const char* key : {"image", "image_url", "video", "video_url"})
        if (item.find(key) || has_type(item, key)) return true;
    return false;
}

// render_content: a string; the text parts and strings of a list; '' for none; str() of a scalar.
std::string render_content(const Json* content) {
    if (!content || content->is_null()) return {};
    if (content->is_string()) return content->as_string();
    if (!content->is_array()) return py_str(content, "content");
    std::string out;
    for (const auto& item : content->as_array()) {
        if (item.is_object() && has_type(item, "text")) out += py_str(item.find("text"), "text part");
        else if (item.is_object() && is_media(item)) raise("Image and video content is not supported (no vision input).");
        else out += py_str(&item, "content part");
    }
    return out;
}

std::string role_of(const Json& message) {
    const auto* role = message.find("role");
    return role && role->is_string() ? role->as_string() : std::string{};
}

std::string tools_prompt(const Json& tools, const Spell& k) {
    std::string out = "# Tools\n\nYou may call one or more functions to assist with the user query.";
    out += "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
    bool first = true;
    for (const auto& tool : tools.as_array()) {
        const auto& t = chat_kit::tool_spec(tool);
        if (t.is_object() && truthy(t.find("defer_loading"))) continue;
        if (!first) out += "\n";
        first = false;
        out += chat_kit::tool_to_json(t);
    }
    out += "\n</tools>\n\nFor function call returns, you should first print " + k.calls_begin;
    out += "\nFor each function call, you should return object like:\n" + k.call_begin + "{function-name}";
    out += k.key_begin + "{arg-key-1}" + k.key_end + k.value_begin + "{arg-value-1}" + k.value_end;
    out += k.key_begin + "{arg-key-2}" + k.key_end + k.value_begin + "{arg-value-2}" + k.value_end + "..." + k.call_end;
    return out + "\nAt the end of function call returns, you should print " + k.calls_end;
}

// render_tool_response: adjacent tool_search_output parts share one <tools> block.
std::string tool_response(const Json& message, const Spell& k) {
    const auto* content = message.find("content");
    std::string out = k.response_begin;
    if (content && content->is_array()) {
        bool open = false;
        for (const auto& item : content->as_array()) {
            const bool search = item.is_object() && has_type(item, "tool_search_output");
            if (search && !open) { out += "<tools>\n"; open = true; }
            else if (!search && open) { out += "</tools>"; open = false; }
            if (search) {
                const auto* listed = item.find("tools");
                if (listed && (listed->is_string() || listed->is_object())) raise("tool_search_output tools must be a list.");
                if (listed && listed->is_array())
                    for (const auto& tool : listed->as_array()) out += chat_kit::tool_to_json(chat_kit::tool_spec(tool)) + "\n";
            } else if (item.is_object() && item.find("output")) {
                out += render_content(item.find("output"));
            } else {
                Json single = Json::array();
                single.push(item);
                out += render_content(&single);
            }
        }
        if (open) out += "</tools>";
    } else {
        out += render_content(content);
    }
    return out + k.response_end;
}

// Jinja == between two values of the ids (strings in practice).
bool same_value(const Json& a, const Json& b) {
    if (a.kind() != b.kind()) return false;
    if (a.is_string()) return a.as_string() == b.as_string();
    if (a.is_number()) return a.as_double() == b.as_double();
    if (a.is_bool()) return a.as_bool() == b.as_bool();
    return a.is_null() || a.dump_spaced() == b.dump_spaced();
}

// render_tool_responses: the cluster in the call order of the preceding assistant turn when every id matches exactly
// one message (tool_call_id, else id); otherwise in message order.
std::string tool_responses(const std::vector<const Json*>& cluster, const std::vector<const Json*>& call_ids, const Spell& k) {
    std::vector<const Json*> ordered;
    bool matched = !call_ids.empty() && call_ids.size() == cluster.size();
    for (std::size_t c = 0; matched && c < call_ids.size(); ++c) {
        std::size_t hits = 0;
        const Json* hit = nullptr;
        for (const auto* message : cluster) {
            const auto* id = message->find("tool_call_id");
            if (!id) id = message->find("id");
            if (id && same_value(*id, *call_ids[c])) { ++hits; hit = message; }
        }
        if (hits == 1U) ordered.push_back(hit);
        else matched = false;
    }
    std::string out;
    for (const auto* message : matched ? ordered : cluster) out += tool_response(*message, k);
    return out;
}

// One entry of merge_tool_responses: a message, or a cluster of consecutive tool messages.
struct Entry {
    const Json* message = nullptr;
    std::vector<const Json*> tools;
    [[nodiscard]] std::string role() const { return message ? role_of(*message) : std::string("tool"); }
};

std::string assistant_turn(const Json& message, bool keep_reasoning, std::vector<const Json*>& call_ids, const Spell& k) {
    const auto body = render_content(message.find("content"));
    std::string reasoning;
    if (const auto* r = message.find("reasoning"); r && r->is_string() && !r->as_string().empty()) reasoning = r->as_string();
    else if (const auto* rc = message.find("reasoning_content"); rc && rc->is_string()) reasoning = rc->as_string();
    const auto content = k.think_begin + (keep_reasoning ? reasoning : std::string{}) + k.think_end + body;
    std::string out = k.start + "assistant" + k.middle;
    const auto* calls = message.find("tool_calls");
    if (!truthy(calls)) return out + content + k.end;
    if (!calls->is_array()) raise("tool_calls must be an array.");
    std::vector<const Json*> ids;
    bool complete = true;
    for (const auto& call : calls->as_array()) {
        const auto* id = call.find("id");
        if (!truthy(id)) id = call.find("tool_call_id");
        if (truthy(id)) ids.push_back(id);
        else complete = false;
    }
    call_ids = complete ? ids : std::vector<const Json*>{};
    out += content + k.calls_begin;
    for (const auto& call : calls->as_array()) {
        const auto* function = call.is_object() ? call.find("function") : nullptr;
        const Json& func = function ? *function : call;
        const auto name = py_str(func.find("name"), "function name");
        out += k.call_begin + name;
        const auto arguments = chat_kit::call_arguments(func, name);
        for (const auto& [key, value] : arguments.as_object())
            out += k.key_begin + key + k.key_end + k.value_begin + (value.is_string() ? value.as_string() : value.dump_spaced()) + k.value_end;
        out += k.call_end;
    }
    return out + k.calls_end + k.end;
}

}  // namespace

std::string render_hy4_chat(const ChatTemplateInput& input, const Hy4ChatDialect& dialect) {
    const Spell k(dialect.suffix);
    std::string effort = "high";
    if (input.reasoning_effort) {
        effort = *input.reasoning_effort;
        if (std::find(dialect.efforts.begin(), dialect.efforts.end(), effort) == dialect.efforts.end()) {
            std::string listed;
            for (auto e = dialect.efforts.rbegin(); e != dialect.efforts.rend(); ++e) listed += (listed.empty() ? "" : "/") + *e;
            raise("reasoning_effort error : " + effort + ", should be " + listed);
        }
    }
    if (input.enable_thinking && !*input.enable_thinking) effort = "no_think";
    const bool has_tools = truthy(&input.tools);
    if (has_tools && !input.tools.is_array()) raise("tools must be a list.");
    const bool preserved = input.preserve_thinking.value_or(has_tools);
    if (!input.messages.is_array()) raise("messages must be a list.");

    std::vector<Entry> entries;
    for (const auto& message : input.messages.as_array()) {
        if (!message.is_object()) raise("Every message must be an object.");
        if (role_of(message) == "tool") {
            if (entries.empty() || entries.back().message) entries.emplace_back();
            entries.back().tools.push_back(&message);
        } else {
            entries.push_back(Entry{&message, {}});
        }
    }
    std::ptrdiff_t last_user = -1;
    for (std::size_t i = 0; i < entries.size(); ++i)
        if (entries[i].role() == "user") last_user = static_cast<std::ptrdiff_t>(i);

    std::string out;
    const auto reasoning_mode = k.reasoning_mode + "reasoning_effort:" + effort;
    if (entries.empty() || entries.front().role() != "system")
        out += k.start + "system" + k.middle + (has_tools ? tools_prompt(input.tools, k) : std::string{}) + reasoning_mode + k.end;
    std::vector<const Json*> call_ids;
    bool last_is_assistant = false;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        const auto role = entry.role();
        if (!entry.message) {
            out += k.start + "tool" + k.middle + tool_responses(entry.tools, call_ids, k) + k.end;
        } else if (role == "system") {
            out += k.start + "system" + k.middle;
            const auto content = render_content(entry.message->find("content"));
            if (i == 0U) {
                if (has_tools) out += tools_prompt(input.tools, k) + (content.empty() ? "" : "\n\n");
                out += content + reasoning_mode;
            } else {
                out += content;
            }
            out += k.end;
        } else if (role == "user") {
            out += k.start + "user" + k.middle + render_content(entry.message->find("content")) + k.end;
        } else if (role == "assistant") {
            const bool keep = effort != "no_think" && (preserved || static_cast<std::ptrdiff_t>(i) > last_user);
            out += assistant_turn(*entry.message, keep, call_ids, k);
        } else {
            out += k.start + py_str(entry.message->find("role"), "role") + k.middle + render_content(entry.message->find("content")) + k.end;
        }
        if (role != "assistant" || !truthy(entry.message->find("tool_calls"))) call_ids.clear();
        last_is_assistant = i + 1U == entries.size() && role == "assistant";
    }
    if (input.add_generation_prompt && !last_is_assistant)
        out += k.start + "assistant" + k.middle + k.think_begin + (effort == "no_think" ? k.think_end : std::string{});
    return out;
}

AssistantOutput parse_hy4_output(const std::string& text, bool thinking, const Json& tools, const Hy4ChatDialect& dialect) {
    const Spell k(dialect.suffix);
    const chat_kit::KvToolGrammar grammar{k.think_end, k.calls_begin, k.call_begin, k.call_end,
                                          k.key_begin, k.key_end, k.value_begin, k.value_end};
    return chat_kit::parse_kv_tool_output(text, thinking, grammar, tools);
}

Hy4ChatDialect hy4_chat_dialect(std::string_view chat_template) {
    // {%- set HYTK = 'SUFFIX' %} and {%- elif reasoning_effort not in ['a', 'b', ...] %}
    constexpr char kQuote = '\x27';
    const std::string_view suffix_key = "set HYTK = '", efforts_key = "reasoning_effort not in [";
    const auto at = chat_template.find(suffix_key);
    const auto list = chat_template.find(efforts_key);
    if (at == std::string_view::npos || list == std::string_view::npos)
        throw std::invalid_argument("hy4 chat template: no HYTK suffix or reasoning_effort list (unknown template version)");
    const auto begin = at + suffix_key.size();
    const auto close = chat_template.find(kQuote, begin);
    const auto list_end = chat_template.find(']', list);
    if (close == std::string_view::npos || list_end == std::string_view::npos)
        throw std::invalid_argument("hy4 chat template: unterminated HYTK suffix or reasoning_effort list");
    Hy4ChatDialect out;
    out.suffix = std::string(chat_template.substr(begin, close - begin));
    out.efforts.clear();
    const auto items = chat_template.substr(list + efforts_key.size(), list_end - list - efforts_key.size());
    for (std::size_t p = items.find(kQuote); p != std::string_view::npos; p = items.find(kQuote, p)) {
        const auto q = items.find(kQuote, p + 1U);
        if (q == std::string_view::npos) throw std::invalid_argument("hy4 chat template: unterminated reasoning_effort item");
        out.efforts.emplace_back(items.substr(p + 1U, q - p - 1U));
        p = q + 1U;
    }
    if (out.suffix.empty() || out.efforts.empty()) throw std::invalid_argument("hy4 chat template: empty HYTK suffix or effort list");
    return out;
}

}  // namespace strix::serve
