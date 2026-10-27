#include "strix/serve/deepseek_chat.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <string_view>
#include <vector>

// Port of the DeepSeek V4.1 reference encoder (encoding.py: render_message, merge_tool_messages,
// sort_tool_results_by_call_order, _drop_thinking_messages, encode_arguments_to_dsml, parse_tool_calls) for text
// messages. Python's json.dumps(ensure_ascii=False) is Json::dump_spaced.
namespace strix::serve {
namespace {

const std::string kBos = "<\xEF\xBD\x9C" "begin\xE2\x96\x81" "of\xE2\x96\x81" "sentence\xEF\xBD\x9C>";
const std::string kThinkStart = "<think>", kThinkEnd = "</think>";
const std::string kDsml = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";
const std::string kUser = "<\xEF\xBD\x9CUser\xEF\xBD\x9C>", kAssistant = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";
const std::string kSystem = "<\xEF\xBD\x9CSystem\xEF\xBD\x9C>", kLatestReminder = "<\xEF\xBD\x9Clatest_reminder\xEF\xBD\x9C>";
const std::string kEos = "<\xEF\xBD\x9C" "end\xE2\x96\x81" "of\xE2\x96\x81" "sentence\xEF\xBD\x9C>";
const std::string kCallsBlock = " calls", kInvoke = " invoke", kParameter = " parameter";

[[noreturn]] void raise(const std::string& message) { throw ChatTemplateError(message); }

std::string str_or(const Json* v, const std::string& fallback = {}) { return v && v->is_string() ? v->as_string() : fallback; }

// One content block of a user turn after merging: text or a tool result (its text).
struct Block {
    bool tool_result = false;
    std::string text, tool_use_id;
};
struct Message {
    std::string role;
    std::string content;
    bool has_blocks = false;
    std::vector<Block> blocks;
    std::string reasoning;
    Json tools;       // array (openai format) or null
    Json tool_calls;  // array (openai format) or null
};

// Text of a content value: a string, or the concatenated text parts of a list (images are not supported).
std::string content_text(const Json* content, const char* role) {
    if (!content || content->is_null()) return {};
    if (content->is_string()) return content->as_string();
    if (!content->is_array()) raise(std::string("Unexpected content type in a ") + role + " message.");
    std::string out;
    for (const auto& part : content->as_array()) {
        const auto type = str_or(part.find("type"), "text");
        if (type == "image" || type == "image_url" || part.find("image_url") || part.find("image"))
            raise("Image input is not supported by this engine.");
        const auto* text = part.find("text");
        if (!text || !text->is_string()) raise("Unexpected item type in content.");
        out += text->as_string();
    }
    return out;
}

std::vector<Block> user_blocks(const Json* content) {
    if (content && content->is_array()) {
        std::vector<Block> out;
        for (const auto& part : content->as_array()) {
            const auto type = str_or(part.find("type"), "text");
            if (type == "image" || type == "image_url" || part.find("image_url") || part.find("image"))
                raise("Image input is not supported by this engine.");
            if (type != "text") raise("Unsupported content block type " + type + ".");
            out.push_back({false, str_or(part.find("text")), {}});
        }
        return out;
    }
    return {{false, content_text(content, "user"), {}}};
}

// (namespace, name) of a possibly qualified tool name.
std::pair<std::optional<std::string>, std::string> split_tool_name(const std::string& name, std::optional<std::string> ns = std::nullopt) {
    const auto sep = name.find("::");
    std::string bare = name;
    if (sep != std::string::npos) {
        const auto prefix = name.substr(0, sep);
        if (ns && *ns != prefix) raise("Conflicting tool namespaces: " + *ns + " != " + prefix);
        ns = prefix;
        bare = name.substr(sep + 2U);
    }
    if (bare.find("::") != std::string::npos) raise("Tool name must not contain '::': " + bare);
    return {ns, bare};
}

std::optional<std::string> namespace_name(const Json* ns) {
    if (!ns || ns->is_null()) return std::nullopt;
    if (ns->is_string()) return ns->as_string();
    if (ns->is_object()) return str_or(ns->find("name"));
    return std::nullopt;
}

// tools_from_openai_format: the function objects with namespace-qualified names (namespace description prepended).
std::vector<Json> tools_for_encoding(const Json& tools) {
    std::vector<Json> out;
    for (const auto& tool : tools.as_array()) {
        const auto* fn = tool.find("function");
        if (!fn || !fn->is_object()) raise("Every tool needs a function object.");
        Json function = Json::object();
        for (const auto& [k, v] : fn->as_object())
            if (k != "namespace") function.set(k, v);
        const Json* ns = tool.find("namespace");
        if (!ns || ns->is_null()) ns = fn->find("namespace");
        const auto [space, name] = split_tool_name(str_or(fn->find("name")), namespace_name(ns));
        function.set("name", Json(space ? *space + "::" + name : name));
        if (ns && ns->is_object()) {
            if (const auto* d = ns->find("description"); d && d->is_string() && !d->as_string().empty())
                function.set("description", Json(d->as_string() + "\n" + str_or(fn->find("description"))));
        }
        out.push_back(std::move(function));
    }
    return out;
}

std::string render_tools(const std::vector<Json>& tools) {
    std::string schemas;
    for (std::size_t i = 0; i < tools.size(); ++i) schemas += (i ? "\n" : "") + tools[i].dump_spaced();
    const std::string calls = kDsml + kCallsBlock, invoke = kDsml + kInvoke, param = kDsml + kParameter;
    return "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" + calls +
           ">\" block like the following:\n\n<" + calls + ">\n<" + invoke + " name=\"$TOOL_NAME\">\n<" + param +
           " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + param + ">\n...\n</" + invoke + ">\n<" + invoke +
           " name=\"$TOOL_NAME2\">\n...\n</" + invoke + ">\n</" + calls +
           ">\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, "
           "objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by " +
           kThinkStart + "), you MUST output your complete reasoning inside " + kThinkStart + "..." + kThinkEnd +
           " BEFORE any tool calls or final response.\n\nOtherwise, output directly after " + kThinkEnd +
           " with tool calls or final response.\n\n### Available Tool Schemas\n\n" + schemas +
           "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
}

// encode_arguments_to_dsml for one OpenAI tool call.
std::string call_to_dsml(const Json& call) {
    const auto* fn = call.find("function");
    if (!fn || !fn->is_object()) raise("Every tool call needs a function object.");
    const Json* ns = call.find("namespace");
    if (!ns || ns->is_null()) ns = fn->find("namespace");
    const auto [space, name] = split_tool_name(str_or(fn->find("name")), namespace_name(ns));
    Json arguments;
    if (const auto* a = fn->find("arguments")) arguments = *a;
    for (int k = 0; k < 2 && arguments.is_string(); ++k) {  // JSON strings, including double-encoded ones
        try {
            arguments = parse_json(arguments.as_string());
        } catch (const JsonError&) {
            break;
        }
    }
    if (!arguments.is_object()) {
        Json wrapped = Json::object();
        wrapped.set("arguments", fn->find("arguments") ? *fn->find("arguments") : Json());
        arguments = std::move(wrapped);
    }
    const std::string param = kDsml + kParameter, invoke = kDsml + kInvoke;
    std::string params;
    bool first = true;
    for (const auto& [k, v] : arguments.as_object()) {
        params += (first ? "" : "\n") + std::string("<") + param + " name=\"" + k + "\" string=\"" + (v.is_string() ? "true" : "false") + "\">" +
                  (v.is_string() ? v.as_string() : v.dump_spaced()) + "</" + param + ">";
        first = false;
    }
    return "<" + invoke + " name=\"" + (space ? *space + "::" + name : name) + "\">\n" + params + "\n</" + invoke + ">";
}

std::string user_text(const Message& m) {
    std::string out;
    for (std::size_t i = 0; i < m.blocks.size(); ++i) {
        if (i) out += "\n\n";
        out += m.blocks[i].tool_result ? "<tool_result>" + m.blocks[i].text + "</tool_result>" : m.blocks[i].text;
    }
    return out;
}

// find_last_user_index: user turns (and mid-conversation system messages) that are not only tool results.
int last_user_index(const std::vector<Message>& ms) {
    for (int i = static_cast<int>(ms.size()) - 1; i >= 0; --i) {
        const auto& m = ms[static_cast<std::size_t>(i)];
        if (m.role == "user" || (m.role == "system" && i > 0)) {
            if (m.has_blocks && std::none_of(m.blocks.begin(), m.blocks.end(), [](const Block& b) { return !b.tool_result; })) continue;
            return i;
        }
    }
    return -1;
}

}  // namespace

std::string render_deepseek_v41_chat(const ChatTemplateInput& input) {
    if (!input.messages.is_array() || input.messages.size() == 0U) raise("messages must be a non-empty array");
    // ---- messages, with tool messages merged into user turns (merge_tool_messages) ----
    std::vector<Message> ms;
    for (const auto& raw : input.messages.as_array()) {
        if (!raw.is_object()) raise("Every message must be an object.");
        const auto role = str_or(raw.find("role"));
        if (role == "tool") {
            Block b{true, content_text(raw.find("content"), "tool"), str_or(raw.find("tool_call_id"))};
            if (!ms.empty() && ms.back().role == "user" && ms.back().has_blocks) ms.back().blocks.push_back(std::move(b));
            else {
                Message m;
                m.role = "user";
                m.has_blocks = true;
                m.blocks.push_back(std::move(b));
                ms.push_back(std::move(m));
            }
            continue;
        }
        if (role == "user") {
            auto blocks = user_blocks(raw.find("content"));
            if (!ms.empty() && ms.back().role == "user" && ms.back().has_blocks) {
                for (auto& b : blocks) ms.back().blocks.push_back(std::move(b));
            } else {
                Message m;
                m.role = "user";
                m.has_blocks = true;
                m.blocks = std::move(blocks);
                ms.push_back(std::move(m));
            }
            continue;
        }
        if (role != "system" && role != "assistant" && role != "developer" && role != "latest_reminder") raise("Unknown role: " + role);
        Message m;
        m.role = role == "developer" ? "system" : role;
        m.content = content_text(raw.find("content"), role.c_str());
        if (role == "assistant") {
            m.reasoning = str_or(raw.find("reasoning_content"), str_or(raw.find("reasoning")));
            if (const auto* tc = raw.find("tool_calls"); tc && tc->is_array() && tc->size() != 0U) m.tool_calls = *tc;
        }
        if (const auto* t = raw.find("tools"); t && t->is_array() && t->size() != 0U) m.tools = *t;
        ms.push_back(std::move(m));
    }
    if (input.tools.is_array() && input.tools.size() != 0U) {  // request-level tools: in the first system message
        if (ms.empty() || ms.front().role != "system") {
            Message s;
            s.role = "system";
            ms.insert(ms.begin(), std::move(s));
        }
        ms.front().tools = input.tools;
    }
    // ---- sort_tool_results_by_call_order ----
    std::map<std::string, std::size_t> call_order;
    for (auto& m : ms) {
        if (m.role == "assistant" && m.tool_calls.is_array()) {
            call_order.clear();
            std::size_t i = 0;
            for (const auto& tc : m.tool_calls.as_array()) {
                auto id = str_or(tc.find("id"));
                if (id.empty())
                    if (const auto* fn = tc.find("function")) id = str_or(fn->find("id"));
                if (!id.empty()) call_order[id] = i;
                ++i;
            }
        } else if (m.role == "user" && m.has_blocks) {
            std::vector<Block> results;
            for (const auto& b : m.blocks)
                if (b.tool_result) results.push_back(b);
            if (results.size() > 1U && !call_order.empty()) {
                std::stable_sort(results.begin(), results.end(), [&](const Block& a, const Block& b) {
                    const auto ia = call_order.find(a.tool_use_id), ib = call_order.find(b.tool_use_id);
                    return (ia == call_order.end() ? 0U : ia->second) < (ib == call_order.end() ? 0U : ib->second);
                });
                std::size_t k = 0;
                for (auto& b : m.blocks)
                    if (b.tool_result) b = results[k++];
            }
        }
    }
    // ---- mode, effort, dropped reasoning ----
    const bool thinking = input.enable_thinking.value_or(true);
    std::string effort = "75";
    if (input.reasoning_effort) {
        const auto& e = *input.reasoning_effort;
        if (e == "low") effort = "50";
        else if (e == "medium") effort = "62";
        else if (e == "high") effort = "75";
        else if (e == "xhigh" || e == "max") effort = "100";
        else if (!e.empty() && e.size() <= 3U && std::all_of(e.begin(), e.end(), [](char ch) { return ch >= '0' && ch <= '9'; }) &&
                 std::stoi(e) >= 1 && std::stoi(e) <= 100)
            effort = std::to_string(std::stoi(e));
        else raise("Invalid reasoning effort " + e + ": low, medium, high, xhigh, max or an integer in [1, 100]");
    }
    bool drop_thinking = !input.preserve_thinking.value_or(false);
    for (const auto& m : ms)
        if (m.tools.is_array()) drop_thinking = false;
    if (thinking && drop_thinking) {  // _drop_thinking_messages
        const int last = last_user_index(ms);
        std::vector<Message> kept;
        for (std::size_t i = 0; i < ms.size(); ++i) {
            auto m = ms[i];
            if (m.role == "assistant" && static_cast<int>(i) < last) m.reasoning.clear();
            kept.push_back(std::move(m));
        }
        ms = std::move(kept);
    }
    // ---- render_message for every message ----
    const int last_user = last_user_index(ms);
    std::string prompt = kBos;
    for (std::size_t index = 0; index < ms.size(); ++index) {
        const auto& m = ms[index];
        const std::string effort_prompt =
            index == 0U && thinking ? "Reasoning Effort: " + effort + " (range 1-100, the higher the value, the more thorough the reasoning)\n\n" : "";
        std::string out = index == 0U && (!effort_prompt.empty() || m.role == "system") ? kSystem : "";
        out += effort_prompt;
        if (m.role == "system") {
            if (index > 0U) out += kSystem;
            out += m.content;
            if (m.tools.is_array()) out += "\n\n" + render_tools(tools_for_encoding(m.tools));
        } else if (m.role == "user") {
            out += kUser + user_text(m);
        } else if (m.role == "latest_reminder") {
            out += kLatestReminder + m.content;
        } else {  // assistant
            std::string calls;
            if (m.tool_calls.is_array()) {
                std::string list;
                for (std::size_t k = 0; k < m.tool_calls.size(); ++k) list += (k ? "\n" : "") + call_to_dsml(m.tool_calls.as_array()[k]);
                calls = "\n\n<" + kDsml + kCallsBlock + ">\n" + list + "\n</" + kDsml + kCallsBlock + ">";
            }
            std::string reasoning;
            if (thinking && (!drop_thinking || static_cast<int>(index) > last_user)) reasoning = m.reasoning + kThinkEnd;
            out += reasoning + m.content + calls + kEos;
        }
        const bool next_is_turn = index + 1U < ms.size() && ms[index + 1U].role != "assistant" && ms[index + 1U].role != "latest_reminder";
        if (!next_is_turn && (m.role == "user" || (m.role == "system" && index > 0U))) {
            out += kAssistant;
            if (!drop_thinking && thinking) out += kThinkStart;
            else if (drop_thinking && thinking && static_cast<int>(index) >= last_user) out += kThinkStart;
            else out += kThinkEnd;
        }
        prompt += out;
    }
    return prompt;
}

AssistantOutput parse_deepseek_v41_output(const std::string& text, bool thinking, const Json& tools) {
    (void)tools;
    AssistantOutput out;
    out.tool_calls = Json::array();  // also when the reply ends inside the thinking block
    std::size_t at = 0;
    if (thinking) {
        const auto end = text.find(kThinkEnd);
        if (end == std::string::npos) {
            out.reasoning = text;
            return out;
        }
        out.reasoning = text.substr(0, end);
        out.reasoning_complete = true;
        at = end + kThinkEnd.size();
    }
    const std::string calls_start = "\n\n<" + kDsml + kCallsBlock;
    const auto calls = text.find(calls_start, at);
    out.content = text.substr(at, calls == std::string::npos ? std::string::npos : calls - at);
    if (calls == std::string::npos) return out;
    // parse_tool_calls: a malformed block leaves the whole text as content
    const std::string block_end = "</" + kDsml + kCallsBlock + ">", call_start = "<" + kDsml + kInvoke, call_end = "</" + kDsml + kInvoke;
    const std::string param_start = "<" + kDsml + kParameter, param_end = "/" + kDsml + kParameter;
    const auto read_until = [&](std::size_t& i, std::initializer_list<const std::string*> stops, std::string& before) -> const std::string* {
        std::size_t best = std::string::npos;
        const std::string* hit = nullptr;
        for (const auto* s : stops) {
            const auto p = text.find(*s, i);
            if (p != std::string::npos && p < best) best = p, hit = s;
        }
        if (!hit) {
            before = text.substr(i);
            i = text.size();
            return nullptr;
        }
        before = text.substr(i, best - i);
        i = best + hit->size();
        return hit;
    };
    try {
        std::size_t i = calls + calls_start.size();
        Json parsed = Json::array();
        std::string before;
        for (;;) {
            const auto* stop = read_until(i, {&call_start, &block_end}, before);
            if (before != ">\n") throw ChatTemplateError("tool call format");
            if (stop == &block_end) break;
            if (!stop) throw ChatTemplateError("tool call format");
            stop = read_until(i, {&param_start, &call_end}, before);
            const std::string name_prefix = " name=\"", name_suffix = "\">\n";
            const auto lead = before.find_first_not_of(" \t\r\n");
            if (lead == std::string::npos || before.compare(lead, 6U, "name=\"") != 0 || before.size() < name_suffix.size() ||
                before.compare(before.size() - name_suffix.size(), name_suffix.size(), name_suffix) != 0)
                throw ChatTemplateError("tool name format");
            const auto name = before.substr(lead + 6U, before.size() - name_suffix.size() - lead - 6U);
            Json arguments = Json::object();
            while (stop == &param_start) {
                std::string param;
                if (!read_until(i, {&param_end}, param)) throw ChatTemplateError("parameter format");
                // ' name="KEY" string="true|false">VALUE<'
                const auto q1 = param.find("\" string=\"");
                if (param.rfind(" name=\"", 0) != 0 || q1 == std::string::npos || param.empty() || param.back() != '<')
                    throw ChatTemplateError("parameter format");
                const auto key = param.substr(7U, q1 - 7U);
                const auto rest = param.substr(q1 + 10U);
                const bool is_string = rest.rfind("true\">", 0) == 0;
                if (!is_string && rest.rfind("false\">", 0) != 0) throw ChatTemplateError("parameter format");
                auto value = rest.substr(is_string ? 6U : 7U);
                value.pop_back();
                if (arguments.find(key)) throw ChatTemplateError("duplicate parameter");
                if (is_string) arguments.set(key, Json(value));
                else {
                    try {
                        arguments.set(key, parse_json(value));
                    } catch (const JsonError&) {
                        arguments.set(key, Json(value));
                    }
                }
                stop = read_until(i, {&param_start, &call_end}, before);
                if (before != ">\n") throw ChatTemplateError("parameter format");
            }
            Json call = Json::object();
            call.set("name", Json(name));
            call.set("arguments", std::move(arguments));
            parsed.push(std::move(call));
        }
        out.tool_calls = std::move(parsed);
    } catch (const ChatTemplateError&) {
        out.content = text.substr(at);
        out.tool_calls = Json::array();
        // a calls block that never closed (cut by the budget, a stop string or EOS) is an unfinished call
        if (text.find("</" + kDsml + kCallsBlock + ">", calls) == std::string::npos) out.unfinished_call = true;
    }
    return out;
}

}  // namespace strix::serve
