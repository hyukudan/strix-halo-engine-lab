#include "minimax/serve/minimax_chat.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace strix::minimax::serve {
namespace {

using strix::serve::Json;

constexpr std::string_view kNs = "]<]minimax[>[";
constexpr std::string_view kBod = "]~!b[";
constexpr std::string_view kBos = "]~b]";
constexpr std::string_view kEos = "[e~[";
constexpr std::string_view kThinkBegin = "<mm:think>";
constexpr std::string_view kThinkEnd = "</mm:think>";
constexpr std::string_view kImage = "]<]image[>[";
constexpr std::string_view kVideo = "]<]video[>[";
constexpr std::string_view kFrame = "]<]frame[>[";

bool is_integer_text(const std::string& text) { return text.find_first_of(".eE") == std::string::npos; }

// Python repr(float).
std::string python_float(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string_view s(buf, static_cast<std::size_t>(r.ptr - buf));  // [-]d[.ddd]e[+-]xx
    std::string out;
    if (!s.empty() && s[0] == '-') { out.push_back('-'); s.remove_prefix(1); }
    const auto e = s.find('e');
    std::string digits;
    for (const char ch : s.substr(0, e)) if (ch != '.') digits.push_back(ch);
    const int exponent = std::atoi(std::string(s.substr(e + 1U)).c_str());
    while (digits.size() > 1U && digits.back() == '0') digits.pop_back();
    const int decpt = exponent + 1;  // value = 0.DIGITS * 10^decpt
    const int nd = static_cast<int>(digits.size());
    if (decpt <= -4 || decpt > 16) {
        out += digits[0];
        if (nd > 1) { out.push_back('.'); out.append(digits, 1, std::string::npos); }
        const int x = decpt - 1;
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", x < 0 ? '-' : '+', x < 0 ? -x : x);
        out += eb;
    } else if (decpt <= 0) {
        out += "0.";
        out.append(static_cast<std::size_t>(-decpt), '0');
        out += digits;
    } else if (decpt >= nd) {
        out += digits;
        out.append(static_cast<std::size_t>(decpt - nd), '0');
        out += ".0";
    } else {
        out.append(digits, 0, static_cast<std::size_t>(decpt));
        out.push_back('.');
        out.append(digits, static_cast<std::size_t>(decpt), std::string::npos);
    }
    return out;
}

std::string python_number(const Json& v, bool json_form) {
    const auto& text = v.number_source();
    if (is_integer_text(text)) {
        if (text.find_first_not_of("-0") == std::string::npos) return "0";
        return text;
    }
    const double d = std::strtod(text.c_str(), nullptr);
    if (std::isinf(d) && json_form) return d < 0 ? "-Infinity" : "Infinity";
    return python_float(d);
}

void dumps(std::string& out, const Json& v) {
    switch (v.kind()) {
    case Json::Kind::null: out += "null"; return;
    case Json::Kind::boolean: out += v.as_bool() ? "true" : "false"; return;
    case Json::Kind::number: out += python_number(v, true); return;
    case Json::Kind::string: strix::serve::append_json_string(out, v.as_string()); return;
    case Json::Kind::array: {
        out.push_back('[');
        bool first = true;
        for (const auto& item : v.as_array()) {
            if (!first) out += ", ";
            first = false;
            dumps(out, item);
        }
        out.push_back(']');
        return;
    }
    case Json::Kind::object: {
        out.push_back('{');
        bool first = true;
        for (const auto& [key, item] : v.as_object()) {
            if (!first) out += ", ";
            first = false;
            strix::serve::append_json_string(out, key);
            out += ": ";
            dumps(out, item);
        }
        out.push_back('}');
        return;
    }
    }
}

// Python truthiness of a (possibly absent) value.
bool truthy(const Json* v) {
    if (!v) return false;
    switch (v->kind()) {
    case Json::Kind::null: return false;
    case Json::Kind::boolean: return v->as_bool();
    case Json::Kind::number: return v->as_double() != 0.0;
    case Json::Kind::string: return !v->as_string().empty();
    case Json::Kind::array: return !v->as_array().empty();
    case Json::Kind::object: return !v->as_object().empty();
    }
    return false;
}

const Json* member(const Json& obj, std::string_view key) { return obj.is_object() ? obj.find(key) : nullptr; }
bool member_is(const Json& obj, std::string_view key, std::string_view value) {
    const auto* v = member(obj, key);
    return v && v->is_string() && v->as_string() == value;
}
const std::string& require_string(const Json* v, const char* what) {
    if (!v || !v->is_string()) throw MinimaxChatError(std::string("minimax chat: ") + what + " must be a string");
    return v->as_string();
}

// {{ value }} of a value the template prints.
std::string print(const Json* v, const char* what) {
    if (!v) return {};  // undefined prints as ''
    if (v->is_array() || v->is_object())
        throw MinimaxChatError(std::string("minimax chat: ") + what + " would be printed as a Python repr");
    return python_str(*v);
}

// macro visible_text(content)
std::string visible_text(const Json* content) {
    if (!content || content->is_null()) return {};
    if (content->is_string()) return content->as_string();
    if (content->is_array()) {
        std::string out;
        for (const auto& item : content->as_array()) {
            if (item.is_object()) {
                if (member_is(item, "type", "text")) out += print(member(item, "text"), "a text part's text");
                else if (member_is(item, "type", "image")) out += kImage;
                else if (member_is(item, "type", "video")) out += kVideo;
                else if (member_is(item, "type", "frame")) out += kFrame;
            } else if (item.is_string()) out += item.as_string();
        }
        return out;
    }
    return print(content, "message content");
}

// macro to_xml(val, ns)
void to_xml(std::string& out, const Json& v) {
    if (v.is_object()) {
        for (const auto& [k, item] : v.as_object()) {
            if (item.is_null()) continue;
            out += kNs; out += '<'; out += k; out += '>';
            to_xml(out, item);
            out += kNs; out += "</"; out += k; out += '>';
        }
    } else if (v.is_array()) {
        for (const auto& item : v.as_array()) {
            out += kNs; out += "<item>";
            to_xml(out, item);
            out += kNs; out += "</item>";
        }
    } else if (v.is_null()) {
    } else if (v.is_bool()) {
        out += v.as_bool() ? "true" : "false";
    } else {
        out += python_str(v);
    }
}

std::string build_system_message(const Json* system_message, const MinimaxChatOptions& o) {
    std::string out;
    if (truthy(system_message) && truthy(member(*system_message, "content"))) {
        out += visible_text(member(*system_message, "content"));
        if (const auto* d = member(*system_message, "current_date"); truthy(d))
            out += "\nCurrent date: " + require_string(d, "current_date");
        if (const auto* l = member(*system_message, "current_location"); truthy(l))
            out += "\nCurrent location: " + require_string(l, "current_location");
    } else {
        out += "Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026. Founded in early 2022, "
               "MiniMax is a global AI foundation model company committed to advancing the frontiers of AI towards AGI.";
    }
    out += "\n\n<thinking_instructions>\n";
    out += "You have a thinking capability that allows you to reason step by step before responding. When thinking is enabled, "
           "wrap your reasoning in ";
    out += kThinkBegin; out += kThinkEnd;
    out += " tags before your response. When thinking is disabled, begin your response directly after the ";
    out += kThinkEnd;
    out += " prefix. When thinking is adaptive, decide on your own whether to think for the current turn.\n";
    static constexpr std::string_view kAdaptive =
        "Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, "
        "or when analyzing function/tool results.\n";
    if (o.thinking_mode) {
        if (*o.thinking_mode == "enabled")
            out += "Current thinking mode: enabled. You MUST think step by step before every response, including after receiving "
                   "function/tool results.\n";
        else if (*o.thinking_mode == "disabled") out += "Current thinking mode: disabled. Do not output any thinking process.\n";
        else if (*o.thinking_mode == "adaptive") out += kAdaptive;
    } else {
        out += kAdaptive;
    }
    out += "</thinking_instructions>";
    if (o.template_model_id) out += "\n\n<model_id>" + *o.template_model_id + "</model_id>";
    if (o.reasoning_effort) out += "\n\n<effort>" + *o.reasoning_effort + "</effort>";
    else if (o.effort) out += "\n\n<effort>" + *o.effort + "</effort>";
    return out;
}

std::string build_developer_message(const Json* developer_message, const MinimaxChatOptions& o) {
    if (truthy(developer_message) && truthy(member(*developer_message, "content")))
        return visible_text(member(*developer_message, "content"));
    return o.model_identity ? *o.model_identity : std::string("You are a helpful assistant.");
}

std::string strip_newlines(std::string_view s) {
    const auto b = s.find_first_not_of('\n');
    if (b == std::string_view::npos) return {};
    const auto e = s.find_last_not_of('\n');
    return std::string(s.substr(b, e - b + 1U));
}

std::string role_of(const Json& m) {
    const auto* r = member(m, "role");
    return r && r->is_string() ? r->as_string() : std::string();
}

}  // namespace

std::string python_json_dumps(const Json& value) {
    std::string out;
    dumps(out, value);
    return out;
}

std::string python_str(const Json& v) {
    switch (v.kind()) {
    case Json::Kind::null: return "None";
    case Json::Kind::boolean: return v.as_bool() ? "True" : "False";
    case Json::Kind::number: return python_number(v, false);
    case Json::Kind::string: return v.as_string();
    default: throw MinimaxChatError("minimax chat: python_str of a container");
    }
}

std::string render_minimax_chat(const Json& messages, const Json* tools, const MinimaxChatOptions& o) {
    if (!messages.is_array()) throw MinimaxChatError("minimax chat: messages must be an array");
    const auto& all = messages.as_array();
    for (const auto& m : all) if (!m.is_object()) throw MinimaxChatError("minimax chat: every message must be an object");
    const Json* system_message = nullptr;
    const Json* developer_message = nullptr;
    std::size_t first = 0;
    if (!all.empty() && role_of(all[0]) == "root") {
        system_message = &all[0];
        first = 1;
        if (all.size() > 1U && (role_of(all[1]) == "system" || role_of(all[1]) == "developer")) {
            developer_message = &all[1];
            first = 2;
        }
    } else if (!all.empty() && (role_of(all[0]) == "system" || role_of(all[0]) == "developer")) {
        developer_message = &all[0];
        first = 1;
    }
    std::vector<const Json*> conv;
    for (std::size_t i = first; i < all.size(); ++i) conv.push_back(&all[i]);

    std::string out;
    out += kBod; out += kBos; out += "system\n";
    out += build_system_message(system_message, o);
    out += kEos; out += '\n';

    out += kBos; out += "developer\n";
    out += build_developer_message(developer_message, o);
    if (tools && truthy(tools)) {
        if (!tools->is_array()) throw MinimaxChatError("minimax chat: tools must be an array");
        out += "\n\n# Tools\nYou may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:\n";
        out += "\n<tools>\n";
        for (const auto& tool : tools->as_array()) {
            const auto* fn = member(tool, "function");
            if (!fn) throw MinimaxChatError("minimax chat: a tool has no function");
            out += "<tool>" + python_json_dumps(*fn) + "</tool>\n";
        }
        out += "</tools>\n\n";
        out += "To call tools, wrap all invocations in a single ";
        out += kNs; out += "<tool_call>"; out += kNs; out += "</tool_call>";
        out += " block. Parameter values containing nested objects or arrays are recursively expanded into XML elements. Example:\n";
        out += '\n'; out += kNs; out += "<tool_call>\n";
        const auto ns = [&](std::string_view s) { out += kNs; out += s; };
        ns("<invoke name=\"tool-name-1\">");
        ns("<param-1>value-1"); ns("</param-1>");
        ns("<param-2>");
        ns("<item>");
        ns("<key-a>val-a"); ns("</key-a>");
        ns("<key-b>val-b"); ns("</key-b>");
        ns("</item>");
        ns("</param-2>");
        ns("</invoke>\n");
        ns("<invoke name=\"tool-name-2\">");
        ns("<param-1>value-1"); ns("</param-1>");
        ns("</invoke>\n");
        ns("</tool_call>");
    }
    out += kEos; out += '\n';

    bool have_last_tool_call = false;  // last_tool_call.name is not none
    for (std::size_t i = 0; i < conv.size(); ++i) {
        const Json& m = *conv[i];
        const auto role = role_of(m);
        if (role == "assistant") {
            out += kBos; out += "ai\n";
            std::string reasoning;
            std::string content = visible_text(member(m, "content"));
            if (const auto* rc = member(m, "reasoning_content"); rc && rc->is_string()) {
                reasoning = rc->as_string();
            } else if (const auto end = content.find(kThinkEnd); end != std::string::npos) {
                // content.split(end)[0].strip('\n').split(begin)[-1].strip('\n'); content.split(end)[-1].strip('\n')
                const auto head = strip_newlines(std::string_view(content).substr(0, end));
                const auto begin = head.rfind(kThinkBegin);
                reasoning = strip_newlines(begin == std::string::npos ? std::string_view(head) : std::string_view(head).substr(begin + kThinkBegin.size()));
                content = strip_newlines(std::string_view(content).substr(content.rfind(kThinkEnd) + kThinkEnd.size()));
            }
            if (!reasoning.empty()) { out += kThinkBegin; out += reasoning; out += kThinkEnd; }
            else out += kThinkEnd;
            out += content;
            const auto* calls = member(m, "tool_calls");
            if (truthy(calls)) {
                if (!calls->is_array()) throw MinimaxChatError("minimax chat: tool_calls must be an array");
                out += kNs; out += "<tool_call>\n";
                for (const auto& raw_call : calls->as_array()) {
                    const Json* call = &raw_call;
                    if (const auto* fn = member(raw_call, "function"); truthy(fn)) call = fn;
                    out += kNs; out += "<invoke name=\"" + require_string(member(*call, "name"), "tool call name") + "\">";
                    const auto* args = member(*call, "arguments");
                    Json parsed;
                    if (args && args->is_string()) {  // extension: OpenAI wire format
                        parsed = strix::serve::parse_json(args->as_string());
                        args = &parsed;
                    }
                    if (!args || !args->is_object()) throw MinimaxChatError("minimax chat: tool call arguments must be an object");
                    for (const auto& [k, v] : args->as_object()) {
                        if (v.is_null()) continue;
                        out += kNs; out += '<'; out += k; out += '>';
                        to_xml(out, v);
                        out += kNs; out += "</"; out += k; out += '>';
                    }
                    out += kNs; out += "</invoke>\n";
                }
                out += kNs; out += "</tool_call>";
                const Json& last = calls->as_array().back();
                const auto* fn = member(last, "function");
                const auto* name = truthy(fn) ? member(*fn, "name") : member(last, "name");
                have_last_tool_call = name && !name->is_null();
            } else {
                have_last_tool_call = false;
            }
            out += kEos; out += '\n';
        } else if (role == "tool") {
            if (!have_last_tool_call)
                throw MinimaxChatError("Message has tool role, but there was no previous assistant message with a tool call!");
            if (i == 0 || role_of(*conv[i - 1U]) != "tool") { out += kBos; out += "tool"; }
            out += "\n<response>";
            const auto* content = member(m, "content");
            if (content && content->is_string()) {
                out += content->as_string();
            } else {
                if (!content || !content->is_array()) throw MinimaxChatError("minimax chat: tool content must be a string or a list");
                for (const auto& tr : content->as_array()) {
                    if (tr.is_object() && member_is(tr, "type", "image")) out += kImage;
                    else if (tr.is_object() && member_is(tr, "type", "video")) out += kVideo;
                    else if (tr.is_object() && member_is(tr, "type", "frame")) out += kFrame;
                    else if (tr.is_object()) {
                        if (const auto* output = member(tr, "output")) out += print(output, "a tool result's output");
                        else if (member_is(tr, "type", "text") && member(tr, "text")) out += print(member(tr, "text"), "a tool result's text");
                        else throw MinimaxChatError("minimax chat: a tool result part would be printed as a Python repr");
                    } else {
                        out += print(&tr, "a tool result part");
                    }
                }
            }
            out += "</response>";
            if (i + 1U == conv.size() || role_of(*conv[i + 1U]) != "tool") { out += kEos; out += '\n'; }
        } else if (role == "system" || role == "developer") {
            out += kBos; out += "developer\n";
            out += build_developer_message(&m, o);
            out += kEos; out += '\n';
        } else if (role == "user") {
            out += kBos; out += "user\n";
            out += visible_text(member(m, "content"));
            out += kEos; out += '\n';
        }
    }
    if (o.add_generation_prompt) {
        out += kBos; out += "ai\n";
        if (o.thinking_mode && *o.thinking_mode == "disabled") out += kThinkEnd;
        else if (o.thinking_mode && *o.thinking_mode == "enabled") out += kThinkBegin;
    }
    return out;
}

}  // namespace strix::minimax::serve
