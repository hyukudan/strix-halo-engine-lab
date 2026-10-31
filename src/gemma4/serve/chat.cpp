#include "chat.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace strix::gemma4::serve {
namespace {
using strix::serve::ChatTemplateError;
constexpr std::string_view quote = "<|\"|>";
const Json empty;
const Json& get(const Json& j, std::string_view key) { const auto* p = j.find(key); return p ? *p : empty; }
bool truth(const Json& j) {
    if (j.is_null()) return false;
    if (j.is_bool()) return j.as_bool();
    if (j.is_string()) return !j.as_string().empty();
    if (j.is_number()) return j.as_double() != 0;
    return j.size() != 0;
}
[[noreturn]] void fail(const std::string& s) { throw ChatTemplateError("gemma4 chat: " + s); }
std::string text(const Json& j) { return j.is_null() ? "" : j.as_string(); }
std::string upper(std::string s) {
    for (auto& c : s) if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    return s;
}
#include "unicode_lower.inc"
template<std::size_t N> bool in_ranges(std::uint32_t cp, const CaseRange (&ranges)[N]) {
    const auto* r = std::lower_bound(std::begin(ranges), std::end(ranges), cp,
        [](const auto& range, auto c) { return range.last < c; });
    return r != std::end(ranges) && r->first <= cp;
}
std::string lower(const std::string& s) {
    std::vector<std::pair<std::uint32_t, std::string_view>> chars;
    // Strict validation also covers Json values constructed directly by a caller.
    (void)strix::serve::parse_json_unique(Json(s).dump());
    for (std::size_t at = 0; at < s.size();) {
        const auto b = static_cast<unsigned char>(s[at]);
        const std::size_t n = b < 0x80U ? 1U : b < 0xE0U ? 2U : b < 0xF0U ? 3U : 4U;
        std::uint32_t cp = b & (n == 1 ? 127U : n == 2 ? 31U : n == 3 ? 15U : 7U);
        for (std::size_t k = 1; k < n; ++k) cp = (cp << 6U) | (static_cast<unsigned char>(s[at + k]) & 63U);
        chars.emplace_back(cp, std::string_view(s).substr(at, n)); at += n;
    }
    std::string out;
    for (std::size_t i = 0; i < chars.size(); ++i) {
        const auto cp = chars[i].first;
        if (cp == 0x3A3U) {
            bool before = false, after = false;
            for (auto j = i; j > 0;) {
                const auto c = chars[--j].first;
                if (!in_ranges(c, ignorable_ranges)) { before = in_ranges(c, cased_ranges); break; }
            }
            for (auto j = i + 1; j < chars.size(); ++j) {
                const auto c = chars[j].first;
                if (!in_ranges(c, ignorable_ranges)) { after = in_ranges(c, cased_ranges); break; }
            }
            if (before && !after) { out += "\xCF\x82"; continue; }
        }
        const auto* m = std::lower_bound(std::begin(lower_mappings), std::end(lower_mappings), cp,
            [](const auto& item, auto c) { return item.cp < c; });
        out += m != std::end(lower_mappings) && m->cp == cp ? m->text : chars[i].second;
    }
    return out;
}
std::string number(const Json& value) {
    const auto& source = value.number_source();
    if (source.find_first_of(".eE") == std::string::npos) return source == "-0" ? "0" : source;
    const auto x = value.as_double();
    if (!std::isfinite(x)) fail("nonfinite tool number");
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), x, std::chars_format::scientific);
    if (result.ec != std::errc{}) fail("tool number conversion failed");
    const std::string scientific(buffer, result.ptr);
    const auto e = scientific.find('e');
    const int exponent = std::stoi(scientific.substr(e + 1));
    std::string digits = scientific.substr(0, e);
    const bool negative = digits.starts_with('-');
    if (negative) digits.erase(0, 1);
    if (const auto dot = digits.find('.'); dot != std::string::npos) digits.erase(dot, 1);
    std::string out = negative ? "-" : "";
    if (exponent >= -4 && exponent < 16) {
        const int point = exponent + 1;
        if (point <= 0) out += "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
        else if (static_cast<std::size_t>(point) >= digits.size()) out += digits + std::string(static_cast<std::size_t>(point) - digits.size(), '0') + ".0";
        else out += digits.substr(0, static_cast<std::size_t>(point)) + '.' + digits.substr(static_cast<std::size_t>(point));
    } else {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += '.' + digits.substr(1);
        out += scientific.substr(e);
    }
    return out;
}
std::vector<const Json::Member*> sorted(const Json& j) {
    std::vector<const Json::Member*> out;
    for (const auto& m : j.as_object()) out.push_back(&m);
    std::stable_sort(out.begin(), out.end(), [](const auto* a, const auto* b) { return lower(a->first) < lower(b->first); });
    return out;
}
std::string quoted(std::string_view s) {
    if (s.find(quote) != std::string_view::npos) fail("string contains ambiguous tool quote delimiter");
    return std::string(quote) + std::string(s) + std::string(quote);
}
void bare_key(std::string_view s) {
    if (s.empty() || s.find_first_of("{}[]:,\r\n\t ") != std::string_view::npos || s.find(quote) != std::string_view::npos)
        fail("unsupported unquoted tool key or function name");
}
void append(std::string& out, const std::string& piece) { if (!out.empty()) out += ','; out += piece; }
std::string parameters(const Json& properties, bool filter = false, unsigned depth = 0) {
    if (depth > 64) fail("schema nesting exceeds 64");
    std::string out;
    for (const auto* member : sorted(properties)) {
        const auto& [key, v] = *member;
        if (filter && (key == "description" || key == "type" || key == "properties" || key == "required" || key == "nullable")) continue;
        bare_key(key);
        const auto type = upper(text(get(v, "type")));
        std::string body;
        if (truth(get(v, "description"))) append(body, "description:" + quoted(text(get(v, "description"))));
        if (type == "STRING" && truth(get(v, "enum"))) append(body, "enum:" + format_argument(get(v, "enum")));
        if (type == "ARRAY" && get(v, "items").is_object() && truth(get(v, "items"))) {
            std::string items;
            for (const auto* im : sorted(get(v, "items"))) {
                const auto& [ik, iv] = *im;
                if (iv.is_null()) continue;
                bare_key(ik);
                if (ik == "properties") append(items, "properties:{" + (iv.is_object() ? parameters(iv, false, depth + 1) : "") + "}");
                else if (ik == "type") {
                    Json types = Json::array();
                    if (iv.is_array()) for (const auto& t : iv.as_array()) types.push(upper(t.as_string()));
                    append(items, "type:" + format_argument(iv.is_string() ? Json(upper(iv.as_string())) : types));
                } else append(items, ik + ":" + format_argument(iv));
            }
            append(body, "items:{" + items + "}");
        }
        if (truth(get(v, "nullable"))) append(body, "nullable:true");
        if (type == "OBJECT") {
            append(body, "properties:{" + parameters(get(v, "properties").is_object() ? get(v, "properties") : v,
                !get(v, "properties").is_object(), depth + 1) + "}");
            if (truth(get(v, "required"))) append(body, "required:" + format_argument(get(v, "required")));
        }
        if (type.empty()) fail("schema property needs a type");
        append(body, "type:" + quoted(type));
        append(out, key + ":{" + body + "}");
    }
    return out;
}
std::string declaration(const Json& tool) {
    if (text(get(tool, "type")) != "function") fail("only function tools are supported");
    const auto& fn = get(tool, "function");
    if (fn.find("description") && !get(fn, "description").is_string()) fail("function description must be a string");
    const auto name = text(get(fn, "name")); bare_key(name);
    std::string out = "declaration:" + name + "{description:" + quoted(text(get(fn, "description")));
    const auto& p = get(fn, "parameters");
    if (truth(p)) {
        if (text(get(p, "type")).empty()) fail("tool parameters need a type");
        out += ",parameters:{";
        if (truth(get(p, "properties"))) out += "properties:{" + parameters(get(p, "properties")) + "},";
        if (truth(get(p, "required"))) out += "required:" + format_argument(get(p, "required")) + ',';
        out += "type:" + quoted(upper(text(get(p, "type")))) + '}';
    }
    if (fn.find("response")) {
        const auto& r = get(fn, "response");
        if (upper(text(get(r, "type"))) != "OBJECT") fail("only OBJECT tool response declarations are supported");
        out += ",response:{";
        if (truth(get(r, "description"))) out += "description:" + quoted(text(get(r, "description"))) + ',';
        out += "type:" + quoted("OBJECT") + '}';
    }
    return out + '}';
}
std::string strip_thinking(std::string s) {
    std::string result;
    std::size_t at = 0;
    for (;;) {
        const auto end = s.find("<channel|>", at);
        auto part = s.substr(at, end == std::string::npos ? end : end - at);
        const auto open = part.find("<|channel>");
        result += part.substr(0, open);
        if (end == std::string::npos) break;
        at = end + 10;
    }
    return strix::serve::strip(result);
}
std::string content(const Json& j, bool system, bool model, bool raw = false) {
    if (j.is_null()) return {};
    const auto one = [&](const std::string& s) { return raw ? s : model ? strip_thinking(s) : strix::serve::strip(s); };
    if (j.is_string()) return one(j.as_string());
    if (!j.is_array()) fail("content must be text or text parts");
    std::string out;
    for (const auto& p : j.as_array()) {
        if (text(get(p, "type")) != "text") fail("non-text content part is unsupported");
        out += one(get(p, "text").as_string());
        if (system) out += ' ';
    }
    return out;
}
std::string response(const std::string& name, const Json& body) {
    bare_key(name);
    auto s = format_argument(body.is_object() ? body : Json(Json::Object{{"value", body}}), false);
    return "<|tool_response>response:" + name + s + "<tool_response|>";
}
} // namespace

std::string format_argument(const Json& value, bool quote_keys) {
    // Bound traversal independently of the shared JSON parser (callers can construct Json).
    const auto visit = [&](const auto& self, const Json& j, unsigned depth) -> std::string {
        if (depth > 64) fail("argument nesting exceeds 64");
        if (j.is_string()) return quoted(j.as_string());
        if (j.is_number()) return number(j);
        if (j.is_object()) {
            std::string out;
            for (const auto* m : sorted(j)) {
                if (!quote_keys) bare_key(m->first);
                append(out, (quote_keys ? quoted(m->first) : m->first) + ':' + self(self, m->second, depth + 1));
            }
            return '{' + out + '}';
        }
        if (j.is_array()) {
            std::string out;
            for (const auto& item : j.as_array()) append(out, self(self, item, depth + 1));
            return '[' + out + ']';
        }
        return j.dump();
    };
    return visit(visit, value, 0);
}

ChatInput chat_input(Json messages, const Json& kwargs) {
    ChatInput out; out.messages = std::move(messages);
    for (const auto& [key, v] : kwargs.as_object()) {
        if (key == "tools") out.tools = v;
        else if (key == "enable_thinking") out.enable_thinking = v.as_bool();
        else if (key == "preserve_thinking") out.preserve_thinking = v.as_bool();
        else if (key == "conversational_mode") out.conversational_mode = v.as_bool();
        else if (key == "add_generation_prompt") out.add_generation_prompt = v.as_bool();
        else fail("unsupported template option: " + key);
    }
    return out;
}

std::string render_chat(const ChatInput& input) {
    auto messages = input.messages.as_array();
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto role = text(get(messages[i], "role"));
        if (role != "user" && role != "assistant" && role != "model" && role != "tool" && role != "system" && role != "developer") fail("unsupported message role");
        if (i != 0 && (role == "system" || role == "developer")) fail("system/developer must be the first message");
        if (role != "assistant" && truth(get(messages[i], "tool_calls"))) fail("tool_calls require assistant role");
    }
    const auto initial_system = [&] { return !messages.empty() && (text(get(messages[0], "role")) == "system" || text(get(messages[0], "role")) == "developer"); };
    if (input.conversational_mode) {
        const std::string instruction = "You are having a spoken conversation with the user. Respond as briefly as possible while still addressing their questions fully.";
        std::string s = initial_system() ? strix::serve::strip(content(get(messages[0], "content"), true, false)) : "";
        if (s != instruction) s = (s.empty() ? "" : s + "\n\n") + instruction;
        if (initial_system()) {
            // Json copies share objects; replace this message instead of mutating the caller.
            messages[0] = Json(Json::Object{{"role", get(messages[0], "role")}, {"content", s}});
        } else messages.insert(messages.begin(), Json(Json::Object{{"role", "system"}, {"content", s}}));
    }
    std::string out = "<bos>";
    std::size_t start = 0;
    enum class Last { other, call, response } last = Last::other;
    if (input.enable_thinking || truth(input.tools) || initial_system()) {
        out += "<|turn>system\n";
        if (input.enable_thinking) out += "<|think|>\n";
        if (initial_system()) { out += content(get(messages[0], "content"), true, false); start = 1; }
        if (truth(input.tools)) for (const auto& t : input.tools.as_array()) out += "<|tool>" + declaration(t) + "<tool|>";
        out += "<turn|>\n";
    }
    std::size_t last_user = SIZE_MAX;
    for (std::size_t i = start; i < messages.size(); ++i) if (text(get(messages[i], "role")) == "user") last_user = i;
    std::string previous;
    for (std::size_t i = start; i < messages.size(); ++i) {
        const auto& m = messages[i];
        const auto role = text(get(m, "role"));
        if (role == "tool") {
            if (i == start || (text(get(messages[i - 1], "role")) != "tool" && !truth(get(messages[i - 1], "tool_calls")))) fail("orphan tool result");
            continue;
        }
        const bool model = role == "assistant" || role == "model";
        last = Last::other;
        if (!(model && previous == "assistant")) out += "<|turn>" + (model ? "model" : role) + '\n';
        const auto& calls = get(m, "tool_calls");
        const auto& r = truth(get(m, "reasoning")) ? get(m, "reasoning") : get(m, "reasoning_content");
        if (truth(r) && (last_user == SIZE_MAX || i > last_user || (input.preserve_thinking && truth(calls))))
            out += "<|channel>thought\n" + r.as_string() + "\n<channel|>";
        const auto captured = content(get(m, "content"), false, model);
        if (model && truth(calls)) out += captured;
        if (truth(calls)) {
            for (const auto& call : calls.as_array()) {
                const auto& f = get(call, "function");
                const auto name = text(get(f, "name")); bare_key(name);
                const auto& args = get(f, "arguments");
                if (!args.is_object() && !args.is_null()) fail("tool arguments must be a JSON object, not a string");
                out += "<|tool_call>call:" + name + format_argument(args.is_null() ? Json::object() : args, false) + "<tool_call|>";
            }
            last = Last::call;
        }
        bool has_response = false;
        const auto& embedded = get(m, "tool_responses");
        if (truth(embedded)) {
            for (const auto& tr : embedded.as_array()) {
                out += response(truth(get(tr, "name")) ? text(get(tr, "name")) : "unknown", get(tr, "response"));
                has_response = true;
            }
        } else if (truth(calls)) {
            for (auto k = i + 1; k < messages.size() && text(get(messages[k], "role")) == "tool"; ++k) {
                const auto& tr = messages[k];
                std::string name = truth(get(tr, "name")) ? text(get(tr, "name")) : "unknown";
                for (const auto& call : calls.as_array()) if (get(call, "id").dump() == get(tr, "tool_call_id").dump())
                    name = text(get(get(call, "function"), "name"));
                const auto& body = get(tr, "content");
                out += response(name, body.is_array() ? Json(content(body, false, false, true)) : body);
                has_response = true;
            }
        }
        if (has_response) last = Last::response;
        if (!(model && truth(calls))) out += captured;
        std::string next;
        for (auto j = i + 1; j < messages.size(); ++j) if (text(get(messages[j], "role")) != "tool") { next = text(get(messages[j], "role")); break; }
        const bool continues = model && next == "assistant" && (!truth(calls) || has_response);
        if (last == Last::call && !has_response) out += "<|tool_response>";
        else if (!continues && !(has_response && strix::serve::strip(captured).empty() && next.empty())) out += "<turn|>\n";
        previous = role;
    }
    if (input.add_generation_prompt) {
        if (last == Last::other) out += "<|turn>model\n";
        else if (last == Last::response && input.enable_thinking) out += "<|channel>thought\n";
    }
    return out;
}

BudgetContinuation continue_after_budget(const ChatInput& original, const Output& partial,
    std::uint32_t used, std::uint32_t budget, std::uint32_t maximum) {
    if (!original.enable_thinking || !budget || maximum <= budget || used != budget || partial.stopped ||
        partial.unfinished_call || partial.unfinished_control ||
        truth(partial.tool_calls)) fail("invalid thinking-budget continuation");
    auto next = original;
    next.messages = Json(original.messages.as_array());
    next.messages.push(Json(Json::Object{{"role", "assistant"}, {"reasoning_content", partial.reasoning}, {"content", partial.content}}));
    next.preserve_thinking = true;
    next.add_generation_prompt = false;
    auto prompt = render_chat(next);
    constexpr std::string_view end = "<turn|>\n";
    if (!prompt.ends_with(end)) fail("budget continuation is not an assistant text turn");
    prompt.resize(prompt.size() - end.size());
    return {std::move(prompt), maximum - used};
}

} // namespace strix::gemma4::serve
