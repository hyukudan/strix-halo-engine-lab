#include "strix/serve/chat_template_kit.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <string>

namespace strix::serve::chat_kit {

void raise(const std::string& message) { throw ChatTemplateError(message); }

bool truthy(const Json* v) {
    if (!v || v->is_null()) return false;
    if (v->is_bool()) return v->as_bool();
    if (v->is_number()) return v->as_double() != 0.0;
    if (v->is_string()) return !v->as_string().empty();
    return v->size() > 0U;
}

std::string py_str(const Json* v, const char* what) {
    if (!v) return {};
    switch (v->kind()) {
    case Json::Kind::null: return "None";
    case Json::Kind::boolean: return v->as_bool() ? "True" : "False";
    case Json::Kind::number: return v->number_source();
    case Json::Kind::string: return v->as_string();
    default: raise(std::string("Unexpected ") + what + " type (an object or array printed as text).");
    }
}

const Json& tool_spec(const Json& tool) {
    const auto* function = tool.find("function");
    if (!function) return tool;
    if (!function->is_object()) raise("A tool's function must be an object.");
    return *function;
}

std::string tool_to_json(const Json& tool) {
    if (!tool.is_object()) raise("Every tool must be an object.");
    std::string out = "{";
    bool first = true;
    for (const auto& [key, value] : tool.as_object()) {
        if (key == "defer_loading" || key == "strict") continue;
        if (!first) out += ", ";
        first = false;
        out += "\"" + key + "\": " + value.dump_spaced();
    }
    return out + "}";
}

Json call_arguments(const Json& call, const std::string& name) {
    const auto* arguments = call.find("arguments");
    if (arguments && arguments->is_object()) return *arguments;
    if (arguments && arguments->is_string()) {
        if (strip(arguments->as_string()).empty()) return Json::object();
        Json parsed;
        try { parsed = parse_request_json(arguments->as_string()); }
        catch (const JsonError&) { raise("Tool call arguments for function \"" + name + "\" are not valid JSON."); }
        if (!parsed.is_object()) raise("Tool call arguments for function \"" + name + "\" must be a JSON object.");
        return parsed;
    }
    raise("Tool call arguments for function \"" + name + "\" must be an object/mapping or a JSON string.");
}

std::string python_float(double value) {
    if (!std::isfinite(value)) raise("Expected a finite JSON number.");
    std::array<char, 32> buffer{};
    const auto scientific = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                          std::chars_format::scientific);
    if (scientific.ec != std::errc{}) raise("Could not format a JSON number.");
    const std::string_view text(buffer.data(), scientific.ptr);
    const auto e = text.find('e');
    int exponent{};
    const auto parsed = std::from_chars(buffer.data() + e + 2U, scientific.ptr, exponent);
    if (parsed.ec != std::errc{}) raise("Could not format a JSON number exponent.");
    if (text[e + 1U] == '-') exponent = -exponent;
    if (exponent < -4 || exponent >= 16) return std::string(text);
    const auto fixed = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed);
    if (fixed.ec != std::errc{}) raise("Could not format a JSON number.");
    std::string result(buffer.data(), fixed.ptr);
    if (result.find('.') == std::string::npos) result += ".0";
    return result;
}

namespace {

Json json_for_template(const Json& value, JsonKeyOrder order) {
    const auto nested_order = order == JsonKeyOrder::recursive ? order : JsonKeyOrder::preserve;
    if (value.is_object()) {
        Json::Object members;
        members.reserve(value.size());
        for (const auto& [key, item] : value.as_object())
            members.emplace_back(key, json_for_template(item, nested_order));
        if (order != JsonKeyOrder::preserve)
            std::ranges::sort(members, {}, &Json::Member::first);
        return Json(std::move(members));
    }
    if (value.is_array()) {
        Json::Array items;
        items.reserve(value.size());
        for (const auto& item : value.as_array()) items.push_back(json_for_template(item, nested_order));
        return Json(std::move(items));
    }
    if (value.is_number()) {
        const auto& number = value.number_source();
        if (number.find_first_of(".eE") != std::string::npos)
            return Json::number_text(python_float(value.as_double()));
        if (number == "-0") return Json(0);
    }
    return value;
}

// As parse_assistant_output: the schema of one parameter of a request tool.
const Json* parameter_schema(const Json& tools, const std::string& function, const std::string& parameter) {
    if (!tools.is_array()) return nullptr;
    for (const auto& tool : tools.as_array()) {
        const auto* spec = tool.find("function");
        if (!spec) spec = &tool;
        const auto* name = spec->find("name");
        if (!name || !name->is_string() || name->as_string() != function) continue;
        const auto* parameters = spec->find("parameters");
        if (!parameters) parameters = spec->find("input_schema");
        const auto* properties = parameters ? parameters->find("properties") : nullptr;
        return properties ? properties->find(parameter) : nullptr;
    }
    return nullptr;
}

// Strings (or no schema) stay text; other types are parsed as JSON when they parse, else kept as text.
Json typed_value(const std::string& text, const Json* schema) {
    const auto* type = schema ? schema->find("type") : nullptr;
    const bool string_type = !type || (type->is_string() && type->as_string() == "string");
    if (string_type) return Json(text);
    try { return parse_json(text); } catch (const JsonError&) { return Json(text); }
}

}  // namespace

std::string compact_json(const Json& value, JsonKeyOrder order) {
    return json_for_template(value, order).dump();
}

std::string spaced_json(const Json& value, JsonKeyOrder order) {
    return json_for_template(value, order).dump_spaced();
}

AssistantOutput parse_kv_tool_output(const std::string& text, bool reasoning_open, const KvToolGrammar& g, const Json& tools) {
    AssistantOutput out;
    out.tool_calls = Json::array();  // also when the reply ends inside the thinking block
    std::string rest = text;
    if (reasoning_open) {
        const auto end = text.find(g.think_end);
        if (end == std::string::npos) { out.reasoning = strip(text); return out; }
        out.reasoning = strip(text.substr(0, end));
        out.reasoning_complete = true;
        rest = text.substr(end + g.think_end.size());
    }
    const auto first_call = rest.find(g.call_start);
    auto content_end = first_call;
    if (!g.calls_start.empty()) content_end = std::min(content_end, rest.find(g.calls_start));
    out.content = strip(content_end == std::string::npos ? rest : rest.substr(0, content_end));
    for (auto at = first_call; at != std::string::npos; at = rest.find(g.call_start, at)) {
        const auto begin = at + g.call_start.size();
        const auto close = rest.find(g.call_end, begin);
        const auto block = rest.substr(begin, close == std::string::npos ? std::string::npos : close - begin);
        at = close == std::string::npos ? rest.size() : close + g.call_end.size();
        if (close == std::string::npos) out.unfinished_call = true;
        const auto name = strip(block.substr(0, block.find(g.key_start)));
        if (name.empty()) continue;
        Json arguments = Json::object();
        for (auto p = block.find(g.key_start); p != std::string::npos; p = block.find(g.key_start, p)) {
            const auto key_end = block.find(g.key_end, p);
            if (key_end == std::string::npos) break;
            const auto key = strip(block.substr(p + g.key_start.size(), key_end - p - g.key_start.size()));
            const auto value_at = block.find(g.value_start, key_end);
            if (value_at == std::string::npos ||
                !strip(block.substr(key_end + g.key_end.size(), value_at - key_end - g.key_end.size())).empty())
                break;
            const auto value_begin = value_at + g.value_start.size();
            const auto value_end = block.find(g.value_end, value_begin);
            const auto value = block.substr(value_begin, value_end == std::string::npos ? std::string::npos : value_end - value_begin);
            arguments.set(key, typed_value(value, parameter_schema(tools, name, key)));
            p = value_end == std::string::npos ? block.size() : value_end + g.value_end.size();
        }
        Json call = Json::object();
        call.set("name", Json(name));
        call.set("arguments", std::move(arguments));
        out.tool_calls.push(std::move(call));
        out.call_complete.push_back(close != std::string::npos);
    }
    return out;
}

}  // namespace strix::serve::chat_kit
