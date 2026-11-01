#include "strix/serve/request_fields.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <string>

namespace strix::serve {
namespace {
// A JSON number as a double; NaN when it is not a number or does not fit one (never equal to a neutral value).
double number_or_nan(const Json& value) {
    if (!value.is_number()) return std::numeric_limits<double>::quiet_NaN();
    try { return value.as_double(); } catch (const JsonError&) { return std::numeric_limits<double>::quiet_NaN(); }
}
}  // namespace

// Generation-affecting fields this server does not execute yet: a non-neutral value is refused (400 naming the
// field), never ignored. Neutral values (0 penalties, empty bias, logprobs false, text format, auto/none choice) pass.
void refuse_unexecuted(const Json& body, bool allow_constraints, bool allow_selection) {
    const auto refuse = [](const std::string& param, const std::string& what) {
        throw RequestFieldError(param, what + " is not supported by this server yet (see /v1/capabilities)");
    };
    const auto set = [&](const char* key) -> const Json* { const auto* v = body.find(key); return v && !v->is_null() ? v : nullptr; };
    for (const char* key : {"prefill_exact_fp16", "prefill_disable"})
        if (set(key)) refuse(key, "prefill identity overrides are server-level only");
    if (!allow_selection) {
        for (const char* key : {"presence_penalty", "frequency_penalty"})
            if (const auto* v = set(key); v && (!v->is_number() || number_or_nan(*v) != 0.0)) refuse(key, key);
        if (const auto* v = set("repetition_penalty"); v && (!v->is_number() || number_or_nan(*v) != 1.0)) refuse("repetition_penalty", "repetition_penalty");
        if (const auto* v = set("logit_bias"); v && (!v->is_object() || v->size() != 0U)) refuse("logit_bias", "logit_bias");
        if (const auto* v = set("logprobs"); v && !(v->is_bool() && !v->as_bool()) && !(v->is_number() && number_or_nan(*v) == 0.0)) refuse("logprobs", "logprobs");
        if (const auto* v = set("top_logprobs"); v && (!v->is_number() || number_or_nan(*v) != 0.0)) refuse("top_logprobs", "top_logprobs");
    }
    if (const auto* v = set("response_format")) {
        const auto* type = v->is_object() ? v->find("type") : nullptr;
        if (!type || !type->is_string() || (type->as_string() != "text" && !(allow_constraints && (type->as_string() == "json_object" || type->as_string() == "json_schema")))) refuse("response_format", "response_format other than text");
    }
    if (const auto* v = set("parallel_tool_calls"); v && !(v->is_bool() && v->as_bool())) refuse("parallel_tool_calls", "parallel_tool_calls false");
    if (const auto* v = set("tool_choice")) {
        const bool chat_neutral = v->is_string() && (v->as_string() == "auto" || v->as_string() == "none");
        const auto* type = v->is_object() ? v->find("type") : nullptr;  // Anthropic form
        const bool messages_neutral = type && type->is_string() && (type->as_string() == "auto" || type->as_string() == "none");
        if (!chat_neutral && !messages_neutral) refuse("tool_choice", "required or named tool_choice");
        if (const auto* single = v->is_object() ? v->find("disable_parallel_tool_use") : nullptr; single && single->is_bool() && single->as_bool())
            refuse("tool_choice.disable_parallel_tool_use", "disable_parallel_tool_use");
    }
    if (const auto* v = set("functions"); v && (!v->is_array() || v->size() != 0U)) refuse("functions", "legacy functions (use tools)");
    if (const auto* v = set("function_call"); v && !(v->is_string() && (v->as_string() == "auto" || v->as_string() == "none")))
        refuse("function_call", "legacy function_call");
    if (const auto* tools = set("tools"); tools && tools->is_array())
        for (std::size_t i = 0; i < tools->size(); ++i) {
            const auto& tool = tools->as_array()[i];
            const auto* function = tool.is_object() ? tool.find("function") : nullptr;
            const auto* strict = function && function->is_object() ? function->find("strict") : nullptr;
            if (strict && strict->is_bool() && strict->as_bool()) refuse("tools[" + std::to_string(i) + "].function.strict", "strict tools");
            const auto* top = tool.is_object() ? tool.find("strict") : nullptr;  // Anthropic form
            if (top && top->is_bool() && top->as_bool()) refuse("tools[" + std::to_string(i) + "].strict", "strict tools");
        }
}

// The exact non-negative integer a JSON number spelling denotes (-?int(.frac)?(e[+-]?exp)?), or nothing when it is
// fractional, negative (other than zero) or above uint64.
std::optional<std::uint64_t> exact_integer(std::string_view text) {
    std::size_t i = 0;
    const bool negative = i < text.size() && text[i] == '-';
    if (negative) ++i;
    std::string digits;  // integer and fraction digits; the value is digits * 10^(exponent - fraction digits)
    std::int64_t scale = 0;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') digits += text[i++];
    if (digits.empty()) return std::nullopt;
    if (i < text.size() && text[i] == '.') {
        ++i;
        const auto start = digits.size();
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') digits += text[i++];
        if (digits.size() == start) return std::nullopt;
        scale -= static_cast<std::int64_t>(digits.size() - start);
    }
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        const bool minus = i < text.size() && text[i] == '-';
        if (i < text.size() && (text[i] == '-' || text[i] == '+')) ++i;
        std::int64_t exponent = 0;
        const auto start = i;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            if (exponent > 100000) return std::nullopt;  // far beyond any uint64 or any zero-preserving scale
            exponent = exponent * 10 + (text[i++] - '0');
        }
        if (i == start) return std::nullopt;
        scale += minus ? -exponent : exponent;
    }
    if (i != text.size()) return std::nullopt;
    const auto first = digits.find_first_not_of('0');
    if (first == std::string::npos) return std::uint64_t{0};  // zero, whatever its sign or spelling
    if (negative) return std::nullopt;
    digits.erase(0, first);
    if (scale < 0) {  // the dropped digits must all be zero
        const auto drop = static_cast<std::size_t>(-scale);
        if (drop >= digits.size() || digits.find_first_not_of('0', digits.size() - drop) != std::string::npos) return std::nullopt;
        digits.resize(digits.size() - drop);
    } else {
        if (static_cast<std::uint64_t>(scale) + digits.size() > 20U) return std::nullopt;
        digits.append(static_cast<std::size_t>(scale), '0');
    }
    std::uint64_t out = 0;
    const auto r = std::from_chars(digits.data(), digits.data() + digits.size(), out);
    if (r.ec != std::errc{} || r.ptr != digits.data() + digits.size()) return std::nullopt;
    return out;
}
std::uint64_t unsigned_integer(const Json& value, const char* field, std::uint64_t maximum) {
    std::uint64_t parsed = 0;
    if (value.is_number()) {
        const auto& text = value.number_source();
        const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
        if (result.ec == std::errc{} && result.ptr == text.data() + text.size() && parsed <= maximum) return parsed;
        // an integral value written as 10.0, 1e1 or -0, read exactly from its digits (never through a double)
        if (const auto exact = exact_integer(text); exact && *exact <= maximum) return *exact;
    }
    throw RequestFieldError(field, std::string(field) + " must be a non-negative integer <= " + std::to_string(maximum));
}

double finite_number(const Json& value, const char* field, double low, double high, bool open_low, bool open_high) {
    double v = 0.0;
    bool finite = false;
    if (value.is_number()) {
        try { v = value.as_double(); finite = std::isfinite(v); }
        catch (const JsonError&) {}
    }
    if (!finite) throw RequestFieldError(field, std::string(field) + " must be a finite number");
    if ((open_low ? v <= low : v < low) || (open_high ? v >= high : v > high))
        throw RequestFieldError(field, std::string(field) + " is outside its supported range");
    return v;
}

}  // namespace strix::serve
