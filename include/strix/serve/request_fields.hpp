#pragma once
// Request field checks shared by the HTTP front ends (Qwen/DeepSeek api.cpp, MiniMax serve): one implementation of the
// "unsupported generation fields return 400 naming the field" contract and of exact integer fields.
#include "strix/serve/json.hpp"

#include <cstdint>
#include <optional>
#include <utility>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::serve {

// A refused request field: the front end answers 400 invalid_request_error with `param` = the field.
struct RequestFieldError : std::invalid_argument {
    RequestFieldError(std::string field, const std::string& message) : std::invalid_argument(message), param(std::move(field)) {}
    std::string param;
};

// Generation-affecting fields a server does not execute: a non-neutral value is refused, never ignored. Neutral values
// (0 penalties, empty bias, logprobs false, text format, auto/none choice) pass. allow_constraints admits json_object /
// json_schema response formats; allow_selection = false also refuses penalties, logit_bias and logprobs.
void refuse_unexecuted(const Json& body, bool allow_constraints = false, bool allow_selection = true);

// The exact non-negative integer a JSON number spelling denotes, or nothing when it is fractional, negative (other than
// zero) or above uint64.
[[nodiscard]] std::optional<std::uint64_t> exact_integer(std::string_view text);
// A non-negative integer field <= maximum, read exactly from its JSON spelling (10.0, 1e1 and -0 are integers).
[[nodiscard]] std::uint64_t unsigned_integer(const Json& value, const char* field, std::uint64_t maximum);
// A finite number field within [low, high] (open bounds when asked).
[[nodiscard]] double finite_number(const Json& value, const char* field, double low, double high, bool open_low, bool open_high);

}  // namespace strix::serve
