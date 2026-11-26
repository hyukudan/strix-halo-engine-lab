#pragma once

#include "strix/profiles/hy4.hpp"

#include <charconv>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::hy4 {

struct ProfileOverrides {
    std::optional<std::uint32_t> prompt_rows;
    std::optional<bool> head_q8;
};

// The shared profile parser records both spellings of each lever. Consume those same options at the
// application boundary and validate their typed execution values once for rank and serve.
inline ProfileOverrides take_profile_options(std::vector<std::string>& args, ::strix::profile::Inputs& inputs) {
    ::strix::profile::take_options(profile::engine(), args, inputs);
    ProfileOverrides out;
    std::vector<std::string> kept;
    kept.reserve(args.size());
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view word = args[i];
        const auto equal = word.find('=');
        const auto name = word.substr(0, equal);
        if (name != "--prompt-rows" && name != "--head-q8") {
            kept.push_back(args[i]);
            continue;
        }
        if (equal == std::string_view::npos && i + 1 == args.size())
            throw std::invalid_argument(std::string(name) + " needs a value");
        const auto value = equal == std::string_view::npos ? std::string_view(args[++i]) : word.substr(equal + 1);
        if (name == "--head-q8") {
            if (value != "0" && value != "1") throw std::invalid_argument("--head-q8 expects 0 or 1");
            out.head_q8 = value == "1";
        } else {
            std::uint32_t rows = 0U;
            const auto* end = value.data() + value.size();
            const auto [ptr, error] = std::from_chars(value.data(), end, rows);
            if (error != std::errc{} || ptr != end)
                throw std::invalid_argument("--prompt-rows needs a non-negative 32-bit integer");
            out.prompt_rows = rows;
        }
    }
    args = std::move(kept);
    return out;
}

}  // namespace strix::hy4
