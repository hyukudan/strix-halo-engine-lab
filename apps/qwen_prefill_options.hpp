#pragma once

#include "strix/prefill_identity.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>

namespace strix::decode_cli {

// Shared by CLI receipts, host inspection and HTTP capability discovery.
inline std::string prefill_identity_fields(unsigned mask) {
    std::ostringstream out;
    out << "\"prefill_identity\":\"" << prefill_identity_name(mask)
        << "\",\"prefill_identity_hash\":\"" << std::hex << std::setw(16) << std::setfill('0')
        << prefill_identity_hash(mask) << "\",\"prefill_options_active\":[";
    bool first = true;
    for (unsigned i = 0; i < kPrefillOptions.size(); ++i) {
        if (!(mask & (1U << i))) continue;
        out << (first ? "" : ",") << '"' << kPrefillOptions[i].name << '"';
        first = false;
    }
    out << ']';
    return out.str();
}

inline std::string_view prefill_trim(std::string_view value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) return {};
    return value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1U);
}

// There is no general Qwen server config loader. This deliberately small startup
// format accepts only these two keys, one assignment per line, with # comments.
// A false value leaves the environment in force; CLI switches can disable more.
inline void read_prefill_config(std::istream& input, bool& exact, unsigned& disabled) {
    std::set<std::string> seen;
    std::string line;
    while (std::getline(input, line)) {
        auto text = prefill_trim(std::string_view(line).substr(0, line.find('#')));
        if (text.empty()) continue;
        const auto equal = text.find('=');
        if (equal == std::string_view::npos) throw std::invalid_argument("prefill config requires key = value");
        const auto key = prefill_trim(text.substr(0, equal));
        const auto value = prefill_trim(text.substr(equal + 1U));
        if (!seen.emplace(key).second) throw std::invalid_argument("duplicate prefill config key");
        if (key == "prefill_exact_fp16") {
            if (value != "true" && value != "false") throw std::invalid_argument("prefill_exact_fp16 requires true or false");
            exact = exact || value == "true";
        } else if (key == "prefill_disable") {
            if (value.size() < 2U || value.front() != '[' || value.back() != ']')
                throw std::invalid_argument("prefill_disable requires an array of quoted option names");
            auto rest = prefill_trim(value.substr(1, value.size() - 2U));
            while (!rest.empty()) {
                if (rest.front() != '"') throw std::invalid_argument("prefill_disable requires quoted option names");
                const auto end = rest.find('"', 1U);
                if (end == std::string_view::npos) throw std::invalid_argument("unterminated prefill option name");
                const auto name = rest.substr(1U, end - 1U);
                if (name.empty() || name.find(',') != std::string_view::npos)
                    throw std::invalid_argument("prefill_disable requires individual option names");
                disabled |= parse_prefill_option_list(name);
                rest = prefill_trim(rest.substr(end + 1U));
                if (rest.empty()) break;
                if (rest.front() != ',') throw std::invalid_argument("prefill_disable requires comma-separated names");
                rest = prefill_trim(rest.substr(1U));
                if (rest.empty()) throw std::invalid_argument("prefill_disable has a trailing comma");
            }
        } else throw std::invalid_argument("unknown prefill config key '" + std::string(key) + "'");
    }
    if (input.bad()) throw std::runtime_error("cannot read prefill config");
}

inline void load_prefill_config(const std::string& path, bool& exact, unsigned& disabled) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("cannot open prefill config");
    read_prefill_config(input, exact, disabled);
}

}  // namespace strix::decode_cli
