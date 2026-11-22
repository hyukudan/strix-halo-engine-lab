#pragma once

#include "strix/serve/json.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <span>

namespace strix::kit {

using Bytes = std::vector<std::uint8_t>;
using Buffers = std::vector<std::pair<std::string, Bytes>>;
inline constexpr char capture_schema[] = "strix-conformance-capture/v1";

// Bytes are opaque. Producers define the scalar encoding of each buffer in the case contract.
struct Capture {
    std::string identity;
    Buffers inputs, expected;
};

inline const Bytes& buffer(const Buffers& buffers, std::string_view name) {
    for (const auto& [key, value] : buffers) if (key == name) return value;
    throw std::invalid_argument("missing capture buffer: " + std::string(name));
}

namespace detail {
using serve::Json;

inline const Json& member(const Json& object, std::string_view key) {
    const auto* result = object.find(key);
    if (!result) throw std::invalid_argument("missing capture field: " + std::string(key));
    return *result;
}

inline unsigned nibble(char ch) {
    if (ch >= '0' && ch <= '9') return static_cast<unsigned>(ch - '0');
    if (ch >= 'a' && ch <= 'f') return 10U + static_cast<unsigned>(ch - 'a');
    throw std::invalid_argument("capture buffer requires lowercase hex");
}

inline Bytes decode_hex(std::string_view text) {
    if (text.size() % 2 != 0) throw std::invalid_argument("odd capture buffer hex length");
    Bytes result;
    result.reserve(text.size() / 2);
    for (std::size_t i = 0; i < text.size(); i += 2)
        result.push_back(static_cast<std::uint8_t>(16U * nibble(text[i]) + nibble(text[i + 1])));
    return result;
}

inline Buffers decode_buffers(const Json& object) {
    Buffers buffers;
    for (const auto& [name, value] : object.as_object()) {
        if (name.empty()) throw std::invalid_argument("empty capture buffer name");
        buffers.emplace_back(name, decode_hex(value.as_string()));
    }
    return buffers;
}

inline Json encode_buffers(const Buffers& buffers) {
    constexpr std::string_view digits = "0123456789abcdef";
    auto result = Json::object();
    std::set<std::string_view> names;
    for (const auto& [name, value] : buffers) {
        if (name.empty() || !names.insert(name).second) throw std::invalid_argument("invalid capture buffer name: " + name);
        std::string hex;
        for (const auto byte : value) {
            hex += digits[byte >> 4U];
            hex += digits[byte & 15U];
        }
        result.set(name, std::move(hex));
    }
    return result;
}
}  // namespace detail

inline Capture parse_capture(std::string_view text) {
    using detail::member;
    const auto root = serve::parse_json_unique(text);
    if (root.size() != 4 || member(root, "schema").as_string() != capture_schema)
        throw std::invalid_argument("unsupported capture schema or wrong member count");
    const auto& identity = member(root, "identity").as_string();
    if (identity.empty()) throw std::invalid_argument("missing capture identity");
    return {identity, detail::decode_buffers(member(root, "inputs")), detail::decode_buffers(member(root, "expected"))};
}

inline std::string write_capture(const Capture& capture) {
    if (capture.identity.empty()) throw std::invalid_argument("missing capture identity");
    auto root = serve::Json::object();
    root.set("schema", capture_schema);
    root.set("identity", capture.identity);
    root.set("inputs", detail::encode_buffers(capture.inputs));
    root.set("expected", detail::encode_buffers(capture.expected));
    auto output = root.dump();
    (void)serve::parse_json_unique(output);
    return output;
}

inline Capture read_capture(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open capture: " + path.string());
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return parse_capture(text);
}

}  // namespace strix::kit
