#pragma once
#include <string>
#include <string_view>

namespace strix::decode_receipt {
// Escape the contents of a JSON string, retaining UTF-8 bytes. Callers provide
// the surrounding quotes. GGUF tensor names may contain quotes or controls.
inline std::string escape_json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (char character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (character == '"' || character == '\\') {
            result.push_back('\\');
            result.push_back(character);
        } else if (byte < 0x20U) {
            result += "\\u00";
            result.push_back(hex[byte >> 4U]);
            result.push_back(hex[byte & 15U]);
        } else
            result.push_back(character);
    }
    return result;
}
} // namespace strix::decode_receipt
