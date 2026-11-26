#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace strix::serve::tokenizer_utf8 {

// Byte-level BPE maps printable bytes to their same-valued Unicode code point
// and assigns the remaining bytes consecutive code points starting at U+0100.
inline constexpr auto kByteBpeCodePoints = [] {
    std::array<std::uint32_t, 256> points{};
    std::uint32_t extra = 0x100U;
    for (std::uint32_t byte = 0; byte < points.size(); ++byte) {
        const bool direct = (byte >= 0x21U && byte <= 0x7EU) || (byte >= 0xA1U && byte <= 0xACU) || byte >= 0xAEU;
        points[byte] = direct ? byte : extra++;
    }
    return points;
}();

inline std::array<std::string, 256> byte_bpe_alphabet() {
    std::array<std::string, 256> symbols;
    for (std::size_t byte = 0; byte < symbols.size(); ++byte) {
        const auto cp = kByteBpeCodePoints[byte];
        if (cp < 0x80U) {
            symbols[byte].push_back(static_cast<char>(cp));
        } else {
            symbols[byte].push_back(static_cast<char>(0xC0U | (cp >> 6U)));
            symbols[byte].push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        }
    }
    return symbols;
}

// Rust String::from_utf8_lossy: every maximal invalid prefix becomes one U+FFFD.
inline void append_lossy(std::string& out, std::string_view bytes) {
    std::size_t i = 0;
    const auto n = bytes.size();
    const auto at = [&](std::size_t k) { return static_cast<unsigned char>(bytes[k]); };
    const auto cont = [&](std::size_t k) { return k < n && (at(k) & 0xC0U) == 0x80U; };
    while (i < n) {
        const auto b0 = at(i);
        if (b0 < 0x80U) { out.push_back(static_cast<char>(b0)); ++i; continue; }
        std::size_t width = b0 >= 0xC2U && b0 <= 0xDFU ? 2U : b0 >= 0xE0U && b0 <= 0xEFU ? 3U : b0 >= 0xF0U && b0 <= 0xF4U ? 4U : 0U;
        std::size_t good = 1;  // bytes of the sequence verified so far
        bool ok = width != 0U;
        if (ok && width >= 3U) {
            const bool second_ok = i + 1U < n && [&] {
                const auto b1 = at(i + 1U);
                if (width == 3U) {
                    if (b0 == 0xE0U) return b1 >= 0xA0U && b1 <= 0xBFU;
                    if (b0 == 0xEDU) return b1 >= 0x80U && b1 <= 0x9FU;
                    return b1 >= 0x80U && b1 <= 0xBFU;
                }
                if (b0 == 0xF0U) return b1 >= 0x90U && b1 <= 0xBFU;
                if (b0 == 0xF4U) return b1 >= 0x80U && b1 <= 0x8FU;
                return b1 >= 0x80U && b1 <= 0xBFU;
            }();
            if (second_ok) good = 2; else ok = false;
        }
        while (ok && good < width) {
            if (cont(i + good)) ++good;
            else ok = false;
        }
        if (ok) { out.append(bytes.substr(i, width)); i += width; }
        else { out += "\xEF\xBF\xBD"; i += good; }
    }
}

}  // namespace strix::serve::tokenizer_utf8
