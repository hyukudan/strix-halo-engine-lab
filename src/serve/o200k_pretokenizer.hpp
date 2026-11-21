#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve::o200k {

inline constexpr std::uint32_t kInvalidBase = 0x110000U;

struct MmRange { std::uint32_t first, last; std::uint8_t value; };
struct MmDecomp { std::uint32_t cp, first, second; };
struct MmCompose { std::uint32_t first, second, composite; };
#include "o200k_unicode_tables.inc"

template <std::size_t N>
std::uint8_t range_value(const MmRange (&table)[N], std::uint32_t cp) {
    std::size_t lo = 0, hi = N;
    while (lo < hi) {
        const auto mid = (lo + hi) / 2U;
        if (table[mid].last < cp) lo = mid + 1U;
        else hi = mid;
    }
    return lo < N && table[lo].first <= cp ? table[lo].value : 0U;
}

inline constexpr std::string_view kPattern =
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// ---------------------------------------------------------------------------------------------
// Pre-tokenizer: Both supported Split layouts return the regex matches, alternatives in order:
//   A1 [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   A2 [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   A3 \p{N}{1,3}
//   A4  ?[^\s\p{L}\p{N}]+[\r\n/]*
//   A5 \s*[\r\n]+
//   A6 \s+(?!\S)
//   A7 \s+
// with backtracking (Oniguruma) semantics. Every character is covered by some alternative, so the
// matches tile the text.
constexpr std::uint8_t kUp = 1U, kLo = 2U, kL = 4U, kN = 8U, kWs = 16U;

inline bool white_space(std::uint32_t cp) {
    return (cp >= 0x09U && cp <= 0x0DU) || cp == 0x20U || cp == 0x85U || cp == 0xA0U || cp == 0x1680U ||
        (cp >= 0x2000U && cp <= 0x200AU) || cp == 0x2028U || cp == 0x2029U || cp == 0x202FU || cp == 0x205FU || cp == 0x3000U;
}

inline std::uint8_t char_flags(std::uint32_t cp) {
    if (cp >= kInvalidBase) return 0U;
    if (cp < 0x80U) {
        if ((cp >= 'a' && cp <= 'z')) return kLo | kL;
        if ((cp >= 'A' && cp <= 'Z')) return kUp | kL;
        if (cp >= '0' && cp <= '9') return kN;
        return white_space(cp) ? kWs : 0U;
    }
    return static_cast<std::uint8_t>(range_value(kMmClassRanges, cp) | (white_space(cp) ? kWs : 0U));
}

inline std::vector<std::pair<std::size_t, std::size_t>> split(std::span<const std::uint32_t> c) {
    const auto n = c.size();
    std::vector<std::uint8_t> f(n);
    for (std::size_t k = 0; k < n; ++k) f[k] = char_flags(c[k]);
    const auto up = [&](std::size_t k) { return k < n && (f[k] & kUp) != 0U; };
    const auto lo = [&](std::size_t k) { return k < n && (f[k] & kLo) != 0U; };
    const auto crlf = [&](std::size_t k) { return k < n && (c[k] == '\r' || c[k] == '\n'); };
    const auto prefix = [&](std::size_t k) { return k < n && !crlf(k) && (f[k] & (kL | kN)) == 0U; };
    const auto sym = [&](std::size_t k) { return k < n && (f[k] & (kWs | kL | kN)) == 0U; };
    const auto ws = [&](std::size_t k) { return k < n && (f[k] & kWs) != 0U; };
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)?  (case folding: also 'ſ U+017F for s)
    const auto contraction = [&](std::size_t e) -> std::size_t {
        if (e >= n || c[e] != '\'') return e;
        const auto fold = [&](std::size_t k) -> char {
            if (k >= n) return '\0';
            if (c[k] == 0x17FU) return 's';
            return c[k] < 0x80U ? static_cast<char>(c[k] | 0x20U) : '\0';
        };
        const char a = fold(e + 1U);
        const char b = fold(e + 2U);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return e + 2U;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return e + 3U;
        return e;
    };
    std::vector<std::pair<std::size_t, std::size_t>> pieces;
    std::size_t i = 0;
    while (i < n) {
        std::size_t end = 0;
        const std::size_t prefix_lengths[2] = {prefix(i) ? 1U : 0U, 0U};
        const std::size_t tries = prefix(i) ? 2U : 1U;
        // A1
        for (std::size_t t = 0; t < tries && !end; ++t) {
            const std::size_t j = i + prefix_lengths[t];
            std::size_t r = j;
            while (up(r)) ++r;
            for (std::size_t k = r + 1U; k-- > j;) {
                if (lo(k)) {
                    std::size_t e = k;
                    while (lo(e)) ++e;
                    end = contraction(e);
                    break;
                }
            }
        }
        // A2
        for (std::size_t t = 0; t < tries && !end; ++t) {
            const std::size_t j = i + prefix_lengths[t];
            if (!up(j)) continue;
            std::size_t e = j;
            while (up(e)) ++e;
            while (lo(e)) ++e;
            end = contraction(e);
        }
        // A3
        if (!end && (f[i] & kN)) {
            end = i;
            while (end < n && end - i < 3U && (f[end] & kN)) ++end;
        }
        // A4
        if (!end) {
            std::size_t start = SIZE_MAX;
            if (c[i] == ' ' && sym(i + 1U)) start = i + 1U;
            else if (sym(i)) start = i;
            if (start != SIZE_MAX) {
                end = start;
                while (sym(end)) ++end;
                while (end < n && (c[end] == '\r' || c[end] == '\n' || c[end] == '/')) ++end;
            }
        }
        // A5-A7
        if (!end && ws(i)) {
            std::size_t run = i;
            while (ws(run)) ++run;
            std::size_t last_newline = SIZE_MAX;
            for (std::size_t k = i; k < run; ++k) if (crlf(k)) last_newline = k;
            if (last_newline != SIZE_MAX) end = last_newline + 1U;
            else if (run == n) end = run;
            else if (run - i >= 2U) end = run - 1U;
            else end = run;
        }
        if (!end) end = i + 1U;  // unreachable: the alternatives cover every character
        pieces.emplace_back(i, end);
        i = end;
    }
    return pieces;
}

}  // namespace strix::serve::o200k
