#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/tokenizer_limits.hpp"
#include "o200k_pretokenizer.hpp"
#include "tokenizer_utf8.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace strix::serve {
namespace {

struct UnicodeRange { std::uint32_t first, last; std::uint8_t flags; };
#include "unicode_tables.inc"

constexpr std::uint8_t kLetter = 1U, kMark = 2U, kNumber = 4U, kPunctuation = 8U, kSymbol = 16U, kDecimal = 32U;

std::uint8_t category(std::uint32_t cp) {
    std::size_t lo = 0, hi = std::size(kUnicodeRanges);
    while (lo < hi) {
        const auto mid = (lo + hi) / 2U;
        if (kUnicodeRanges[mid].last < cp) lo = mid + 1U;
        else hi = mid;
    }
    return lo < std::size(kUnicodeRanges) && kUnicodeRanges[lo].first <= cp ? kUnicodeRanges[lo].flags : 0U;
}
// Unicode White_Space.
bool white_space(std::uint32_t cp) {
    return (cp >= 0x09U && cp <= 0x0DU) || cp == 0x20U || cp == 0x85U || cp == 0xA0U || cp == 0x1680U ||
        (cp >= 0x2000U && cp <= 0x200AU) || cp == 0x2028U || cp == 0x2029U || cp == 0x202FU || cp == 0x205FU || cp == 0x3000U;
}

// Code points of a UTF-8 string with their byte offsets; invalid bytes decode as themselves
// (one code point per byte, flagged invalid) so every input byte belongs to exactly one piece.
struct CodePoints {
    std::vector<std::uint32_t> cp;
    std::vector<std::size_t> offset; // size cp.size() + 1
};
CodePoints decode_utf8(std::string_view text, bool opaque_invalid = false) {
    CodePoints out;
    out.cp.reserve(text.size());
    out.offset.reserve(text.size() + 1U);
    std::size_t i = 0;
    while (i < text.size()) {
        out.offset.push_back(i);
        const auto b0 = static_cast<unsigned char>(text[i]);
        std::uint32_t cp = b0;
        std::size_t length = 1;
        if (b0 >= 0xC2U && b0 <= 0xDFU) length = 2;
        else if (b0 >= 0xE0U && b0 <= 0xEFU) length = 3;
        else if (b0 >= 0xF0U && b0 <= 0xF4U) length = 4;
        if (length > 1U) {
            bool valid = i + length <= text.size();
            std::uint32_t value = length == 2U ? (b0 & 0x1FU) : length == 3U ? (b0 & 0x0FU) : (b0 & 0x07U);
            for (std::size_t k = 1; valid && k < length; ++k) {
                const auto b = static_cast<unsigned char>(text[i + k]);
                if ((b & 0xC0U) != 0x80U) valid = false;
                value = (value << 6U) | (b & 0x3FU);
            }
            if (valid && ((length == 3U && (value < 0x800U || (value >= 0xD800U && value <= 0xDFFFU))) ||
                          (length == 4U && (value < 0x10000U || value > 0x10FFFFU)))) valid = false;
            if (valid) cp = value; else length = 1;
        }
        out.cp.push_back(opaque_invalid && length == 1U && b0 >= 0x80U ? o200k::kInvalidBase + b0 : cp);
        i += length;
    }
    out.offset.push_back(text.size());
    return out;
}

// A pre-tokenizer is a rule: which splitting scheme runs and the few facts that differ between users of one scheme.
// Rules are chosen by the tokenizer's `pre` string; no code path below names a model.
enum class SplitScheme : std::uint8_t {
    word_runs,         // contractions, one optional lead symbol + word run, digit runs, symbol runs, whitespace runs
    isolating_splits,  // three isolating splits (digit groups, word/symbol classes, whitespace)
    right_aligned_decimal, // isolated decimal runs, split into groups of three from the right
    class_spans,       // Unicode class spans with case-aware words (o200k::split)
};
struct PreSplitRule {
    std::string_view key;          // accepted `pre` string
    SplitScheme scheme;
    bool marks_join_words;         // word_runs: a combining mark continues a word
    std::uint8_t digit_run_max;    // word_runs: digits grouped per piece (1 = one digit)
    bool whole_added_tokens;       // added tokens match whole in text; a piece that is a vocabulary entry is emitted as is
    bool retype_added_block;       // GGUF types the trailing added block as normal: retype it user-defined, scan its specials
    bool opaque_invalid_bytes;     // invalid UTF-8 stays opaque on encode and decode is lossy-repaired
};
constexpr PreSplitRule kPreSplitRules[] = {
    {"qwen35", SplitScheme::word_runs, true, 1U, false, false, false},
    {"joyai-llm", SplitScheme::isolating_splits, true, 1U, false, true, false},
    {"deepseek-v3", SplitScheme::isolating_splits, true, 1U, false, true, false},
    {"hy_v4", SplitScheme::isolating_splits, true, 1U, false, true, false},
    {"glm4", SplitScheme::word_runs, false, 3U, true, false, false},
    {"decimal-triplets", SplitScheme::right_aligned_decimal, true, 3U, false, false, false},
    {"o200k", SplitScheme::class_spans, true, 1U, true, false, true},
};
const PreSplitRule* find_pre_split_rule(std::string_view pre) {
    for (const auto& rule : kPreSplitRules) if (rule.key == pre) return &rule;
    return nullptr;
}

// The qwen35 pattern:
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Alternatives are tried in order at each position, each with its greedy/backtracking meaning.
// The GLM-4/5 tokenizer.json pattern is the same alternatives with words of \p{L} only (a mark is a symbol) and
// numbers of up to three digits:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
std::vector<std::pair<std::size_t, std::size_t>> pre_split(const CodePoints& text, const PreSplitRule& rule) {
    const auto& c = text.cp;
    const auto n = c.size();
    const std::uint32_t word = rule.marks_join_words ? (kLetter | kMark) : kLetter;
    const auto letter_or_mark = [&](std::size_t k) { return k < n && (category(c[k]) & word) != 0U; };
    const auto ws = [&](std::size_t k) { return k < n && white_space(c[k]); };
    const auto crlf = [&](std::size_t k) { return k < n && (c[k] == '\r' || c[k] == '\n'); };
    const auto lower = [&](std::size_t k) { return k < n && c[k] < 0x80U ? static_cast<char>(c[k] | 0x20U) : '\0'; };
    std::vector<std::pair<std::size_t, std::size_t>> pieces;
    std::size_t i = 0;
    while (i < n) {
        std::size_t end = 0;
        // 1. contractions
        if (c[i] == '\'') {
            const char a = lower(i + 1U), b = lower(i + 2U);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) end = i + 3U;
            else if (a == 's' || a == 't' || a == 'm' || a == 'd') end = i + 2U;
        }
        // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!end) {
            const auto cat = category(c[i]);
            if (c[i] != '\r' && c[i] != '\n' && !(cat & (kLetter | kNumber)) && letter_or_mark(i + 1U)) {
                end = i + 1U;
                while (letter_or_mark(end)) ++end;
            } else if (letter_or_mark(i)) {
                end = i;
                while (letter_or_mark(end)) ++end;
            }
        }
        // 3. \p{N}{1,digit_run_max}
        if (!end && (category(c[i]) & kNumber)) {
            end = i + 1U;
            while (end < n && end < i + rule.digit_run_max && (category(c[end]) & kNumber)) ++end;
        }
        // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (!end) {
            const auto symbol = [&](std::size_t k) { return k < n && !white_space(c[k]) && !(category(c[k]) & (word | kNumber)); };
            std::size_t start = 0;
            if (c[i] == ' ' && symbol(i + 1U)) start = i + 1U;
            else if (symbol(i)) start = i;
            else start = SIZE_MAX;
            if (start != SIZE_MAX) {
                end = start;
                while (symbol(end)) ++end;
                while (crlf(end)) ++end;
            }
        }
        // 5-7. whitespace runs
        if (!end && ws(i)) {
            std::size_t run = i;
            while (ws(run)) ++run;
            std::size_t last_newline = SIZE_MAX;
            for (std::size_t k = i; k < run; ++k) if (crlf(k)) last_newline = k;
            if (last_newline != SIZE_MAX) end = last_newline + 1U;           // \s*[\r\n]+
            else if (run == n) end = run;                                     // \s+(?!\S) at the end
            else if (run - i >= 2U) end = run - 1U;                           // \s+(?!\S) leaves one
            else end = run;                                                   // \s+
        }
        if (!end) end = i + 1U; // unreachable with this pattern; keeps progress on any input
        pieces.emplace_back(text.offset[i], text.offset[end]);
        i = end;
    }
    return pieces;
}

// The DeepSeek V3/V4 pre-tokenizer (GGUF `joyai-llm` / `deepseek-v3` / `hy_v4`): a sequence of three isolating splits, each
// applied to every piece the previous one produced (matches and the text between them are both pieces):
//   1. \p{N}{1,3}
//   2. [\x{4E00}-\x{9FA5}\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+
//   3. [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*
//      |\s*[\r\n]+|\s+(?!\S)|\s+
// In pattern 3 the alternatives are tried in order at each position; a piece's end is its text's end.
std::vector<std::pair<std::size_t, std::size_t>> pre_split_isolating(const CodePoints& text, bool right_aligned) {
    const auto& c = text.cp;
    const auto n = c.size();
    using Range = std::pair<std::size_t, std::size_t>;
    // isolates the maximal runs (at most `limit` long) of code points with `in`, inside every range of `pieces`
    const auto isolate = [&](const std::vector<Range>& pieces, const auto& in, std::size_t limit) {
        std::vector<Range> out;
        for (const auto& [a, b] : pieces) {
            std::size_t gap = a, i = a;
            while (i < b) {
                if (!in(c[i])) { ++i; continue; }
                if (gap < i) out.emplace_back(gap, i);
                std::size_t e = i;
                while (e < b && e - i < limit && in(c[e])) ++e;
                out.emplace_back(i, e);
                i = gap = e;
            }
            if (gap < b) out.emplace_back(gap, b);
        }
        return out;
    };
    std::vector<Range> pieces;
    if (n != 0U) pieces.emplace_back(0U, n);
    if (right_aligned) {
        const auto decimal = [](std::uint32_t cp) { return (category(cp) & kDecimal) != 0U; };
        const auto numbers = isolate(pieces, [](std::uint32_t cp) { return (category(cp) & kNumber) != 0U; }, SIZE_MAX);
        const auto runs = isolate(numbers, decimal, SIZE_MAX);
        pieces.clear();
        for (const auto& [a, b] : runs) {
            if (!decimal(c[a])) {
                pieces.emplace_back(a, b);
                continue;
            }
            // The serialized pipeline first limits the regex span to 510 digits. Since
            // 510 is divisible by three, only the leftmost triplet can be partial.
            auto start = a;
            if (const auto remainder = (b - a) % 3U; remainder != 0U) {
                pieces.emplace_back(a, a + remainder);
                start += remainder;
            }
            for (; start < b; start += 3U) pieces.emplace_back(start, start + 3U);
        }
    } else {
        pieces = isolate(pieces, [](std::uint32_t cp) { return (category(cp) & kNumber) != 0U; }, 3U);
    }
    pieces = isolate(pieces, [](std::uint32_t cp) {
        return (cp >= 0x4E00U && cp <= 0x9FA5U) || (cp >= 0x3040U && cp <= 0x309FU) || (cp >= 0x30A0U && cp <= 0x30FFU);
    }, SIZE_MAX);
    std::vector<std::pair<std::size_t, std::size_t>> out;
    for (const auto& [a, b] : pieces) {
        const auto letter_or_mark = [&](std::size_t k) { return k < b && (category(c[k]) & (kLetter | kMark)) != 0U; };
        const auto punct_or_symbol = [&](std::size_t k) { return k < b && (category(c[k]) & (kPunctuation | kSymbol)) != 0U; };
        const auto ws = [&](std::size_t k) { return k < b && white_space(c[k]); };
        const auto crlf = [&](std::size_t k) { return k < b && (c[k] == '\r' || c[k] == '\n'); };
        const auto ascii_letter = [&](std::size_t k) { return k < b && ((c[k] >= 'a' && c[k] <= 'z') || (c[k] >= 'A' && c[k] <= 'Z')); };
        const auto ascii_punct = [&](std::size_t k) {
            constexpr std::string_view set = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
            return k < b && c[k] < 0x80U && set.find(static_cast<char>(c[k])) != std::string_view::npos;
        };
        const auto match = [&](std::size_t i) -> std::size_t {  // end of the match at i, 0 when none
            // 1. one ASCII punctuation character and ASCII letters
            if (ascii_punct(i) && ascii_letter(i + 1U)) {
                std::size_t e = i + 1U;
                while (ascii_letter(e)) ++e;
                return e;
            }
            // 2. [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
            if (!crlf(i) && !(category(c[i]) & (kLetter | kPunctuation | kSymbol)) && letter_or_mark(i + 1U)) {
                std::size_t e = i + 1U;
                while (letter_or_mark(e)) ++e;
                return e;
            }
            if (letter_or_mark(i)) {
                std::size_t e = i;
                while (letter_or_mark(e)) ++e;
                return e;
            }
            // 3.  ?[\p{P}\p{S}]+[\r\n]*
            std::size_t start = SIZE_MAX;
            if (c[i] == ' ' && punct_or_symbol(i + 1U)) start = i + 1U;
            else if (punct_or_symbol(i)) start = i;
            if (start != SIZE_MAX) {
                std::size_t e = start;
                while (punct_or_symbol(e)) ++e;
                while (crlf(e)) ++e;
                return e;
            }
            // 4-6. whitespace runs
            if (ws(i)) {
                std::size_t run = i;
                while (ws(run)) ++run;
                std::size_t last_newline = SIZE_MAX;
                for (std::size_t k = i; k < run; ++k) if (crlf(k)) last_newline = k;
                if (last_newline != SIZE_MAX) return last_newline + 1U;  // \s*[\r\n]+
                if (run == b) return run;                               // \s+(?!\S) at the piece end
                if (run - i >= 2U) return run - 1U;                     // \s+(?!\S) leaves one
                return run;                                             // \s+
            }
            return 0U;
        };
        std::size_t gap = a, i = a;
        while (i < b) {
            const std::size_t e = match(i);
            if (e == 0U) { ++i; continue; }
            if (gap < i) out.emplace_back(text.offset[gap], text.offset[i]);
            out.emplace_back(text.offset[i], text.offset[e]);
            i = gap = e;
        }
        if (gap < b) out.emplace_back(text.offset[gap], text.offset[b]);
    }
    return out;
}

std::vector<std::pair<std::size_t, std::size_t>> pre_ranges(const CodePoints& text, const PreSplitRule& rule) {
    if (rule.scheme == SplitScheme::isolating_splits || rule.scheme == SplitScheme::right_aligned_decimal)
        return pre_split_isolating(text, rule.scheme == SplitScheme::right_aligned_decimal);
    if (rule.scheme == SplitScheme::class_spans) {
        auto spans = o200k::split(text.cp);
        for (auto& [begin, end] : spans) {
            begin = text.offset[begin];
            end = text.offset[end];
        }
        return spans;
    }
    return pre_split(text, rule);
}

bool field_is_string(const Json& object, std::string_view name, std::string_view expected) {
    const auto* field = object.find(name);
    return field && field->is_string() && field->as_string() == expected;
}

bool field_is_bool(const Json& object, std::string_view name, bool expected) {
    const auto* field = object.find(name);
    return field && field->is_bool() && field->as_bool() == expected;
}

bool field_is_unset(const Json& object, std::string_view name) {
    const auto* field = object.find(name);
    return !field || field->is_null();
}

bool plain_hf_byte_bpe_layout(const Json& root, const Json& model, const Json& pre, const Json& split, const Json& bytelevel) {
    const auto* decoder = root.find("decoder");
    const auto* processor = root.find("post_processor");
    return field_is_string(pre, "type", "Sequence") && field_is_string(split, "type", "Split") &&
           field_is_string(split, "behavior", "Isolated") && field_is_bool(split, "invert", false) &&
           field_is_bool(bytelevel, "add_prefix_space", false) && field_is_bool(bytelevel, "use_regex", false) &&
           decoder && field_is_string(*decoder, "type", "ByteLevel") &&
           (!processor || processor->is_null() || field_is_string(*processor, "type", "ByteLevel")) &&
           field_is_unset(root, "truncation") && field_is_unset(root, "padding") &&
           field_is_unset(model, "dropout") && field_is_unset(model, "unk_token") &&
           field_is_unset(model, "continuing_subword_prefix") && field_is_unset(model, "end_of_word_suffix") &&
           (!model.find("byte_fallback") || field_is_bool(model, "byte_fallback", false)) &&
           (!model.find("fuse_unk") || field_is_bool(model, "fuse_unk", false));
}


bool isolated_regex(const Json& split, std::string_view expression) {
    const auto* pattern = split.find("pattern");
    return field_is_string(split, "type", "Split") && field_is_string(split, "behavior", "Isolated") &&
           field_is_bool(split, "invert", false) && pattern && field_is_string(*pattern, "Regex", expression);
}

bool decimal_triplet_layout(const Json::Array& stages) {
    constexpr auto cjk = u8"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+";
    const std::string_view cjk_pattern(reinterpret_cast<const char*>(cjk));
    constexpr std::string_view classes =
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
        "|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
        "| ?[\\p{P}\\p{S}]+[\r\n]*"
        "|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
    // Recognize the serialized schema as a unit. These are data signatures for
    // the shared scanner, which does not evaluate regexes from an asset.
    return stages.size() == 7U && field_is_string(stages[0], "type", "Digits") &&
           field_is_bool(stages[0], "individual_digits", false) &&
           isolated_regex(stages[1], R"(\p{Nd}{1,510}(?=(?>\p{Nd}{510})*(?:\P{Nd}|$))|\G\p{Nd}{510})") &&
           isolated_regex(stages[2], R"(\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z))") &&
           isolated_regex(stages[3], R"(\A\p{Nd}{3}|\G\p{Nd}{3})") &&
           isolated_regex(stages[4], cjk_pattern) &&
           isolated_regex(stages[5], classes) &&
           field_is_string(stages[6], "type", "ByteLevel") &&
           field_is_bool(stages[6], "add_prefix_space", false) && field_is_bool(stages[6], "use_regex", false);
}

// encode() is the raw BPE API. A supported single-prefix template is validated
// at load time, but protocol renderers own insertion of framing IDs.
bool raw_prefix_template(const Json& root, const Json& vocab) {
    const auto* processor = root.find("post_processor");
    if (!processor || processor->is_null()) return true;
    if (!field_is_string(*processor, "type", "TemplateProcessing")) return false;
    const auto* single = processor->find("single");
    const auto* pair = processor->find("pair");
    const auto* specials = processor->find("special_tokens");
    if (!single || !single->is_array() || single->size() != 2U || !pair || !pair->is_array() ||
        pair->size() != 4U || !specials || !specials->is_object() || specials->size() != 1U) return false;
    const auto& [prefix, entry] = specials->as_object().front();
    const auto* ids = entry.find("ids");
    const auto* tokens = entry.find("tokens");
    const auto* vocab_id = vocab.find(prefix);
    if (!field_is_string(entry, "id", prefix) || !vocab_id || !ids || !ids->is_array() || ids->size() != 1U ||
        ids->as_array().front().as_int() != vocab_id->as_int() ||
        !tokens || !tokens->is_array() || tokens->size() != 1U || tokens->as_array().front().as_string() != prefix)
        return false;
    const auto matches = [](const Json& item, std::string_view key, std::string_view id, std::int64_t type) {
        const auto* body = item.find(key);
        const auto* type_id = body ? body->find("type_id") : nullptr;
        return body && field_is_string(*body, "id", id) && type_id && type_id->as_int() == type;
    };
    const auto& a = single->as_array();
    const auto& b = pair->as_array();
    return matches(a[0], "SpecialToken", prefix, 0) && matches(a[1], "Sequence", "A", 0) &&
           matches(b[0], "SpecialToken", prefix, 0) && matches(b[1], "Sequence", "A", 0) &&
           matches(b[2], "SpecialToken", prefix, 1) && matches(b[3], "Sequence", "B", 1);
}

// GGUF header metadata (only the tokenizer keys are kept).
struct GgufTokenizer {
    std::vector<std::string> tokens, merges;
    std::vector<std::int32_t> types;
    std::string model, pre;
    std::uint32_t eos = UINT32_MAX;
};
constexpr std::uint64_t kMaxGgufTokenizerEntries = 1'000'000;  // as gguf::Limits::max_metadata_entries
constexpr std::uint64_t kMaxGgufTokenizerMerges = std::uint64_t{1} << 24U;
GgufTokenizer read_gguf_tokenizer(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("tokenizer: cannot open " + path);
    std::error_code size_error;
    const auto file_bytes = std::filesystem::file_size(path, size_error);
    if (size_error) throw std::runtime_error("tokenizer: cannot size " + path);
    // Bytes after the read position: every array count is checked against it before anything is sized from the count.
    const auto remaining = [&]() -> std::uint64_t {
        const auto at = file.tellg();
        if (at < 0 || static_cast<std::uint64_t>(at) > file_bytes) throw std::runtime_error("tokenizer: GGUF read position is invalid");
        return file_bytes - static_cast<std::uint64_t>(at);
    };
    const auto read = [&](void* p, std::size_t n) {
        if (!file.read(static_cast<char*>(p), static_cast<std::streamsize>(n))) throw std::runtime_error("tokenizer: truncated GGUF header");
    };
    const auto u32 = [&] { std::uint32_t v; read(&v, 4); return v; };
    const auto u64 = [&] { std::uint64_t v; read(&v, 8); return v; };
    const auto string = [&] {
        const auto n = u64();
        if (n > (64ULL << 20U)) throw std::runtime_error("tokenizer: GGUF string too long");
        std::string s(static_cast<std::size_t>(n), '\0');
        if (n) read(s.data(), s.size());
        return s;
    };
    static constexpr std::array<std::size_t, 13> sizes{1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    char magic[4];
    read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) throw std::runtime_error("tokenizer: not a GGUF file");
    if (u32() < 2U) throw std::runtime_error("tokenizer: GGUF version below 2");
    (void)u64();
    const auto entries = u64();
    if (entries > kMaxGgufTokenizerEntries) throw std::runtime_error("tokenizer: GGUF metadata entry count is out of range");
    GgufTokenizer out;
    for (std::uint64_t e = 0; e < entries; ++e) {
        const auto key = string();
        const auto type = u32();
        if (type == 8U) {
            auto value = string();
            if (key == "tokenizer.ggml.model") out.model = std::move(value);
            else if (key == "tokenizer.ggml.pre") out.pre = std::move(value);
            continue;
        }
        if (type == 9U) {
            const auto item = u32();
            const auto count = u64();
            // A string takes at least its 8-byte length; any other element at least one byte.
            const std::uint64_t element_bytes = item == 8U ? 8U : item < sizes.size() ? sizes[item] : 0U;
            if (element_bytes != 0U && count > remaining() / element_bytes)
                throw std::runtime_error("tokenizer: GGUF array count exceeds the file");
            if (item == 8U) {
                std::vector<std::string>* target = key == "tokenizer.ggml.tokens" ? &out.tokens :
                    key == "tokenizer.ggml.merges" ? &out.merges : nullptr;
                if (target) {
                    const auto limit = key == "tokenizer.ggml.tokens" ? static_cast<std::uint64_t>(kMaxTokenizerVocabId) + 1U
                                                                      : kMaxGgufTokenizerMerges;
                    if (count > limit) throw std::runtime_error("tokenizer: GGUF array count is out of range");
                    target->reserve(static_cast<std::size_t>(count));
                }
                for (std::uint64_t k = 0; k < count; ++k) {
                    auto value = string();
                    if (target) target->push_back(std::move(value));
                }
                continue;
            }
            if (item >= sizes.size() || !sizes[item]) throw std::runtime_error("tokenizer: unsupported GGUF array element type");
            if (key == "tokenizer.ggml.token_type" && (item == 5U || item == 4U)) {
                if (count > static_cast<std::uint64_t>(kMaxTokenizerVocabId) + 1U)
                    throw std::runtime_error("tokenizer: GGUF array count is out of range");
                out.types.resize(static_cast<std::size_t>(count));
                read(out.types.data(), out.types.size() * 4U);
                continue;
            }
            // count * element size <= remaining (checked above), so the product fits and is not negative.
            if (!file.seekg(static_cast<std::streamoff>(sizes[item] * count), std::ios::cur))
                throw std::runtime_error("tokenizer: GGUF array skip failed");
            continue;
        }
        if (type >= sizes.size() || !sizes[type]) throw std::runtime_error("tokenizer: unsupported GGUF metadata type");
        std::uint64_t value = 0;
        read(&value, sizes[type]);
        if (key == "tokenizer.ggml.eos_token_id") out.eos = static_cast<std::uint32_t>(value);
    }
    return out;
}

}  // namespace

struct BpeTokenizer::Impl {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    std::unordered_map<std::string, std::uint32_t> ids;
    std::unordered_map<std::string, std::uint32_t> ranks;
    std::vector<std::uint32_t> specials; // control and user-defined, longest text first
    // first bytes at which a special token may start: '<' and '[' (qwen35), every special's first byte (DeepSeek,
    // whose added tokens such as the DSML markers also start inside text)
    std::array<bool, 256> special_start{};
    std::array<std::string, 256> byte_unicode = tokenizer_utf8::byte_bpe_alphabet();
    std::unordered_map<std::uint32_t, unsigned char> unicode_byte;
    std::string pre;
    const PreSplitRule* rule = &kPreSplitRules[0];
    // tokenizer.json `ignore_merges`: a pre-token piece that is itself a vocabulary entry is emitted as that ID
    bool ignore_merges = false;
    std::uint32_t eos = UINT32_MAX;
    // Piece cache: memoises bpe(), which is a pure function of the piece. Encoders on different threads read
    // it at the same time, so a hit takes no lock and writes no shared memory: the table is insert-only with
    // immutable entries published by compare-and-swap. When it reaches its entry limit a new table replaces
    // it; readers retain ownership, and a segment refreshes its handle on a full-table miss.
    // The limit is per table (concurrent inserts may overshoot); retired tables live until readers leave.
    struct PieceTable {
        struct Entry {
            std::size_t hash;
            std::string piece;
            std::vector<std::uint32_t> ids;
        };
        static constexpr std::size_t slots = std::size_t{1} << 19U, limit = 200000U; // about 38 % at the threshold
        std::unique_ptr<std::atomic<Entry*>[]> table{new std::atomic<Entry*>[slots]()};
        std::atomic<std::size_t> size{0};
        PieceTable() = default;
        PieceTable(const PieceTable&) = delete;
        PieceTable& operator=(const PieceTable&) = delete;
        ~PieceTable() {
            for (std::size_t i = 0; i < slots; ++i) delete table[i].load(std::memory_order_relaxed);
        }
        const Entry* find(std::size_t hash, std::string_view piece) const {
            for (std::size_t i = hash & (slots - 1U);; i = (i + 1U) & (slots - 1U)) {
                const Entry* entry = table[i].load(std::memory_order_acquire);
                if (!entry) return nullptr;
                if (entry->hash == hash && entry->piece == piece) return entry;
            }
        }
        // False when the table is full: the caller replaces it.
        bool insert(std::size_t hash, std::string_view piece, const std::vector<std::uint32_t>& piece_ids) {
            if (size.load(std::memory_order_relaxed) >= limit) return false;
            auto entry = std::make_unique<Entry>(Entry{hash, std::string(piece), piece_ids});
            for (std::size_t i = hash & (slots - 1U);; i = (i + 1U) & (slots - 1U)) {
                Entry* seen = table[i].load(std::memory_order_acquire);
                if (!seen && table[i].compare_exchange_strong(seen, entry.get(), std::memory_order_release,
                                                              std::memory_order_acquire)) {
                    (void)entry.release();
                    size.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                if (seen->hash == hash && seen->piece == piece) return true; // another encoder published it first
            }
        }
    };
    mutable std::mutex cache_mutex; // guards the pointer at segment start and rollover
    mutable std::shared_ptr<PieceTable> cache = std::make_shared<PieceTable>();

    std::shared_ptr<PieceTable> piece_table() const {
        const std::lock_guard lock(cache_mutex);
        return cache;
    }
    void bpe(std::string_view piece, std::vector<std::uint32_t>& out, std::shared_ptr<PieceTable>& pieces) const {
        const std::size_t hash = std::hash<std::string_view>{}(piece);
        if (const auto* hit = pieces->find(hash, piece)) {
            out.insert(out.end(), hit->ids.begin(), hit->ids.end());
            return;
        }
#ifdef STRIX_BPE_CACHE_TEST_MISS
        STRIX_BPE_CACHE_TEST_MISS(pieces); // standalone host regression observer; absent in production
#endif
        std::vector<std::string> symbols;
        symbols.reserve(piece.size());
        for (const char ch : piece) symbols.push_back(byte_unicode[static_cast<unsigned char>(ch)]);
        if (ignore_merges && symbols.size() > 1U) {
            std::string whole;
            for (const auto& symbol : symbols) whole += symbol;
            if (const auto id = ids.find(whole); id != ids.end()) symbols.assign(1U, std::move(whole));
        }
        std::string key;
        while (symbols.size() > 1U) {
            std::uint32_t best = UINT32_MAX;
            for (std::size_t k = 0; k + 1U < symbols.size(); ++k) {
                key.assign(symbols[k]); key.push_back(' '); key += symbols[k + 1U];
                if (const auto r = ranks.find(key); r != ranks.end() && r->second < best) best = r->second;
            }
            if (best == UINT32_MAX) break;
            std::vector<std::string> merged;
            merged.reserve(symbols.size());
            for (std::size_t k = 0; k < symbols.size();) {
                if (k + 1U < symbols.size()) {
                    key.assign(symbols[k]); key.push_back(' '); key += symbols[k + 1U];
                    if (const auto r = ranks.find(key); r != ranks.end() && r->second == best) {
                        merged.push_back(symbols[k] + symbols[k + 1U]);
                        k += 2U;
                        continue;
                    }
                }
                merged.push_back(std::move(symbols[k]));
                ++k;
            }
            symbols = std::move(merged);
        }
        std::vector<std::uint32_t> result;
        result.reserve(symbols.size());
        for (const auto& symbol : symbols) {
            const auto id = ids.find(symbol);
            if (id != ids.end()) {
                if (rule->scheme == SplitScheme::right_aligned_decimal &&
                    static_cast<TokenType>(types[id->second]) != TokenType::normal)
                    throw std::runtime_error("tokenizer: ordinary BPE reached an added token");
                result.push_back(id->second);
                continue;
            }
            throw std::runtime_error("tokenizer: BPE symbol is not in the vocabulary");
        }
        while (!pieces->insert(hash, piece, result)) {
            const std::lock_guard lock(cache_mutex);
            if (cache == pieces) cache = std::make_shared<PieceTable>();
            pieces = cache; // another segment may already have replaced it; retry the computed result
        }
        out.insert(out.end(), result.begin(), result.end());
    }
    void encode_plain(std::string_view text, std::vector<std::uint32_t>& out) const {
        const auto points = decode_utf8(text, rule->opaque_invalid_bytes);
        std::shared_ptr<PieceTable> pieces = piece_table();
        for (const auto& [begin, end] : pre_ranges(points, *rule))
            bpe(text.substr(begin, end - begin), out, pieces);
    }
};

BpeTokenizer::BpeTokenizer() : impl_(std::make_unique<Impl>()) {}
BpeTokenizer::~BpeTokenizer() = default;

std::shared_ptr<const BpeTokenizer> BpeTokenizer::from_parts(std::vector<std::string> tokens, std::vector<std::int32_t> types,
    std::span<const std::string> merges, std::string pre) {
    const auto* rule = find_pre_split_rule(pre);
    if (!rule)
        throw std::invalid_argument("tokenizer: unsupported byte-level BPE pre-tokenizer " + pre);
    if (tokens.empty() || types.size() != tokens.size()) throw std::invalid_argument("tokenizer: token and type tables differ");
    std::shared_ptr<BpeTokenizer> out(new BpeTokenizer());
    auto& impl = *out->impl_;
    impl.pre = std::move(pre);
    impl.rule = rule;
    impl.ignore_merges = rule->whole_added_tokens;
    impl.tokens = std::move(tokens);
    impl.types = std::move(types);
    if (rule->retype_added_block) {
        // DeepSeek GGUFs type only the special added tokens as control; the others (<｜User｜>, <think>, the DSML
        // markers, ...) are typed normal although the reference tokenizer matches every added token as a whole. The
        // added tokens are the block from the first control token past the byte-level base to the vocabulary's end:
        // they become user-defined (matched before pre-tokenization, decoded as their text).
        std::size_t first_added = impl.tokens.size();
        for (std::size_t i = 256; i < impl.types.size(); ++i)
            if (static_cast<TokenType>(impl.types[i]) == TokenType::control) { first_added = i; break; }
        for (std::size_t i = first_added; i < impl.types.size(); ++i)
            if (static_cast<TokenType>(impl.types[i]) == TokenType::normal) impl.types[i] = static_cast<std::int32_t>(TokenType::user_defined);
    }
    impl.ids.reserve(impl.tokens.size());
    for (std::size_t i = 0; i < impl.tokens.size(); ++i) {
        impl.ids.emplace(impl.tokens[i], static_cast<std::uint32_t>(i));
        const auto t = static_cast<TokenType>(impl.types[i]);
        if (t == TokenType::control || t == TokenType::user_defined) impl.specials.push_back(static_cast<std::uint32_t>(i));
    }
    std::stable_sort(impl.specials.begin(), impl.specials.end(),
        [&](auto a, auto b) { return impl.tokens[a].size() > impl.tokens[b].size(); });
    impl.ranks.reserve(merges.size());
    for (std::size_t r = 0; r < merges.size(); ++r) impl.ranks.emplace(merges[r], static_cast<std::uint32_t>(r));
    impl.special_start['<'] = impl.special_start['['] = true;
    if (rule->retype_added_block || rule->whole_added_tokens)
        for (const auto id : impl.specials)
            if (!impl.tokens[id].empty()) impl.special_start[static_cast<unsigned char>(impl.tokens[id][0])] = true;
    for (std::size_t b = 0; b < 256U; ++b) {
        const auto points = decode_utf8(impl.byte_unicode[b]);
        impl.unicode_byte.emplace(points.cp.at(0), static_cast<unsigned char>(b));
    }
    return out;
}

std::shared_ptr<const BpeTokenizer> BpeTokenizer::load_hf_json(const std::string& path) {
    return load_hf(path, {});
}

std::shared_ptr<const BpeTokenizer> BpeTokenizer::load_hf_byte_bpe(const std::string& path, std::string_view eos) {
    if (eos.empty()) throw std::invalid_argument("tokenizer: an explicit EOS is required");
    return load_hf(path, eos);
}

std::shared_ptr<const BpeTokenizer> BpeTokenizer::load_hf(const std::string& path, std::string_view explicit_eos) {
    const bool strict = !explicit_eos.empty();
    const std::string text = read_tokenizer_file(path, "tokenizer");
    const Json root = strict ? parse_json_unique(text, 64U << 20U, kMaxRequestJsonValues) : parse_json(text);
    const Json* model = root.find("model");
    const Json* pre = root.find("pre_tokenizer");
    if (!model || !pre)
        throw std::invalid_argument("tokenizer: " + path + " has no BPE model");
    const Json* vocab = model->find("vocab");
    const Json* merge_list = model->find("merges");
    if (!vocab || !vocab->is_object() || !merge_list || !merge_list->is_array())
        throw std::invalid_argument("tokenizer: " + path + " requires an object for model.vocab and an array for model.merges");
    if (const Json* type = model->find("type"); !type || type->as_string() != "BPE")
        throw std::invalid_argument("tokenizer: " + path + " is not a BPE model");
    // HF byte-level BPE layouts with a recognized Split, ignore_merges and no normalizer.
    static const std::string glm_pattern =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    constexpr std::string_view qwen_pattern =
        R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    const Json* seq = pre->find("pretokenizers");
    const auto* ignore = model->find("ignore_merges");
    const bool is_decimal = strict && seq && seq->is_array() && field_is_string(*pre, "type", "Sequence") &&
                            decimal_triplet_layout(seq->as_array());
    if (!seq || !seq->is_array() || (seq->size() != 2U && !is_decimal) ||
        (strict ? (ignore && !ignore->is_bool()) : !field_is_bool(*model, "ignore_merges", true)) ||
        !field_is_unset(root, "normalizer")) {
        throw std::invalid_argument("tokenizer: " + path + " has an unsupported byte-level BPE layout");
    }
    const auto& split = seq->as_array()[0];
    const auto& bytelevel = seq->as_array().back();
    const auto* pattern = split.find("pattern");
    const bool is_glm = pattern && field_is_string(*pattern, "Regex", glm_pattern);
    const bool plain_layout = plain_hf_byte_bpe_layout(root, *model, *pre, split, bytelevel);
    const bool is_o200k = pattern && field_is_string(*pattern, "Regex", o200k::kPattern) && plain_layout;
    const bool is_qwen = strict && pattern && field_is_string(*pattern, "Regex", qwen_pattern);
    const auto* decoder = root.find("decoder");
    const auto* unk = model->find("unk_token");
    const bool decimal_layout = is_decimal && decoder && field_is_string(*decoder, "type", "ByteLevel") &&
        field_is_unset(root, "padding") && field_is_unset(root, "truncation") &&
        field_is_unset(*model, "dropout") && field_is_unset(*model, "continuing_subword_prefix") &&
        field_is_unset(*model, "end_of_word_suffix") &&
        (!model->find("fuse_unk") || field_is_bool(*model, "fuse_unk", false)) &&
        (!model->find("byte_fallback") || model->find("byte_fallback")->is_bool()) &&
        (!unk || unk->is_null() || (unk->is_string() && vocab->find(unk->as_string()))) &&
        raw_prefix_template(root, *vocab);
    if ((!is_glm && !is_o200k && !is_qwen && !decimal_layout) ||
        (strict && !plain_layout && !decimal_layout) || !field_is_string(bytelevel, "type", "ByteLevel")) {
        throw std::invalid_argument("tokenizer: " + path + " has an unsupported byte-level pre-tokenizer");
    }
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    const auto put = [&](std::int64_t id, const std::string& t, TokenType type) {
        if (id < 0 || id > kMaxTokenizerVocabId || (strict && static_cast<std::uint64_t>(id) >= kMaxRequestJsonValues))
            throw std::invalid_argument("tokenizer: token ID out of range");
        const auto i = static_cast<std::size_t>(id);
        if (strict && (t.empty() || (i < tokens.size() && !tokens[i].empty() &&
            (tokens[i] != t || types[i] != static_cast<std::int32_t>(TokenType::normal)))))
            throw std::invalid_argument("tokenizer: conflicting token ID or empty token");
        if (i >= tokens.size()) { tokens.resize(i + 1U); types.resize(i + 1U, static_cast<std::int32_t>(TokenType::unused)); }
        tokens[i] = t;
        types[i] = static_cast<std::int32_t>(type);
    };
    for (const auto& [t, id] : vocab->as_object()) put(id.as_int(), t, TokenType::normal);
    if (const Json* added = root.find("added_tokens")) {
        for (const auto& a : added->as_array()) {
            if (is_o200k || strict) {
                for (const std::string_view flag : {"single_word", "lstrip", "rstrip", "normalized"}) {
                    if (a.find(flag) && !field_is_bool(a, flag, false))
                        throw std::invalid_argument("tokenizer: unsupported added token flags");
                }
            }
            const Json* special = a.find("special");
            const auto* id = a.find("id");
            const auto* content = a.find("content");
            if (!id || !content) throw std::invalid_argument("tokenizer: added token needs id and content");
            put(id->as_int(), content->as_string(),
                special && special->as_bool() ? TokenType::control : TokenType::user_defined);
        }
    }
    std::vector<std::string> merges;
    merges.reserve(merge_list->size());
    for (const auto& m : merge_list->as_array()) {
        if (m.is_string()) merges.push_back(m.as_string());
        else {
            if (strict && (!m.is_array() || m.size() != 2U))
                throw std::invalid_argument("tokenizer: a merge must have two parts");
            merges.push_back(m.as_array().at(0).as_string() + " " + m.as_array().at(1).as_string());
        }
    }
    auto out = from_parts(std::move(tokens), std::move(types), merges,
                          is_decimal ? "decimal-triplets" : is_o200k ? "o200k" : is_qwen ? "qwen35" : "glm4");
    auto& impl = *const_cast<BpeTokenizer&>(*out).impl_;
    if (strict) {
        if (is_decimal) {
            // Every byte that can occur in UTF-8 needs an ordinary symbol.
            // C0, C1 and F5 through FF never occur in valid UTF-8 input.
            // Unknown-token and byte-fallback paths are unreachable there.
            for (std::size_t byte = 0; byte < impl.byte_unicode.size(); ++byte) {
                if (byte == 0xc0U || byte == 0xc1U || byte > 0xf4U) continue;
                const auto id = out->find(impl.byte_unicode[byte]);
                if (id == UINT32_MAX || out->type(id) != TokenType::normal)
                    throw std::invalid_argument("tokenizer: decimal-triplet BPE needs every ordinary UTF-8 byte");
            }
        }
        impl.ignore_merges = ignore && ignore->as_bool();
        for (const auto id : impl.specials)
            impl.special_start[static_cast<unsigned char>(impl.tokens[id][0])] = true;
        impl.eos = out->find(explicit_eos);
        if (impl.eos == UINT32_MAX || out->type(impl.eos) != TokenType::control)
            throw std::invalid_argument("tokenizer: explicit EOS must be a special added token");
        return out;
    }
    const auto model_end = is_o200k ? out->find("<|content_model_end_sampling|>") : UINT32_MAX;
    impl.eos = model_end == UINT32_MAX ? out->find("<|endoftext|>") : model_end;
    return out;
}

std::shared_ptr<const BpeTokenizer> BpeTokenizer::load_hf_json(const std::string& path, const HfIdentity& expected) {
    auto tokenizer = load_hf_json(path);
    if (tokenizer->pre() != expected.pre || tokenizer->vocab_size() != expected.vocab_size ||
        tokenizer->eos_token() != expected.eos || tokenizer->find(expected.sentinel) != expected.sentinel_id) {
        throw std::invalid_argument("tokenizer: " + path + " does not match the checkpoint tokenizer identity");
    }
    return tokenizer;
}

std::shared_ptr<const BpeTokenizer> BpeTokenizer::load_gguf(const std::string& path) {
    auto parts = read_gguf_tokenizer(path);
    if (parts.model != "gpt2") throw std::invalid_argument("tokenizer: GGUF tokenizer model is not byte-level BPE (gpt2)");
    auto out = from_parts(std::move(parts.tokens), std::move(parts.types), parts.merges, std::move(parts.pre));
    const_cast<BpeTokenizer&>(*out).impl_->eos = parts.eos;
    return out;
}

std::vector<std::uint32_t> BpeTokenizer::encode(std::string_view text, bool parse_special) const {
    return encode_marks(text, parse_special, nullptr);
}

std::vector<std::uint32_t> BpeTokenizer::encode_ordinary(std::string_view text) const {
    std::vector<std::uint32_t> out;
    if (!text.empty()) impl_->encode_plain(text, out);
    return out;
}

std::vector<std::uint32_t> BpeTokenizer::encode_marked(std::string_view text, bool parse_special,
                                                       std::vector<EncodeMark>& marks) const {
    return encode_marks(text, parse_special, &marks);
}

std::size_t BpeTokenizer::mark_lookahead() const noexcept {
    std::size_t longest = 0;
    for (const auto id : impl_->specials) longest = std::max(longest, impl_->tokens[id].size());
    return longest;
}

std::vector<std::uint32_t> BpeTokenizer::encode_marks(std::string_view text, bool parse_special,
                                                      std::vector<EncodeMark>* marks) const {
    std::vector<std::uint32_t> out;
    // HF non-special added tokens still match with special parsing off.
    if (!parse_special && !impl_->rule->whole_added_tokens &&
        impl_->rule->scheme != SplitScheme::right_aligned_decimal) { impl_->encode_plain(text, out); return out; }
    std::size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        if (impl_->special_start[static_cast<unsigned char>(text[i])]) {
            for (const auto id : impl_->specials) {
                if (!parse_special && static_cast<TokenType>(impl_->types[id]) != TokenType::user_defined) continue;
                const auto& s = impl_->tokens[id];
                if (!s.empty() && text.compare(i, s.size(), s) == 0) {
                    if (i > start) impl_->encode_plain(text.substr(start, i - start), out);
                    out.push_back(id);
                    i += s.size();
                    start = i;
                    if (marks) marks->push_back({i, out.size()});
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) ++i;
    }
    if (start < text.size()) impl_->encode_plain(text.substr(start), out);
    return out;
}

bool BpeTokenizer::encode_some(std::string_view text, bool parse_special, EncodeCursor& cursor, std::size_t slice_bytes,
                               std::vector<std::uint32_t>& out) const {
    if (cursor.encoded > cursor.scanned || cursor.scanned > text.size()) throw std::invalid_argument("tokenizer: invalid encode cursor");
    if (cursor.encoded == text.size()) return false;
    if (slice_bytes == 0U) slice_bytes = 1U;
    // The same special-token scan as encode(): positions below `scanned` start no special token, so every position
    // in (encoded, scanned) lies inside one plain segment of the one-shot encoding.
    const bool scan_specials = parse_special || impl_->rule->whole_added_tokens ||
                               impl_->rule->scheme == SplitScheme::right_aligned_decimal;
    const auto ascii_letter = [&](std::size_t k) {
        const auto b = static_cast<unsigned char>(text[k]);
        return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z');
    };
    std::size_t target = cursor.encoded + std::min(slice_bytes, text.size() - cursor.encoded);
    for (;;) {
        while (cursor.scanned < target) {
            const std::size_t i = cursor.scanned;
            if (scan_specials && impl_->special_start[static_cast<unsigned char>(text[i])]) {
                for (const auto id : impl_->specials) {
                    if (!parse_special && static_cast<TokenType>(impl_->types[id]) != TokenType::user_defined) continue;
                    const auto& s = impl_->tokens[id];
                    if (!s.empty() && text.compare(i, s.size(), s) == 0) {
                        if (i > cursor.encoded) impl_->encode_plain(text.substr(cursor.encoded, i - cursor.encoded), out);
                        out.push_back(id);
                        cursor.encoded = cursor.scanned = cursor.searched = i + s.size();
                        return true;
                    }
                }
            }
            ++cursor.scanned;
        }
        if (cursor.scanned == text.size()) {
            impl_->encode_plain(text.substr(cursor.encoded), out);
            cursor.encoded = cursor.searched = text.size();
            return true;
        }
        // A piece boundary that no later text can move: an ASCII letter followed by a space. The letter ends a piece
        // in every pre-tokenizer (letter runs stop at the space, nothing else contains a letter), no piece before it
        // reads past the space, and the scanners never look behind a piece start, so the two sides encode alone.
        const std::size_t floor = std::max(cursor.encoded, cursor.searched);
        for (std::size_t q = cursor.scanned - 1U; q > floor; --q) {
            if (text[q] == ' ' && ascii_letter(q - 1U)) {
                impl_->encode_plain(text.substr(cursor.encoded, q - cursor.encoded), out);
                cursor.encoded = cursor.searched = q;
                return true;
            }
        }
        cursor.searched = cursor.scanned - 1U;
        target = cursor.scanned + std::min(slice_bytes, text.size() - cursor.scanned);
    }
}

std::string BpeTokenizer::token_bytes(std::uint32_t id) const {
    if (id >= impl_->tokens.size()) throw std::out_of_range("tokenizer: token ID outside the vocabulary");
    const auto t = static_cast<TokenType>(impl_->types[id]);
    const auto& text = impl_->tokens[id];
    if (t == TokenType::control || t == TokenType::user_defined || t == TokenType::unused) return text;
    std::string out;
    const auto points = decode_utf8(text);
    for (const auto cp : points.cp) {
        const auto b = impl_->unicode_byte.find(cp);
        if (b == impl_->unicode_byte.end()) throw std::runtime_error("tokenizer: token text outside the byte alphabet");
        out.push_back(static_cast<char>(b->second));
    }
    return out;
}

std::string BpeTokenizer::decode(std::span<const std::uint32_t> ids, bool skip_control) const {
    std::string out;
    for (const auto id : ids) {
        if (impl_->rule->opaque_invalid_bytes && id >= impl_->tokens.size()) continue;
        if (skip_control && id < impl_->types.size() && static_cast<TokenType>(impl_->types[id]) == TokenType::control) continue;
        out += token_bytes(id);
    }
    if (impl_->rule->opaque_invalid_bytes) {
        std::string valid;
        valid.reserve(out.size());
        tokenizer_utf8::append_lossy(valid, out);
        return valid;
    }
    return out;
}

std::size_t BpeTokenizer::vocab_size() const noexcept { return impl_->tokens.size(); }
BpeTokenizer::TokenType BpeTokenizer::type(std::uint32_t id) const {
    if (id >= impl_->types.size()) throw std::out_of_range("tokenizer: token ID outside the vocabulary");
    return static_cast<TokenType>(impl_->types[id]);
}
std::uint32_t BpeTokenizer::find(std::string_view token_text) const {
    const auto hit = impl_->ids.find(std::string(token_text));
    return hit == impl_->ids.end() ? UINT32_MAX : hit->second;
}
const std::string& BpeTokenizer::pre() const noexcept { return impl_->pre; }
std::uint32_t BpeTokenizer::eos_token() const noexcept { return impl_->eos; }

std::vector<std::string> BpeTokenizer::pre_tokenize(std::string_view text) {
    return pre_tokenize(text, "qwen35");
}

std::vector<std::string> BpeTokenizer::pre_tokenize(std::string_view text, std::string_view pre) {
    const auto* rule = find_pre_split_rule(pre);
    if (!rule) throw std::invalid_argument("tokenizer: unknown pre-tokenizer");
    std::vector<std::string> out;
    const auto points = decode_utf8(text, rule->opaque_invalid_bytes);
    for (const auto& [begin, end] : pre_ranges(points, *rule)) out.emplace_back(text.substr(begin, end - begin));
    return out;
}

std::string Utf8StreamDecoder::push(std::string_view bytes) {
    held_.append(bytes);
    // Emit everything up to the start of a trailing incomplete sequence.
    std::size_t keep = held_.size();
    for (std::size_t back = 1; back <= 3U && back <= held_.size(); ++back) {
        const auto b = static_cast<unsigned char>(held_[held_.size() - back]);
        if ((b & 0xC0U) == 0x80U) continue; // continuation byte
        const std::size_t need = b >= 0xF0U ? 4U : b >= 0xE0U ? 3U : b >= 0xC0U ? 2U : 1U;
        if (need > back) keep = held_.size() - back;
        break;
    }
    std::string out = held_.substr(0, keep);
    held_.erase(0, keep);
    return out;
}
std::string Utf8StreamDecoder::flush() {
    std::string out;
    for (std::size_t i = 0; i < held_.size(); ++i) out += "\xEF\xBF\xBD";
    held_.clear();
    return out;
}

}  // namespace strix::serve
