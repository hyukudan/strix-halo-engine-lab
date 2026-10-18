#include "strix/serve/bpe_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace strix::serve {
namespace {

struct UnicodeRange { std::uint32_t first, last; std::uint8_t flags; };
#include "unicode_tables.inc"

constexpr std::uint8_t kLetter = 1U, kMark = 2U, kNumber = 4U;

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
CodePoints decode_utf8(std::string_view text) {
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
        out.cp.push_back(cp);
        i += length;
    }
    out.offset.push_back(text.size());
    return out;
}

// The qwen35 pattern:
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Alternatives are tried in order at each position, each with its greedy/backtracking meaning.
std::vector<std::pair<std::size_t, std::size_t>> pre_split(const CodePoints& text) {
    const auto& c = text.cp;
    const auto n = c.size();
    const auto letter_or_mark = [&](std::size_t k) { return k < n && (category(c[k]) & (kLetter | kMark)) != 0U; };
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
        // 3. \p{N}
        if (!end && (category(c[i]) & kNumber)) end = i + 1U;
        // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (!end) {
            const auto symbol = [&](std::size_t k) { return k < n && !white_space(c[k]) && !(category(c[k]) & (kLetter | kMark | kNumber)); };
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

std::array<std::string, 256> byte_to_unicode() {
    std::array<std::string, 256> out{};
    std::array<bool, 256> direct{};
    for (int b = '!'; b <= '~'; ++b) direct[static_cast<std::size_t>(b)] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[static_cast<std::size_t>(b)] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[static_cast<std::size_t>(b)] = true;
    std::uint32_t extra = 0;
    for (std::size_t b = 0; b < 256U; ++b) {
        const std::uint32_t cp = direct[b] ? static_cast<std::uint32_t>(b) : 256U + extra++;
        std::string s;
        if (cp < 0x80U) s.push_back(static_cast<char>(cp));
        else { s.push_back(static_cast<char>(0xC0U | (cp >> 6U))); s.push_back(static_cast<char>(0x80U | (cp & 0x3FU))); }
        out[b] = s;
    }
    return out;
}

// GGUF header metadata (only the tokenizer keys are kept).
struct GgufTokenizer {
    std::vector<std::string> tokens, merges;
    std::vector<std::int32_t> types;
    std::string model, pre;
    std::uint32_t eos = UINT32_MAX;
};
GgufTokenizer read_gguf_tokenizer(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("tokenizer: cannot open " + path);
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
            if (item == 8U) {
                std::vector<std::string>* target = key == "tokenizer.ggml.tokens" ? &out.tokens :
                    key == "tokenizer.ggml.merges" ? &out.merges : nullptr;
                if (target) target->reserve(static_cast<std::size_t>(count));
                for (std::uint64_t k = 0; k < count; ++k) {
                    auto value = string();
                    if (target) target->push_back(std::move(value));
                }
                continue;
            }
            if (item >= sizes.size() || !sizes[item]) throw std::runtime_error("tokenizer: unsupported GGUF array element type");
            if (key == "tokenizer.ggml.token_type" && (item == 5U || item == 4U)) {
                out.types.resize(static_cast<std::size_t>(count));
                read(out.types.data(), out.types.size() * 4U);
                continue;
            }
            file.seekg(static_cast<std::streamoff>(sizes[item] * count), std::ios::cur);
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
    std::array<std::string, 256> byte_unicode = byte_to_unicode();
    std::unordered_map<std::uint32_t, unsigned char> unicode_byte;
    std::string pre;
    std::uint32_t eos = UINT32_MAX;
    mutable std::mutex cache_mutex;
    mutable std::unordered_map<std::string, std::vector<std::uint32_t>> cache;

    void bpe(std::string_view piece, std::vector<std::uint32_t>& out) const {
        {
            const std::lock_guard lock(cache_mutex);
            if (const auto hit = cache.find(std::string(piece)); hit != cache.end()) {
                out.insert(out.end(), hit->second.begin(), hit->second.end());
                return;
            }
        }
        std::vector<std::string> symbols;
        symbols.reserve(piece.size());
        for (const char ch : piece) symbols.push_back(byte_unicode[static_cast<unsigned char>(ch)]);
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
            if (id != ids.end()) { result.push_back(id->second); continue; }
            throw std::runtime_error("tokenizer: BPE symbol is not in the vocabulary");
        }
        {
            const std::lock_guard lock(cache_mutex);
            if (cache.size() > 200000U) cache.clear();
            cache.emplace(std::string(piece), result);
        }
        out.insert(out.end(), result.begin(), result.end());
    }
    void encode_plain(std::string_view text, std::vector<std::uint32_t>& out) const {
        const auto points = decode_utf8(text);
        for (const auto& [begin, end] : pre_split(points)) bpe(text.substr(begin, end - begin), out);
    }
};

BpeTokenizer::BpeTokenizer() : impl_(std::make_unique<Impl>()) {}
BpeTokenizer::~BpeTokenizer() = default;

std::shared_ptr<const BpeTokenizer> BpeTokenizer::from_parts(std::vector<std::string> tokens, std::vector<std::int32_t> types,
    std::span<const std::string> merges, std::string pre) {
    if (pre != "qwen35") throw std::invalid_argument("tokenizer: only the qwen35 byte-level BPE pre-tokenizer is implemented, not " + pre);
    if (tokens.empty() || types.size() != tokens.size()) throw std::invalid_argument("tokenizer: token and type tables differ");
    std::shared_ptr<BpeTokenizer> out(new BpeTokenizer());
    auto& impl = *out->impl_;
    impl.pre = std::move(pre);
    impl.tokens = std::move(tokens);
    impl.types = std::move(types);
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
    for (std::size_t b = 0; b < 256U; ++b) {
        const auto points = decode_utf8(impl.byte_unicode[b]);
        impl.unicode_byte.emplace(points.cp.at(0), static_cast<unsigned char>(b));
    }
    return out;
}

std::shared_ptr<const BpeTokenizer> BpeTokenizer::load_gguf(const std::string& path) {
    auto parts = read_gguf_tokenizer(path);
    if (parts.model != "gpt2") throw std::invalid_argument("tokenizer: GGUF tokenizer model is not byte-level BPE (gpt2)");
    auto out = from_parts(std::move(parts.tokens), std::move(parts.types), parts.merges, std::move(parts.pre));
    const_cast<BpeTokenizer&>(*out).impl_->eos = parts.eos;
    return out;
}

std::vector<std::uint32_t> BpeTokenizer::encode(std::string_view text, bool parse_special) const {
    std::vector<std::uint32_t> out;
    if (!parse_special) { impl_->encode_plain(text, out); return out; }
    std::size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        if (text[i] == '<' || text[i] == '[') {
            for (const auto id : impl_->specials) {
                const auto& s = impl_->tokens[id];
                if (!s.empty() && text.compare(i, s.size(), s) == 0) {
                    if (i > start) impl_->encode_plain(text.substr(start, i - start), out);
                    out.push_back(id);
                    i += s.size();
                    start = i;
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
        if (skip_control && id < impl_->types.size() && static_cast<TokenType>(impl_->types[id]) == TokenType::control) continue;
        out += token_bytes(id);
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
    std::vector<std::string> out;
    const auto points = decode_utf8(text);
    for (const auto& [begin, end] : pre_split(points)) out.emplace_back(text.substr(begin, end - begin));
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
