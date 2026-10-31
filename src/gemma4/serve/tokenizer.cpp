#include "tokenizer.hpp"

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace strix::gemma4::serve {
namespace {
using strix::serve::Json;
constexpr std::string_view space_marker = "\xE2\x96\x81";
const Json& field(const Json& j, std::string_view key) {
    const auto* p = j.find(key);
    if (!p) throw std::runtime_error("gemma4 tokenizer: missing " + std::string(key));
    return *p;
}
void require(bool ok) {
    if (!ok) throw std::runtime_error("gemma4 tokenizer: unsupported or invalid asset contract");
}
std::uint64_t pair_key(std::uint32_t a, std::uint32_t b) { return (std::uint64_t{a} << 32U) | b; }
std::string replace_spaces(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        if (c == ' ') out += space_marker;
        else out += c;
    }
    return out;
}
std::string decode_spaces(std::string_view s) {
    std::string out;
    for (std::size_t i = 0; i < s.size();) {
        if (s.substr(i, space_marker.size()) == space_marker) { out += ' '; i += space_marker.size(); }
        else out += s[i++];
    }
    return out;
}
// Reject malformed input, while preserving every valid Unicode scalar (including NUL).
std::size_t scalar_length(std::string_view s, std::size_t at) {
    const auto c = static_cast<unsigned char>(s[at]);
    if (c < 0x80U) return 1;
    const std::size_t n = c >= 0xC2U && c <= 0xDFU ? 2U : c >= 0xE0U && c <= 0xEFU ? 3U :
        c >= 0xF0U && c <= 0xF4U ? 4U : 0U;
    require(n != 0 && at + n <= s.size());
    std::uint32_t cp = c & (n == 2 ? 31U : n == 3 ? 15U : 7U);
    for (std::size_t k = 1; k < n; ++k) {
        const auto b = static_cast<unsigned char>(s[at + k]);
        require((b & 0xC0U) == 0x80U);
        cp = (cp << 6U) | (b & 63U);
    }
    require(cp >= (n == 2 ? 0x80U : n == 3 ? 0x800U : 0x10000U) && cp <= 0x10FFFFU &&
        !(cp >= 0xD800U && cp <= 0xDFFFU));
    return n;
}
// The shared API parser intentionally replaces duplicate keys by a linear scan.
// Assets need strict duplicate rejection and linear loading of a 262K-key vocabulary.
// Keep the same Json representation/string escape parser, with an asset-local reader.
class AssetReader {
public:
    explicit AssetReader(std::string_view s) : s_(s) {}
    Json document() { auto j = value(0); skip(); require(at_ == s_.size()); return j; }
private:
    void skip() { while (at_ < s_.size() && (s_[at_] == ' ' || s_[at_] == '\r' || s_[at_] == '\n' || s_[at_] == '\t')) ++at_; }
    bool take(char c) { skip(); if (at_ < s_.size() && s_[at_] == c) { ++at_; return true; } return false; }
    std::string string() {
        skip(); require(at_ < s_.size() && s_[at_] == '"');
        const auto start = at_++;
        bool escaped = false;
        while (at_ < s_.size()) {
            const auto c = static_cast<unsigned char>(s_[at_]);
            require(c >= 0x20U);
            if (c == '"') {
                const auto end = at_++;
                if (escaped) return strix::serve::parse_json_unique(s_.substr(start, at_ - start)).as_string();
                return std::string(s_.substr(start + 1, end - start - 1));
            }
            if (c == '\\') { escaped = true; at_ += 2; }
            else at_ += scalar_length(s_, at_);
        }
        require(false); return {};
    }
    Json value(unsigned depth) {
        require(depth < 64); skip(); require(at_ < s_.size());
        if (s_[at_] == '"') return Json(string());
        if (take('{')) {
            Json::Object members;
            std::unordered_set<std::string> keys;
            if (take('}')) return Json(std::move(members));
            do {
                auto key = string(); require(keys.emplace(key).second); require(take(':'));
                members.emplace_back(std::move(key), value(depth + 1));
            } while (take(','));
            require(take('}')); return Json(std::move(members));
        }
        if (take('[')) {
            Json::Array items;
            if (take(']')) return Json(std::move(items));
            do { items.push_back(value(depth + 1)); } while (take(','));
            require(take(']')); return Json(std::move(items));
        }
        const auto start = at_;
        while (at_ < s_.size() && std::string_view(",]} \r\n\t").find(s_[at_]) == std::string_view::npos) ++at_;
        return strix::serve::parse_json_unique(s_.substr(start, at_ - start));
    }
    std::string_view s_;
    std::size_t at_ = 0;
};
} // namespace

struct Tokenizer::Impl {
    struct Merge { std::uint32_t rank, id; };
    std::vector<std::string> tokens, bytes;
    std::unordered_map<std::string_view, std::uint32_t> ids;
    std::unordered_map<std::uint64_t, Merge> merges;
    std::vector<std::uint32_t> specials;
    std::vector<bool> special;
    std::vector<bool> byte_token;
    std::array<std::uint32_t, 256> fallback{};
    bool ranked_queue = true;

    void bpe(std::string_view input, std::vector<std::uint32_t>& out) const {
        const auto text = replace_spaces(input);
        struct Node { std::uint32_t id; std::size_t prev, next, version = 0; bool alive = true; };
        constexpr auto none = SIZE_MAX;
        std::vector<Node> nodes;
        nodes.reserve(text.size());
        const auto add = [&](std::uint32_t id) {
            const auto i = nodes.size();
            nodes.push_back({id, i ? i - 1U : none, i + 1U});
        };
        for (std::size_t i = 0; i < text.size();) {
            const auto n = scalar_length(text, i);
            if (const auto p = ids.find(std::string_view(text).substr(i, n)); p != ids.end()) add(p->second);
            else for (std::size_t k = 0; k < n; ++k) add(fallback[static_cast<unsigned char>(text[i + k])]);
            i += n;
        }
        if (nodes.empty()) return;
        nodes.back().next = none;
        if (!ranked_queue) {
            // Literal ranked-pair reference, with leftmost tie breaking. It shares
            // symbolization but no candidate queue or invalidation bookkeeping.
            for (;;) {
                auto best = none;
                Merge selected{UINT32_MAX, 0};
                for (auto i = std::size_t{0}; i != none; i = nodes[i].next) {
                    if (nodes[i].next == none) break;
                    const auto m = merges.find(pair_key(nodes[i].id, nodes[nodes[i].next].id));
                    if (m != merges.end() && m->second.rank < selected.rank) { best = i; selected = m->second; }
                }
                if (best == none) break;
                auto& a = nodes[best]; auto& b = nodes[a.next];
                a.id = selected.id; a.next = b.next; b.alive = false;
            }
            for (auto i = std::size_t{0}; i != none; i = nodes[i].next) out.push_back(nodes[i].id);
            return;
        }
        struct Candidate {
            std::uint32_t rank, id;
            std::size_t left, right, version, right_version;
            bool operator<(const Candidate& b) const { return rank != b.rank ? rank > b.rank : left > b.left; }
        };
        std::priority_queue<Candidate> queue;
        const auto offer = [&](std::size_t a) {
            if (a == none || nodes[a].next == none) return;
            const auto b = nodes[a].next;
            if (const auto m = merges.find(pair_key(nodes[a].id, nodes[b].id)); m != merges.end())
                queue.push({m->second.rank, m->second.id, a, b, nodes[a].version, nodes[b].version});
        };
        for (std::size_t i = 0; i < nodes.size(); ++i) offer(i);
        while (!queue.empty()) {
            const auto c = queue.top(); queue.pop();
            auto& a = nodes[c.left];
            auto& b = nodes[c.right];
            if (!a.alive || !b.alive || a.next != c.right || a.version != c.version || b.version != c.right_version) continue;
            a.id = c.id; a.next = b.next; ++a.version; b.alive = false;
            if (a.next != none) nodes[a.next].prev = c.left;
            offer(a.prev); offer(c.left);
        }
        for (const auto& n : nodes) if (n.alive) out.push_back(n.id);
    }
};

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;

std::shared_ptr<const Tokenizer> Tokenizer::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("gemma4 tokenizer: cannot open asset");
    const auto size = file.tellg();
    require(size > 0 && size < (128LL << 20U));
    std::string data(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    require(static_cast<bool>(file.read(data.data(), static_cast<std::streamsize>(data.size()))));
    const auto root = AssetReader(data).document();
    require(field(root, "version").as_string() == "1.0");
    const auto expected = strix::serve::parse_json(R"({"type":"Replace","pattern":{"String":" "},"content":"\u2581"})");
    require(field(root, "normalizer").dump() == expected.dump());
    require(field(root, "pre_tokenizer").dump() == R"({"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false})");
    require(field(root, "decoder").dump() == R"({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]})");
    require(field(root, "post_processor").dump() == R"({"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}})");
    require(field(root, "truncation").is_null() && field(root, "padding").is_null());
    const auto& model = field(root, "model");
    require(field(model, "type").as_string() == "BPE" && field(model, "dropout").is_null() &&
        field(model, "byte_fallback").as_bool() && !field(model, "ignore_merges").as_bool() &&
        field(model, "fuse_unk").as_bool() && field(model, "unk_token").as_string() == "<unk>" &&
        field(model, "continuing_subword_prefix").is_null() && field(model, "end_of_word_suffix").is_null());
    auto out = std::shared_ptr<Tokenizer>(new Tokenizer);
    auto& p = *out->impl_;
    if (const auto* mode = std::getenv("STRIX_G4_TOKENIZER_HEAP")) {
        require(std::string_view(mode) == "0" || std::string_view(mode) == "1");
        p.ranked_queue = std::string_view(mode) != "0";
    }
    const auto& vocab = field(model, "vocab").as_object();
    require(vocab.size() == 262144U);
    p.tokens.resize(vocab.size()); p.bytes.resize(vocab.size()); p.special.resize(vocab.size()); p.byte_token.resize(vocab.size());
    std::vector<bool> seen(vocab.size());
    for (const auto& [text, value] : vocab) {
        const auto id = value.as_int();
        require(id >= 0 && static_cast<std::size_t>(id) < vocab.size() && !seen[static_cast<std::size_t>(id)]);
        seen[static_cast<std::size_t>(id)] = true;
        p.tokens[static_cast<std::size_t>(id)] = text;
    }
    p.ids.reserve(vocab.size());
    for (std::uint32_t i = 0; i < p.tokens.size(); ++i) {
        require(p.ids.emplace(p.tokens[i], i).second);
        p.bytes[i] = decode_spaces(p.tokens[i]);
    }
    for (unsigned b = 0; b < 256; ++b) {
        char token[8]; std::snprintf(token, sizeof(token), "<0x%02X>", b);
        const auto i = p.ids.find(token); require(i != p.ids.end());
        p.fallback[b] = i->second;
        p.byte_token[i->second] = true;
        p.bytes[i->second] = std::string(1, static_cast<char>(b));
    }
    const auto& merges = field(model, "merges").as_array();
    p.merges.reserve(merges.size());
    for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
        const auto& m = merges[rank].as_array(); require(m.size() == 2);
        const auto a = p.ids.find(m[0].as_string()), b = p.ids.find(m[1].as_string());
        const auto c = p.ids.find(m[0].as_string() + m[1].as_string());
        require(a != p.ids.end() && b != p.ids.end() && c != p.ids.end());
        require(p.merges.emplace(pair_key(a->second, b->second), Impl::Merge{rank, c->second}).second);
    }
    for (const auto& a : field(root, "added_tokens").as_array()) {
        const auto id = field(a, "id").as_int();
        require(id >= 0 && static_cast<std::size_t>(id) < p.tokens.size());
        const auto i = static_cast<std::uint32_t>(id);
        require(p.tokens[i] == field(a, "content").as_string() && !p.tokens[i].empty() &&
            !field(a, "single_word").as_bool() && !field(a, "lstrip").as_bool() &&
            !field(a, "rstrip").as_bool() && !field(a, "normalized").as_bool() && field(a, "special").as_bool());
        require(!p.special[i]); p.special[i] = true; p.specials.push_back(i);
    }
    std::sort(p.specials.begin(), p.specials.end(), [&](auto a, auto b) { return p.tokens[a].size() > p.tokens[b].size(); });
    return out;
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
    std::vector<std::uint32_t> out;
    if (!parse_special) { impl_->bpe(text, out); return out; }
    std::size_t plain = 0, at = 0;
    while (at < text.size()) {
        bool found = false;
        if (text[at] == '<') for (const auto id : impl_->specials) {
            const auto& token = impl_->tokens[id];
            if (text.substr(at, token.size()) != token) continue;
            impl_->bpe(text.substr(plain, at - plain), out);
            out.push_back(id); at += token.size(); plain = at; found = true; break;
        }
        if (!found) at += scalar_length(text, at);
    }
    impl_->bpe(text.substr(plain), out);
    return out;
}
std::string Tokenizer::token_bytes(std::uint32_t id) const {
    if (id >= impl_->bytes.size()) throw std::out_of_range("gemma4 tokenizer: token ID");
    return impl_->bytes[id];
}
std::string Tokenizer::decode(std::span<const std::uint32_t> ids, bool skip_special) const {
    std::string out, run;
    const auto flush = [&] {
        if (run.empty()) return;
        // The asset's ByteFallback decoder validates the entire consecutive byte
        // group. An invalid group emits one replacement per byte, even for a valid
        // prefix inside that group; ordinary incremental UTF-8 replacement differs.
        bool valid = true;
        try { (void)strix::serve::parse_json_unique(Json(run).dump()); }
        catch (const strix::serve::JsonError&) { valid = false; }
        if (valid) out += run;
        else for (std::size_t i = 0; i < run.size(); ++i) out += "\xEF\xBF\xBD";
        run.clear();
    };
    for (const auto id : ids) {
        if (id >= impl_->bytes.size()) throw std::out_of_range("gemma4 tokenizer: token ID");
        if (skip_special && impl_->special[id]) continue;
        if (impl_->byte_token[id]) run += impl_->bytes[id];
        else { flush(); out += impl_->bytes[id]; }
    }
    flush(); return out;
}
std::uint32_t Tokenizer::find(std::string_view text) const {
    const auto i = impl_->ids.find(text); return i == impl_->ids.end() ? UINT32_MAX : i->second;
}
std::size_t Tokenizer::vocab_size() const noexcept { return impl_->tokens.size(); }

} // namespace strix::gemma4::serve
