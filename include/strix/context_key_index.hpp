#pragma once
// Earlier occurrences of a short fixed-length token key in a growing token context (host only). Shared by the
// context drafters (strix/mtp_ngram_draft.hpp, strix/mtp_context_draft.hpp).
//
// One slot per distinct key holds its most recent end position and one link per position holds the previous end
// with the same key: indexing a token is one map probe and one store, with no allocation per key. A drafter indexes
// only its shortest suffix length and measures a longer exact suffix on the candidates by comparing tokens backwards.
// This replaces one map entry (node + vector) per position AND suffix length, which every request paid over its whole
// prompt before the first decode cycle (80K prompt, lengths 2..16 and 2..4: ~356 ms).
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strix {

class ContextKeyIndex {
public:
    explicit ContextKeyIndex(std::uint32_t key_tokens) : key_tokens_(key_tokens) {
        if (!key_tokens_) throw std::invalid_argument("context key index needs at least one key token");
    }
    [[nodiscard]] std::uint32_t key_tokens() const noexcept { return key_tokens_; }
    void clear() noexcept { head_.clear(); link_.clear(); }
    void reserve(std::size_t tokens) { link_.reserve(tokens); head_.reserve(tokens / 2U); }
    // Indexes the key ending at the last token of `context` (the token just appended): one call per token, in order.
    void push(std::span<const std::uint32_t> context) {
        if (context.size() != link_.size() + 1U) throw std::logic_error("context key index push is out of order");
        if (context.size() >= std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("context key index exceeds its position capacity");
        if (context.size() < key_tokens_) { link_.push_back(0); return; }
        auto& head = head_[key_of(context.data() + (context.size() - key_tokens_))];
        link_.push_back(head);
        head = static_cast<std::uint32_t>(context.size());
    }
    // Calls visit(end) for every indexed end whose key matches `key` (key.size() == key_tokens()), most recent first,
    // until visit returns false. Keys of more than two tokens are hashed: callers compare the tokens.
    template <class Visit> void for_each(std::span<const std::uint32_t> key, Visit&& visit) const {
        if (key.size() != key_tokens_) throw std::invalid_argument("context key index lookup key has the wrong length");
        const auto hit = head_.find(key_of(key.data()));
        if (hit == head_.end()) return;
        for (auto e = hit->second; e != 0U; e = link_[e - 1U])
            if (!visit(static_cast<std::size_t>(e - 1U))) return;
    }

private:
    [[nodiscard]] std::uint64_t key_of(const std::uint32_t* k) const noexcept {
        if (key_tokens_ == 1U) return k[0];
        if (key_tokens_ == 2U) return (static_cast<std::uint64_t>(k[0]) << 32U) | k[1];
        std::uint64_t h = 0xcbf29ce484222325ULL;
        for (std::uint32_t i = 0; i < key_tokens_; ++i) {
            h ^= k[i] + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
            h *= 0xff51afd7ed558ccdULL;
        }
        return h;
    }
    std::uint32_t key_tokens_;
    std::unordered_map<std::uint64_t, std::uint32_t> head_;  // key -> most recent end + 1
    std::vector<std::uint32_t> link_;                        // per position: previous end + 1 with the same key, 0 none
};

}  // namespace strix
