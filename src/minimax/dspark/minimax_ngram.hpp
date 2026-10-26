#pragma once

// Copy (prompt-lookup / n-gram) drafter for MiniMax speculative decoding, host only: the Qwen copy drafter
// design reduced to what the MiniMax verify needs.
//
// The history is the prompt plus every committed token. Every position where an `anchor`-token window ends is indexed.
// propose(k): take the last `anchor` tokens; if they occurred earlier, draft the tokens that followed the MOST RECENT
// earlier occurrence, but only as far as ALL earlier occurrences agree (an ambiguous anchor, such as a JSON key with
// different values per record, stops the draft where the records diverge). Costs no forward pass; the verify is the
// same exact greedy window as DSpark's, so the output tokens never depend on this drafter.

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace strix::minimax::dspark {

class CopyDrafter {
public:
    explicit CopyDrafter(std::uint32_t anchor = 8, std::uint32_t max_occurrences = 64)
        : anchor_(anchor), max_occ_(max_occurrences) {}
    void append(std::span<const std::uint32_t> tokens);
    void append(std::uint32_t token) { append(std::span<const std::uint32_t>(&token, 1)); }
    // Up to k drafted tokens following the current history (empty when the anchor never occurred before).
    [[nodiscard]] std::vector<std::uint32_t> propose(std::uint32_t k) const;
    // Copy continuation inside a neural draft chain (design notes): the present is the history followed by `extra`
    // (drafts not yet verified). Occurrences of its last 3 tokens in the history whose backward match with the present
    // is >= min_match tokens; the longest match wins (the most recent on ties). Up to k tokens follow it, reading through
    // the proposal itself when the copy reaches the present (self-repetition). Empty when no occurrence matches enough.
    [[nodiscard]] std::vector<std::uint32_t> continuation(std::span<const std::uint32_t> extra, std::uint32_t k,
                                                          std::uint32_t min_match) const;
    [[nodiscard]] std::size_t size() const noexcept { return history_.size(); }

private:
    std::uint64_t key_at(std::size_t end) const;  // hash of the anchor window ending before position `end`
    std::uint32_t anchor_, max_occ_;
    std::vector<std::uint32_t> history_;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> ends_;  // window hash -> positions right after it
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> ends3_; // 3-token key -> positions right after it
};

} // namespace strix::minimax::dspark
