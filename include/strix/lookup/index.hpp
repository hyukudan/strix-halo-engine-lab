#pragma once

// Prompt-lookup (n-gram) index shared by the engines' drafting.
//
// Index: an exact, incremental longest-suffix matcher over one stream's committed tokens (prompt + generated), with
// bounded memory. Interface kept minimal so a shared n-gram component can replace the local
// implementation: append committed tokens, query the longest earlier occurrence of the current suffix (optionally
// extended by up to 7 hypothetical tokens) and its continuation.
//
// Exactness: lookup only proposes draft tokens; greedy verification decides every emitted token, so output ids equal
// drafting off for any proposal. The index reads committed tokens only (never drafted ones), so a rejected draft
// needs no rollback.
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix::lookup {

inline constexpr std::uint32_t max_draft = 7; // the verify pass holds at most 8 rows (anchor + 7 drafts)

struct Match {
    std::uint32_t length = 0;                    // matched suffix length (0: no earlier occurrence)
    std::uint64_t distance = 0;                  // current end - earlier occurrence end (recency)
    std::array<std::uint32_t, max_draft> next{}; // continuation after the earlier occurrence
    std::uint32_t available = 0;                 // valid entries of `next`
    std::uint64_t source_end = 0;                // stream(+extra) position of the earlier occurrence's last token
};

class Index {
public:
    virtual ~Index() = default;
    // Committed token at the next stream position.
    virtual void append(std::uint32_t token) = 0;
    // Longest suffix (>= min_length) of stream + extra that also ends at an earlier position of the stream; the
    // continuation is taken from the stream (+ extra) after that occurrence, up to `want` tokens.
    [[nodiscard]] virtual Match longest(std::span<const std::uint32_t> extra, std::uint32_t min_length,
                                        std::uint32_t want) const = 0;
    // Tokens of stream + extra at positions source_end + 1, source_end + 2, ... (at most out.size(), never past the
    // last position, never older than the ring): a continuation longer than `next`, e.g. a copy cursor. Returns the
    // count written.
    [[nodiscard]] virtual std::size_t continuation(std::uint64_t source_end, std::span<const std::uint32_t> extra,
                                                   std::span<std::uint32_t> out) const = 0;
    // Token at stream position `position` (committed tokens only; position < tokens() and within the ring).
    [[nodiscard]] virtual std::uint32_t at(std::uint64_t position) const = 0;
    // Oldest stream position still held by the ring.
    [[nodiscard]] virtual std::uint64_t oldest() const = 0;
    [[nodiscard]] virtual std::uint64_t tokens() const = 0;
    [[nodiscard]] virtual std::size_t bytes() const = 0;
};

struct IndexConfig {
    std::uint32_t capacity = 1U << 16; // tokens kept (older positions are forgotten)
    std::uint32_t max_match = 64;      // matched-length report cap (exact backward extension)
};

// Local implementation: hashed n-grams at lengths {1,2,3,4,6,8,12,16} (open addressing, newest occurrence wins,
// collisions resolved by token comparison), token ring of `capacity`.
[[nodiscard]] std::unique_ptr<Index> make_ngram_index(const IndexConfig& config = {});

// Separates the blocks of a Corpus; never a vocabulary id, so no query matches across it and no continuation
// runs past it.
inline constexpr std::uint32_t corpus_break = 0xFFFFFFFFU;

// A fixed auxiliary token corpus (e.g. restored tool results, strix/session/tool_source.hpp) matched against an
// external suffix: the stream being decoded is not part of it. Same index as above (every position kept), each
// block ended by corpus_break. Read-only after construction, so drafters of concurrent requests may share one.
class Corpus {
public:
    // `tokens`: blocks separated by corpus_break (a final break is added).
    explicit Corpus(std::span<const std::uint32_t> tokens);
    // Longest suffix of `suffix` (>= min_length, hashed at most 16 tokens, extended exactly up to suffix.size())
    // that occurs in the corpus, newest occurrence, with up to `want` (<= max_draft) continuation tokens before the
    // block's end. length 0: no occurrence. source_end: corpus position of the match's last token.
    [[nodiscard]] Match match(std::span<const std::uint32_t> suffix, std::uint32_t min_length, std::uint32_t want) const;
    // Continuation after corpus position source_end, up to out.size() tokens, stopping at the block's end.
    [[nodiscard]] std::size_t continuation(std::uint64_t source_end, std::span<std::uint32_t> out) const;
    [[nodiscard]] std::uint64_t tokens() const { return index_->tokens(); }
    [[nodiscard]] std::size_t blocks() const { return blocks_; }
    [[nodiscard]] std::size_t bytes() const { return index_->bytes(); }

private:
    std::unique_ptr<Index> index_;
    std::size_t blocks_ = 0;
};

} // namespace strix::lookup

