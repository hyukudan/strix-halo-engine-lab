#pragma once

// Engram hash: token ids -> table rows (formula in strix/engram/spec.hpp). Pure host integer code; the rows of a
// token depend only on token ids, so a runner hashes a step's tokens for EVERY Engram layer before layer 0 runs and
// issues their reads then (token-time prefetch).

#include "strix/engram/spec.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix::engram {

// Marks a token id of a sequence history as a dead n-gram position (e.g. an image row): no n-gram reads through
// it, its own n-grams are all padding, and the bit stays in the history.
inline constexpr std::uint32_t kDeadBit = 0x8000'0000U;
[[nodiscard]] constexpr std::uint32_t dead_token(std::uint32_t token) noexcept { return token | kDeadBit; }

// Rows of one position from its n-gram's compressed ids (ngram[s] = id of t - s, padding already applied),
// written [layer][column]. offsets = column_offsets(spec).
void hash_position(const Spec& spec, std::span<const std::uint64_t> offsets, std::span<const std::uint64_t> ngram,
                   std::span<std::uint64_t> rows);
// The columns of one layer e only (hash_position = this for every layer): rows.size() == spec.columns().
void hash_layer(const Spec& spec, std::span<const std::uint64_t> offsets, std::uint32_t e,
                std::span<const std::uint64_t> ngram, std::span<std::uint64_t> rows);

class Hasher {
public:
    explicit Hasher(Spec spec);
    [[nodiscard]] const Spec& spec() const noexcept { return spec_; }
    [[nodiscard]] std::uint32_t layers() const noexcept { return spec_.layers(); }
    [[nodiscard]] std::uint32_t columns() const noexcept { return spec_.columns(); }
    // Appends `token` to the sequence and writes its rows, [layer][column] (layers() * columns()).
    void push(std::uint32_t token, std::span<std::uint64_t> rows);
    // Rows of every token of a sequence starting at position 0 (prompt): [token][layer][column].
    [[nodiscard]] std::vector<std::uint64_t> hash_sequence(std::span<const std::uint32_t> tokens);
    [[nodiscard]] std::uint64_t positions() const noexcept { return history_.size(); }
    // Stateless twin of push for a runner that keeps its own token history: the rows of position t of `tokens`
    // (tokens[0 .. t], t < tokens.size()) at layer e, relative to the layer's table, rows.size() == columns().
    // Same padding, unmapped and dead rules as push; the Hasher's own history is not read or changed.
    void rows_at(std::span<const std::uint32_t> tokens, std::size_t t, std::uint32_t e, std::span<std::uint64_t> rows) const;
    // Verify windows: drop the trailing positions so the history ends after `positions` tokens.
    void truncate(std::uint64_t positions);
    void reset() noexcept { history_.clear(); }

private:
    Spec spec_;
    std::vector<std::uint64_t> offsets_;  // [layer][column]
    std::uint32_t pad_ = 0;               // compressed pad id
    std::vector<std::uint32_t> history_;  // compressed ids; kDead for a dead position
    std::vector<std::uint64_t> ngram_;
    static constexpr std::uint32_t kDead = 0xFFFF'FFFFU;
    static constexpr std::size_t kMaxStatelessNgram = 8;  // rows_at's on-stack n-gram (Hy4 3, DS41 3)
    // Compressed id of one history token (kDead for a dead position); throws on an unmapped id under Unmapped::reject.
    [[nodiscard]] std::uint32_t compressed(std::uint32_t token) const;
    // ngram[o] = compressed id of t - o, padded at the sequence start and behind a dead position.
    template <class IdAt>
    void fill_ngram(std::size_t t, IdAt id_at, std::span<std::uint64_t> ngram) const {
        bool blocked = false;
        for (std::size_t o = 0; o < ngram.size(); ++o) {
            blocked = blocked || t < o || id_at(t - o) == kDead;  // nothing reads through a dead position
            ngram[o] = blocked ? pad_ : id_at(t - o);
        }
    }
};

} // namespace strix::engram
