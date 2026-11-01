#pragma once

// Prompt-lookup (n-gram) drafting for the gemma4 engine.
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

namespace strix::gemma4::engine::lookup {

inline constexpr std::uint32_t max_draft = 7; // the verify pass holds at most 8 rows (anchor + 7 drafts)

struct Match {
    std::uint32_t length = 0;                    // matched suffix length (0: no earlier occurrence)
    std::uint64_t distance = 0;                  // current end - earlier occurrence end (recency)
    std::array<std::uint32_t, max_draft> next{}; // continuation after the earlier occurrence
    std::uint32_t available = 0;                 // valid entries of `next`
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
    [[nodiscard]] virtual std::uint64_t tokens() const = 0;
    [[nodiscard]] virtual std::size_t bytes() const = 0;
};

struct IndexConfig {
    std::uint32_t capacity = 1U << 16; // tokens kept (older positions are forgotten); STRIX_G4_LOOKUP_TOKENS
    std::uint32_t max_match = 64;      // matched-length report cap (exact backward extension)
};

// Local implementation: hashed n-grams at lengths {1,2,3,4,6,8,12,16} (open addressing, newest occurrence wins,
// collisions resolved by token comparison), token ring of `capacity`.
[[nodiscard]] std::unique_ptr<Index> make_ngram_index(const IndexConfig& config = {});

} // namespace strix::gemma4::engine::lookup

// ------------------------------------------------------------------------------------------------ hybrid session
#include "gemma4/engine/coordinator.hpp"

namespace strix::gemma4::engine::lookup {

// STRIX_G4_LOOKUP: auto (default; also 1/on) = lookup drafting beside the MTP drafter (per-request selector) and on
// its own when no drafter is loaded | 0/off. Read once per runtime.
struct LookupConfig {
    bool enabled = true;
    std::uint32_t min_length = 2;    // STRIX_G4_LOOKUP_MIN (1..16): shortest match proposed
    double margin = 1.0;             // STRIX_G4_LOOKUP_MARGIN (1..2): lookup must beat the alternative by this factor
    std::uint32_t evidence = 4;      // shadow-scored candidates a match bucket needs before it may be used
    IndexConfig index;               // STRIX_G4_LOOKUP_TOKENS (capacity)
    [[nodiscard]] static LookupConfig from_environment();
    [[nodiscard]] strix::serve::Json json() const;
};

// Wraps an MTP-style session (or none): every committed token feeds a per-stream Index. At C1 (the confidence-window
// path) each cycle picks ONE linear draft: the inner window (MTP) or a lookup continuation of the best cap, or (no inner
// drafter) lookup vs a plain step. Selection is exact-safe and conservative: the lookup side uses lower-bound prefix
// survival per match-length bucket learned by shadow-scoring every cycle's candidate against the committed stream;
// the alternative uses this stream's measured yield and cycle time (upper bound). Multi-stream / forced windows go to
// the inner session unchanged. Greedy verification decides every token: ids == drafting off.
[[nodiscard]] DraftFactory hybrid_factory(DraftFactory inner, LookupConfig config);
// Runtime admission: lookup wraps the MTP drafter or stands alone without a drafter; DFlash and e4b-eagle sessions are
// never wrapped, and drafting off (--dflash off / STRIX_G4_DFLASH=0) leaves it out entirely.
[[nodiscard]] inline bool wraps(const LookupConfig& config, bool drafter_loaded, bool mtp_drafter, bool drafting_off) {
    return config.enabled && !drafting_off && (!drafter_loaded || mtp_drafter);
}

} // namespace strix::gemma4::engine::lookup
