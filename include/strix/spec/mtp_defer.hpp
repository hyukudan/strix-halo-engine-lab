// Deferred prompt MTP rows (shared speculative layer, host-only policy, model- and backend-agnostic). Bound today by the
// CUDA Qwen engine (Config::defer_prompt_mtp, src/cuda/models/qwen/qwen_prefill_driver.cu); host test
// tests/qwen_mtp_defer_tests.cpp. Any engine whose prompt prefill also launches MTP predictor rows can bind it.
//
// The MTP predictor rows of a prompt only feed draft chains: the first token is the verify of the pending token, which
// does not depend on them. So the last prefill block of a prompt may record its MTP call instead of launching it, and
// the call runs later with the same arguments (same bits). The call reads the engine set's block buffers (the block's
// last-layer rows and the next tokens), so it must run before anything rewrites them, and before anything reads the rows
// it writes. It is discarded when the session's rows are replaced or freed.
#pragma once

#include <cstdint>

namespace strix::spec::mtp_defer {

// Rows of the tap buffer: decode windows write rows [0, window_rows); prefill blocks write theirs after them, so no
// window overwrites a block's rows. The buffer holds window_rows + block capacity rows.
constexpr std::uint64_t block_tap_first_row(std::uint32_t window_rows) noexcept { return window_rows; }
constexpr std::uint64_t tap_rows(std::uint32_t window_rows, std::uint32_t block_rows) noexcept {
    return std::uint64_t{window_rows} + block_rows;
}

enum class Use {
    engine_block,  // a prefill block on the engine set (prefill_chunk, run): rewrites the block buffers
    async_block,   // an overlapped prefill chunk (its own block buffers): rewrites only that session's state
    read_rows,     // draft, draft_cohort, row export, snapshot capture of the session: reads its MTP rows
    replace_rows,  // import, snapshot restore, resize, destroy of the session: its rows are replaced or freed
    window,        // a verify window or commit: writes tap rows [0, window_rows) and rows after the prompt only
};

enum class Action { none, settle, drop };

// pending: a record exists; same: it is the session `use` acts on; fresh: the use restarts that session's state
// (prefill from position 0). reads_block: the record includes the last block's call, which reads the engine set's block
// buffers; a record of stashed calls only (stash_block) reads the stash, which the next block of the SAME prompt does not
// rewrite, so that block keeps it pending. Any other session's engine block settles it (one stash, one owner).
constexpr Action on_use(bool pending, bool same, Use use, bool fresh = false, bool reads_block = true) noexcept {
    if (!pending) return Action::none;
    switch (use) {
    case Use::engine_block: return same && fresh ? Action::drop : same && !reads_block ? Action::none : Action::settle;
    case Use::async_block: return !same ? Action::none : fresh ? Action::drop : Action::settle;  // before its set swaps in
    case Use::read_rows: return same ? Action::settle : Action::none;
    case Use::replace_rows: return same ? Action::drop : Action::none;
    case Use::window: return Action::none;
    }
    return Action::settle;  // unreachable: an unknown use keeps the rows correct
}

// The block that completes the prompt is deferred in place: its call reads the block buffers, intact until the next block.
constexpr bool defer_block(bool enabled, std::uint64_t first, std::uint32_t rows, std::uint64_t position) noexcept {
    return enabled && first + rows == position;
}

// An earlier prompt block (first + rows < position) is deferred through the stash: its tap rows and next tokens are copied
// out of the block buffers (which the next block rewrites) to stash rows [stashed, stashed + pairs), and its call runs later
// with those copies and otherwise identical arguments (same rows, positions and inputs: the same MTP bits). Calls settle in
// block order (stashed calls first, then the last block's). A block whose pairs do not fit after the calls already stashed
// is not stashed: the caller settles the stash first and launches inline, so the order of the calls never changes.
constexpr bool stash_block(bool enabled, std::uint64_t first, std::uint32_t rows, std::uint64_t position, std::uint64_t stashed,
                           std::uint32_t pairs, std::uint64_t stash_rows) noexcept {
    return enabled && pairs > 0 && first + rows < position && stashed <= stash_rows && pairs <= stash_rows - stashed;
}

// Stash rows for every non-final block of a prompt of `position` positions started at `first` with blocks of at most
// `block_rows`: the pairs before the last block, at most position - first - 1. 0 when the prompt is one block.
constexpr std::uint64_t stash_rows_for(std::uint64_t first, std::uint64_t position, std::uint32_t block_rows) noexcept {
    return position > first + block_rows ? position - first - 1U : 0U;
}

// Bytes of `rows` stash rows: the wide tap row (BF16) and the next token.
constexpr std::uint64_t stash_bytes(std::uint64_t rows, std::uint32_t wide) noexcept {
    return rows * (std::uint64_t{wide} * 2U + 4U);
}

}  // namespace strix::spec::mtp_defer
