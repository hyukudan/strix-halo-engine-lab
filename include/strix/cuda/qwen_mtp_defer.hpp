// Deferred prompt MTP rows of the CUDA Qwen engine (Config::defer_prompt_mtp), host-only policy used by the driver
// (src/cuda/models/qwen/qwen_prefill_driver.cu) and its host test (tests/qwen_mtp_defer_tests.cpp).
//
// The MTP predictor rows of a prompt only feed draft chains: the first token is the verify of the pending token, which
// does not depend on them. So the last prefill block of a prompt may record its MTP call instead of launching it, and
// the call runs later with the same arguments (same bits). The call reads the engine set's block buffers (the block's
// layer-47 rows and the next tokens), so it must run before anything rewrites them, and before anything reads the rows
// it writes. It is discarded when the session's rows are replaced or freed.
#pragma once

#include <cstdint>

namespace strix::cuda::qwen_mtp_defer {

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
// (prefill from position 0).
constexpr Action on_use(bool pending, bool same, Use use, bool fresh = false) noexcept {
    if (!pending) return Action::none;
    switch (use) {
    case Use::engine_block: return same && fresh ? Action::drop : Action::settle;
    case Use::async_block: return !same ? Action::none : fresh ? Action::drop : Action::settle;
    case Use::read_rows: return same ? Action::settle : Action::none;
    case Use::replace_rows: return same ? Action::drop : Action::none;
    case Use::window: return Action::none;
    }
    return Action::settle;  // unreachable: an unknown use keeps the rows correct
}

// Only the block that completes the prompt is deferred (earlier blocks are settled by the next block anyway).
constexpr bool defer_block(bool enabled, std::uint64_t first, std::uint32_t rows, std::uint64_t position) noexcept {
    return enabled && first + rows == position;
}

}  // namespace strix::cuda::qwen_mtp_defer
