#pragma once

#include <cstdint>
#include <type_traits>

namespace strix::session {

// One stream-ordered upload. Cache slots, relative offsets and split geometry
// are derived from the absolute position against the captured layout.
struct StepDescriptor {
    std::uint32_t position;
    std::uint32_t token;
};
static_assert(sizeof(StepDescriptor) == 8U && std::is_trivially_copyable_v<StepDescriptor>);

// Wide replay uses the original position/token pair inside each row. The
// single-row binding continues to use StepDescriptor directly.
inline constexpr std::uint32_t step_replay_max_rows = 32;
inline constexpr std::uint32_t step_replay_valid = 1;

// A replay graph width: a power of two in [1, step_replay_max_rows].
[[nodiscard]] constexpr bool is_step_replay_width(std::uint32_t width) noexcept {
    return width != 0 && width <= step_replay_max_rows && (width & (width - 1U)) == 0;
}

// Width of the replay graph that runs a verify window of `rows` rows: the smallest replay width
// >= rows (rows in [1, step_replay_max_rows]). Rows past `rows` are padding: they repeat the last
// id, are never kept, and do not change the kept rows (causal rows). 0 when rows is out of range.
[[nodiscard]] constexpr std::uint32_t padded_step_replay_width(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > step_replay_max_rows) return 0;
    std::uint32_t width = 1;
    while (width < rows) width <<= 1U;
    return width;
}

// Largest active window whose padded graph fits the append plan at this frontier.
template <class Plan>
[[nodiscard]] constexpr std::uint32_t step_replay_rows_that_fit(
    const Plan& plan, std::uint32_t first, std::uint32_t rows) {
    if (rows > step_replay_max_rows) rows = step_replay_max_rows;
    auto width = padded_step_replay_width(rows);
    while (width != 0U && !plan.admits(first, width)) width >>= 1U;
    return rows < width ? rows : width;
}

// Resident session geometry that admits every padded verify window. A speculator keeps a window of
// R rows inside the request (first + R <= positions, R <= verify_rows), but the window runs on width
// W = padded_step_replay_width(R) and the pending append of all W rows must be admitted by the
// session plan: W <= max_chunk and first + W <= context. So the plan's chunk is at least the padded
// width of the largest window and its context carries the largest padding overhang W(R) - R (the
// padded rows are never kept and never published). verify_rows <= 1 keeps the request geometry.
// At max_context the chooser must trim windows with step_replay_rows_that_fit.
struct StepReplaySessionGeometry {
    std::uint32_t positions;
    std::uint32_t append_rows;
};

[[nodiscard]] constexpr StepReplaySessionGeometry step_replay_session_geometry(
    std::uint32_t positions, std::uint32_t chunk_rows, std::uint32_t verify_rows, std::uint32_t max_context) noexcept {
    StepReplaySessionGeometry geometry{positions, chunk_rows};
    if (verify_rows <= 1U) return geometry;
    if (verify_rows > step_replay_max_rows) verify_rows = step_replay_max_rows;
    std::uint32_t overhang = 0;
    for (std::uint32_t rows = 1; rows <= verify_rows; ++rows) {
        const auto width = padded_step_replay_width(rows);
        if (width - rows > overhang) overhang = width - rows;
    }
    const auto widest = padded_step_replay_width(verify_rows);
    if (widest > geometry.append_rows) geometry.append_rows = widest;
    geometry.positions = positions > max_context || max_context - positions < overhang ? max_context
                                                                                       : positions + overhang;
    return geometry;
}

struct StepReplayRow {
    StepDescriptor input;
    std::uint32_t slot, parent, ancestors, flags, rng_lo, rng_hi;
};

struct StepReplayHeader {
    std::uint32_t rows, valid_mask, step, feed_mask, sampling_policy, reserved[3];
};

struct StepReplayBlock {
    StepReplayHeader header;
    StepReplayRow rows[step_replay_max_rows];
};

static_assert(sizeof(StepReplayRow) == 32 && sizeof(StepReplayHeader) == 32);
static_assert(sizeof(StepReplayBlock) == 1056 && std::is_trivially_copyable_v<StepReplayBlock>);

enum class StepReplayAction { eager, capture, replay };

// The owning session invalidates this state before replacing any captured
// allocation or rolling back. An unsupported capture stays eager until reset.
class StepReplayPolicy {
public:
    [[nodiscard]] StepReplayAction select(bool enabled, unsigned rows, bool decode, bool stable_layout) noexcept {
        if (!enabled || rows != 1U || !decode || !stable_layout) {
            invalidate();
            return StepReplayAction::eager;
        }
        return select_bound(rows);
    }

    // The program executor has already validated the shape and owns one
    // policy per graph key. Legacy callers have only the scalar descriptor.
    [[nodiscard]] StepReplayAction select_bound(unsigned rows) const noexcept {
        if (rows == 0 || rows > step_replay_max_rows || failed_) return StepReplayAction::eager;
        return ready_rows_ == rows ? StepReplayAction::replay : StepReplayAction::capture;
    }

    void captured(unsigned rows = 1U) noexcept { ready_rows_ = rows; }
    void invalidate() noexcept { ready_rows_ = 0; }
    void fail() noexcept { invalidate(); failed_ = true; }
    void reset() noexcept { *this = {}; }

private:
    unsigned ready_rows_{};
    bool failed_{};
};

}  // namespace strix::session
