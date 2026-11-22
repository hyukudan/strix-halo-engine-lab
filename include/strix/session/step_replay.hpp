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

[[nodiscard]] constexpr bool is_step_replay_width(std::uint32_t rows) noexcept {
    return rows != 0 && rows <= step_replay_max_rows && (rows & (rows - 1U)) == 0;
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
