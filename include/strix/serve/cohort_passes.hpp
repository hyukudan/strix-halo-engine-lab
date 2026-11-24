#pragma once

// Pass plan of one cohort decode step: `lanes` decode lanes run as `count` batched passes of at most `cap` rows.
//
// A backend binds this when its batched forward keeps a row's arithmetic only up to a row cap (for example a dense
// rows family whose one-row body holds up to a fixed row count and switches to a tile GEMM above it). Running the
// cohort in passes of at most `cap` rows then gives every lane the bits it has at any cohort width up to the cap.
//
// Balanced: the first `extra` passes take base + 1 rows and the others base rows, so passes are contiguous lane
// ranges in order, larger passes come first, and with a cap of at least 3 a cohort above the cap never leaves a pass
// of one row (17 -> 9 + 8, 24 -> 12 + 12, 32 -> 16 + 16, 33 -> 11 + 11 + 11, 34 -> 12 + 11 + 11 at cap 16).
// A cap of 0 or a cap of at least `lanes` is one pass of every lane.
//
// Pure host arithmetic: no device types. The backend supplies the cap; the engine runs one append per pass.
namespace strix::serve {

struct CohortPasses {
    unsigned count = 0U;
    unsigned base = 0U;
    unsigned extra = 0U;
    [[nodiscard]] constexpr unsigned rows(unsigned pass) const noexcept { return base + (pass < extra ? 1U : 0U); }
    [[nodiscard]] constexpr unsigned first(unsigned pass) const noexcept {
        return pass * base + (pass < extra ? pass : extra);
    }
};

[[nodiscard]] constexpr CohortPasses plan_cohort_passes(unsigned lanes, unsigned cap) noexcept {
    if (lanes == 0U) return {};
    if (cap == 0U || cap >= lanes) return {1U, lanes, 0U};
    const unsigned count = lanes / cap + (lanes % cap != 0U ? 1U : 0U);
    return {count, lanes / count, lanes % count};
}

}  // namespace strix::serve
