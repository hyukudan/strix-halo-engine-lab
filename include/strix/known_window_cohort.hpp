#pragma once

#include "strix/execution_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace strix {

// A cohort window carries the rows of several independent sessions. Each session's rows are one
// segment: consecutive positions over that session's own state and scratch arena. While a
// KnownWindowCohortScope is alive on this thread, the stateless wide launchers (routed expert
// union, HC mixers, router, LM head) admit such rows: every row still passes its own complete R1
// admission against its own resources, and only the single-session cross-row checks (one state
// binding, one scratch arena with a fixed row stride, consecutive positions across all rows) are
// replaced by per-segment ones. Outside a scope every launcher keeps its single-session contract.
inline bool& known_window_cohort_flag() noexcept {
    thread_local bool active = false;
    return active;
}
[[nodiscard]] inline bool known_window_cohort_active() noexcept { return known_window_cohort_flag(); }

class KnownWindowCohortScope {
public:
    KnownWindowCohortScope() noexcept : previous_(known_window_cohort_flag()) { known_window_cohort_flag() = true; }
    ~KnownWindowCohortScope() { known_window_cohort_flag() = previous_; }
    KnownWindowCohortScope(const KnownWindowCohortScope&) = delete;
    KnownWindowCohortScope& operator=(const KnownWindowCohortScope&) = delete;
private:
    bool previous_;
};

// Row r starts a new session segment: inside a cohort scope, a row whose session state binding
// differs from the previous row's. Row 0 always starts one.
[[nodiscard]] inline bool known_window_segment_start(std::span<const ExecutionDispatchContext> rows, std::size_t r) noexcept {
    if (r == 0) return true;
    return known_window_cohort_active() && rows[r].session_state_memory != rows[r - 1U].session_state_memory;
}
// The single-session position rule (row r at row 0's position + r), per segment in a cohort.
[[nodiscard]] inline bool known_window_row_position_follows(std::span<const ExecutionDispatchContext> rows, std::size_t r) noexcept {
    if (known_window_segment_start(rows, r)) return true;
    return rows[r - 1U].position != UINT64_MAX && rows[r].position == rows[r - 1U].position + 1U;
}

}  // namespace strix
