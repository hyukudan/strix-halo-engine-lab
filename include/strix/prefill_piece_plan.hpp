#pragma once
// Prompt piece planner shared by every engine and backend. Host only: no device types, no allocation decisions.
//
// A prompt of `rows` rows runs as consecutive pieces (blocks / chunks). The planner chooses only the piece boundaries; it
// never changes the arithmetic of a row nor any allocated capacity (callers size their buffers from the schedule maximum
// and prove chunk-size invariance of their model where they rely on it). Each trait is a pure function, composed by the
// engine's own wrapper (qwen_prefill_schedule.hpp plan_blocks, cuda_core/submit_window.hpp plan_chunks):
//  * greedy_pieces: piece i holds min(rows left, schedule[min(i, last)]); the last schedule entry repeats.
//  * doubling_ramp: the schedule [r, 2r, 4r, ... (< cap), cap] with r = ramp0 rounded down to `align` (at least `align`),
//    so a card steps up from idle instead of starting at its largest piece.
//  * tail_first: the final remainder runs first when it is smaller than the piece before it (count, sum, maximum kept).
//  * reserve_final: the last piece gets `final_rows` rows (moved from or to the earlier pieces, each keeping >= 1 row and
//    staying within `cap`).
// A backend-specific limit (for example a grid dimension) is validated by the caller before planning.
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strix::prefill {

struct PieceSpan {
    std::uint32_t pos;   // first row of the piece, relative to the prompt start
    std::uint32_t rows;  // rows in the piece, >= 1
};

// schedule: non-empty, every entry >= 1 (the caller validates). rows == 0 gives an empty plan.
[[nodiscard]] inline std::vector<std::uint32_t> greedy_pieces(std::uint64_t rows, const std::vector<std::uint32_t>& schedule) {
    std::vector<std::uint32_t> plan;
    for (std::uint64_t at = 0; at < rows;) {
        const auto n = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows - at, schedule[std::min(plan.size(), schedule.size() - 1)]));
        plan.push_back(n);
        at += n;
    }
    return plan;
}

// cap >= 1. ramp0 == 0 or ramp0 >= cap: [cap]. align 0 counts as 1.
[[nodiscard]] inline std::vector<std::uint32_t> doubling_ramp(std::uint32_t cap, std::uint32_t ramp0, std::uint32_t align) {
    std::vector<std::uint32_t> schedule;
    if (align == 0) align = 1;
    if (ramp0 > 0 && ramp0 < cap) {
        std::uint64_t r = ramp0 / align * align;
        if (r == 0) r = align;
        for (; r < cap; r *= 2) schedule.push_back(static_cast<std::uint32_t>(r));
    }
    schedule.push_back(cap);
    return schedule;
}

inline void tail_first(std::vector<std::uint32_t>& plan) {
    if (plan.size() > 1 && plan.back() < plan[plan.size() - 2]) std::rotate(plan.begin(), plan.end() - 1, plan.end());
}

// plan: the pieces of `rows` rows, each <= cap. final_rows == 0 or a single piece: unchanged.
inline void reserve_final(std::vector<std::uint32_t>& plan, std::uint64_t rows, std::uint32_t final_rows, std::uint32_t cap) {
    if (!final_rows || plan.size() < 2) return;
    const auto target = static_cast<std::uint32_t>(std::min<std::uint64_t>(final_rows, rows - plan.size() + 1));
    if (target > plan.back()) {
        auto need = target - plan.back();
        for (std::size_t i = plan.size() - 1; i-- > 0 && need;) {
            const auto take = std::min(need, plan[i] - 1);
            plan[i] -= take;
            plan.back() += take;
            need -= take;
        }
    } else if (target < plan.back()) {
        auto extra = plan.back() - target;
        for (std::size_t i = 0; i + 1 < plan.size() && extra; ++i) {
            const auto take = std::min(extra, cap - plan[i]);
            plan[i] += take;
            plan.back() -= take;
            extra -= take;
        }
    }
}

[[nodiscard]] inline std::vector<PieceSpan> piece_spans(const std::vector<std::uint32_t>& plan) {
    std::vector<PieceSpan> out;
    out.reserve(plan.size());
    std::uint32_t pos = 0;
    for (const auto n : plan) {
        out.push_back({pos, n});
        pos += n;
    }
    return out;
}

}  // namespace strix::prefill
