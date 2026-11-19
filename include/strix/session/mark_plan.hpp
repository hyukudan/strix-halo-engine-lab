#pragma once
// Prompt-state marks: the positions where one prompt pass keeps a restorable state (a prefix-cache checkpoint) on its
// way through, so a later request that repeats the prompt up to a mark resumes there instead of at row 0.
//
// Backend-neutral and model-neutral: rows, blocks and token counts only. A family binds it by feeding the plan to its
// own capture mechanism (chunk ends or in-run snapshot points); the plan never changes the rows a pass computes, only
// where a state is exported.
//
// Marks of one pass, ascending, unique, every one a multiple of `block`, above `restored` and at most the end mark:
// - grid: absolute multiples of `stride` (anchored at row 0, so two passes over the same prefix mark the same rows);
// - root: the deepest block boundary strictly inside the prefix other conversations repeat (system prompt + tools,
//   GenerationJob::stable_prefix). Strictly inside, so the token at the mark (which a checkpoint with a draft head
//   records and matches on restore) is part of the repeated prefix;
// - end: the deepest block boundary a restore of this same prompt may use (tokens - tail).
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strix::session {

struct MarkPlanRequest {
    std::uint64_t tokens = 0;    // prompt length
    std::uint64_t restored = 0;  // rows already present before the pass (a restore); no mark at or below it
    std::uint32_t block = 0;     // checkpoint granularity in rows, > 0
    std::uint64_t stride = 0;    // grid spacing in rows, a multiple of block; 0: no grid
    std::uint64_t root = 0;      // tokens of the repeated prefix (system + tools); 0: no root mark
    std::uint64_t tail = 2;      // rows a restoring request still prefills after a mark, >= 1
};

// Deepest row a restore of a prompt of `tokens` may resume at (0 when the prompt is not longer than `tail`).
[[nodiscard]] constexpr std::uint64_t mark_limit(std::uint64_t tokens, std::uint64_t tail) noexcept {
    return tokens > tail ? tokens - tail : 0;
}

// Root mark of a prompt: the deepest block boundary p with p < root and p <= mark_limit(tokens, tail); 0: none.
[[nodiscard]] constexpr std::uint64_t root_mark(std::uint64_t root, std::uint64_t tokens, std::uint32_t block,
                                                std::uint64_t tail) noexcept {
    if (!block || !root) return 0;
    const auto limit = mark_limit(tokens, tail);
    const auto inside = root - 1;
    return (inside < limit ? inside : limit) / block * block;
}

[[nodiscard]] inline std::vector<std::uint64_t> plan_marks(const MarkPlanRequest& r) {
    if (!r.block) throw std::invalid_argument("mark plan: block must be positive");
    if (r.stride % r.block) throw std::invalid_argument("mark plan: stride must be 0 or a multiple of block");
    if (!r.tail) throw std::invalid_argument("mark plan: tail must be at least 1");
    const auto last = mark_limit(r.tokens, r.tail) / r.block * r.block;
    std::vector<std::uint64_t> marks;
    if (last <= r.restored) return marks;
    const auto root = root_mark(r.root, r.tokens, r.block, r.tail);
    bool root_pending = root > r.restored && root < last;
    if (r.stride) {
        for (auto p = (r.restored / r.stride + 1) * r.stride; p < last; p += r.stride) {
            if (root_pending && root <= p) {
                if (root < p) marks.push_back(root);
                root_pending = false;
            }
            marks.push_back(p);
        }
    }
    if (root_pending) marks.push_back(root);
    marks.push_back(last);
    return marks;
}

}  // namespace strix::session
