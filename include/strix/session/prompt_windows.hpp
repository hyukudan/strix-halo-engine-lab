#pragma once
// Batched prompt windows (backend-neutral): positions [pos, pos + count) become single lead steps up to the next aligned
// position, windows of at most rmax rows (each a multiple of the alignment and at least min_rows), and single tail steps
// for the rest (< min_rows). A binding states its Rule from what its row kernels admit (e.g. a grouped expert prefill
// that takes multiples of 8 from 16 rows, an indexer whose groups must close inside one window). The same windows run
// on every stage of a layer pipeline: a stage window's plan from an aligned start concatenates to the one-rank plan, so
// the pipeline's bits equal the one-rank prefill at the same rmax.
// Home: shared (host shared). Bound by: bindings that run the prompt in windows (one node and pipeline stages).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::session::prompt_windows {

// Unaligned, bounded prompt chunks. Only the final chunk needs a head result.
// A scalar reference binds max_rows=1. The caller owns append transactions
// and any recovery needed after a callback has committed part of the prompt.
template <class Token, class Append>
void append_chunks(std::span<const Token> prompt, std::size_t max_rows, Append&& append) {
    if (prompt.empty() || !max_rows) throw std::invalid_argument("invalid prompt chunk bounds");
    while (!prompt.empty()) {
        const auto rows = std::min(prompt.size(), max_rows);
        append(prompt.first(rows), rows == prompt.size());
        prompt = prompt.subspan(rows);
    }
}

struct Rule {
    std::uint32_t align = 8;      // window start and rows are multiples of align
    std::uint32_t min_rows = 16;  // smallest window (a multiple of align)
    [[nodiscard]] constexpr bool valid() const { return align > 0U && min_rows >= align && min_rows % align == 0U; }
};

[[nodiscard]] constexpr bool window_ok(const Rule& r, std::uint32_t rows) {
    return r.valid() && rows >= r.min_rows && rows % r.align == 0U;
}

struct Window {
    std::uint32_t p0 = 0, rows = 0;
};
struct Plan {
    std::uint32_t lead = 0;  // single steps before the first window (to an aligned position)
    std::vector<Window> windows;
    std::uint32_t tail = 0;  // single steps after the last window (< min_rows positions)
};

// rmax not admissible: every position is a step.
[[nodiscard]] inline Plan plan(const Rule& r, std::uint32_t pos, std::uint32_t count, std::uint32_t rmax) {
    Plan p;
    if (!window_ok(r, rmax)) {
        p.lead = count;
        return p;
    }
    p.lead = (r.align - pos % r.align) % r.align;
    if (p.lead > count) p.lead = count;
    std::uint32_t at = pos + p.lead, left = count - p.lead;
    while (left >= r.min_rows) {
        const std::uint32_t whole = left / r.align * r.align;
        const std::uint32_t rows = whole < rmax ? whole : rmax;
        p.windows.push_back({at, rows});
        at += rows;
        left -= rows;
    }
    p.tail = left;
    return p;
}

// One stage window [start, start + rows) of a layer pipeline (strix/pair/layer_pipeline.hpp layers::windows(prompt,
// rmax)): the windows over its admissible head, single steps for the rest.
[[nodiscard]] inline Plan stage_plan(const Rule& r, std::uint32_t start, std::uint32_t rows, std::uint32_t rmax) {
    if (!window_ok(r, rmax) || start % rmax != 0U || rows == 0U || rows > rmax)
        throw std::invalid_argument("prompt windows: stage start must be a multiple of the admissible width, 0 < rows <= width");
    return plan(r, start, rows, rmax);
}

}  // namespace strix::session::prompt_windows
