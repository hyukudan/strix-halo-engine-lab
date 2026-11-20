#pragma once
// Exact tensor-parallel decode over N ranks: the rank-local layouts of head-split tensors and the all-gather join.
// Host only, model-agnostic. Companion of column_split.hpp (which decides how many units each rank owns).
//
// A rank owns contiguous unit ranges (64-column blocks, heads). Head-split tensors can own several global runs: a
// GDN-style recurrence whose value head h reads key head h % key_heads gives a rank that owns key heads [kb, ke) the
// value heads {h : h % key_heads in [kb, ke)}. The rank keeps them in a compact local order (the list order of its
// segments). After its kernels wrote local rows [rows][width_q] into slot q of a gather buffer, an all-gather fills
// every slot and one join pass places every global column: joined[r][c] = slots[base[c] + r * stride[c]]. Each
// column comes from exactly one rank, word for word, so the joined row is the one-rank row.
// Home: shared (host shared). Engines describe their own projections with these segments.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "strix/pair/column_split.hpp"

namespace strix::pair::tp {

struct Segment {
    std::uint32_t global = 0, count = 0;  // global units [global, global + count), in local order
};

[[nodiscard]] inline std::uint32_t width(std::span<const Segment> s) noexcept {
    std::uint32_t w = 0U;
    for (const auto& x : s) w += x.count;
    return w;
}

// Value heads owned with key heads `keys` when value head h reads key head h % key_heads. Local value head
// j = m * nk + i (nk = keys.size()) is global m * key_heads + keys.begin + i, so local j reads local key j % nk:
// the kernel's own modulo rule holds on the compact arrays (key_heads = nk, value_heads = width).
[[nodiscard]] inline std::vector<Segment> mod_value_heads(std::uint32_t key_heads, std::uint32_t value_heads, Share keys) {
    if (key_heads == 0U || value_heads % key_heads != 0U || keys.end > key_heads || keys.begin > keys.end)
        throw std::invalid_argument("tp: value heads must be a multiple of key heads and the key share inside them");
    std::vector<Segment> out;
    if (keys.size() == 0U) return out;
    for (std::uint32_t m = 0; m < value_heads / key_heads; ++m) out.push_back({m * key_heads + keys.begin, keys.size()});
    return out;
}

// Segments of units -> segments of elements (unit_elems per unit), shifted by `offset` elements.
[[nodiscard]] inline std::vector<Segment> expand(std::span<const Segment> s, std::uint32_t unit_elems, std::uint32_t offset = 0U) {
    std::vector<Segment> out;
    out.reserve(s.size());
    for (const auto& x : s) out.push_back({offset + x.global * unit_elems, x.count * unit_elems});
    return out;
}

// Vocabulary head columns per rank. The head is split in `unit`-column blocks only when its format allows a column
// share (`splittable`) and V is a whole number of blocks; otherwise it is replicated and every rank owns [0, V) (each
// rank runs the whole head and its own argmax). Either way the union of what the ranks compute covers every column:
// split shares tile [0, V) exactly once, replicated ranks each cover it.
struct HeadColumns {
    bool split = false;
    std::vector<Segment> cols;  // per rank: [global, global + count) columns
};
[[nodiscard]] inline HeadColumns head_columns(bool splittable, std::uint32_t vocab, std::span<const double> rates,
                                              std::uint32_t unit = 64U) {
    if (vocab == 0U || unit == 0U || rates.empty()) throw std::invalid_argument("tp: head needs a vocabulary, a unit and ranks");
    HeadColumns h;
    h.split = splittable && vocab % unit == 0U;
    if (!h.split) {
        h.cols.assign(rates.size(), Segment{0U, vocab});
        return h;
    }
    for (const auto& s : split_units(vocab / unit, rates)) h.cols.push_back({s.begin * unit, s.size() * unit});
    return h;
}

// local position -> global element (gather of a replicated full row into the compact local order).
[[nodiscard]] inline std::vector<std::int32_t> select_map(std::span<const Segment> s) {
    std::vector<std::int32_t> out;
    out.reserve(width(s));
    for (const auto& x : s)
        for (std::uint32_t i = 0; i < x.count; ++i) out.push_back(static_cast<std::int32_t>(x.global + i));
    return out;
}

// Join of an all-gather. Slot q starts at q * slot_stride (elements) and holds rank q's rows [rows][width_q].
// Every global column in [0, columns) must be owned by exactly one rank.
struct JoinMap {
    std::vector<std::int32_t> base, stride;  // per global column
    std::size_t slot_stride = 0;
    std::uint32_t max_width = 0;             // the largest rank width: bytes of one exchanged row = max_width * elem
};
[[nodiscard]] inline JoinMap join_map(std::span<const std::vector<Segment>> per_rank, std::uint32_t columns, std::size_t slot_stride) {
    JoinMap m;
    m.slot_stride = slot_stride;
    m.base.assign(columns, -1);
    m.stride.assign(columns, 0);
    for (std::size_t q = 0; q < per_rank.size(); ++q) {
        const std::uint32_t w = width(per_rank[q]);
        if (w > m.max_width) m.max_width = w;
        std::uint32_t j = 0U;
        for (const auto& s : per_rank[q])
            for (std::uint32_t i = 0; i < s.count; ++i, ++j) {
                const std::uint32_t c = s.global + i;
                if (c >= columns || m.base[c] >= 0) throw std::invalid_argument("tp join: a column outside the row or owned twice");
                const std::size_t at = q * slot_stride + j;
                if (at > std::size_t(std::numeric_limits<std::int32_t>::max())) throw std::invalid_argument("tp join: slots too large");
                m.base[c] = static_cast<std::int32_t>(at);
                m.stride[c] = static_cast<std::int32_t>(w);
            }
    }
    for (std::uint32_t c = 0; c < columns; ++c)
        if (m.base[c] < 0) throw std::invalid_argument("tp join: a column no rank owns");
    return m;
}

// Argmax of a row split over ranks. The one-rank rule (argmax_kernel): the largest value, ties to the lowest index,
// a value wins only if it is > -inf (so NaN and -inf never win; a row without one keeps the sentinel index). That is a
// total order on the candidates, so the pick does not depend on how the row is partitioned: each rank reports its
// (value, global index) and every rank folds the reports in rank order with `beats`.
struct ArgPick {
    float value = -std::numeric_limits<float>::infinity();
    std::int32_t index = 0x7fffffff;
};
[[nodiscard]] inline bool beats(const ArgPick& a, const ArgPick& b) noexcept {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
}

}  // namespace strix::pair::tp
