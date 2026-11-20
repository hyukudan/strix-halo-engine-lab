#pragma once

// Exact tensor-parallel decode over N ranks: column shares and the step cost the planner compares with one decoder.
// Host only, model-agnostic.
//
// A dense projection's output columns are split in indivisible units: 64-column blocks of the IU4 core
// (core::dense_iu4::run_cols), or heads / head groups whose columns, state and attention must stay on one rank. Rank r
// computes the columns of its units with the one-rank arithmetic, then an all-gather joins the shares. Every column is
// computed by exactly one rank in the one-rank order, so the joined rows are bitwise the one-rank rows.
// Shares follow each rank's measured read rate (strix-topology-v1 bandwidth_gbytes_ps): a decode step reads the
// weights once, so the rank time is its bytes over its rate.
// Home: shared (host shared). Engines bind the units of their own projections.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::pair::tp {

struct Share {
    std::uint32_t begin = 0, end = 0;  // [begin, end) in units
    [[nodiscard]] std::uint32_t size() const noexcept { return end - begin; }
};

// `units` over ranks in proportion to `rates` (each finite and > 0): the floor of each exact quota, then the leftover
// units one each by largest remainder, ties to the lower rank. Shares are contiguous in rank order and cover
// [0, units) exactly once; a rank may get no unit.
[[nodiscard]] inline std::vector<Share> split_units(std::uint32_t units, std::span<const double> rates) {
    if (rates.empty()) throw std::invalid_argument("tp split: no rank");
    double total = 0.0;
    for (const double r : rates) {
        if (!std::isfinite(r) || r <= 0.0) throw std::invalid_argument("tp split: every rank rate must be finite and > 0");
        total += r;
    }
    const std::size_t n = rates.size();
    std::vector<std::uint32_t> count(n, 0U);
    std::vector<double> rem(n, 0.0);
    std::uint32_t given = 0U;
    for (std::size_t i = 0; i < n; ++i) {
        const double quota = static_cast<double>(units) * (rates[i] / total);
        const double whole = std::floor(quota);
        count[i] = static_cast<std::uint32_t>(std::min(whole, static_cast<double>(units)));
        rem[i] = quota - whole;
        given += count[i];
    }
    while (given > units) {  // rounding guard: take from the largest share, highest rank first
        std::size_t at = n - 1U;
        for (std::size_t i = n; i-- > 0;) if (count[i] > count[at]) at = i;
        --count[at];
        --given;
    }
    while (given < units) {
        std::size_t at = 0U;
        for (std::size_t i = 1; i < n; ++i) if (rem[i] > rem[at]) at = i;
        ++count[at];
        rem[at] = -1.0;  // one leftover unit per rank per pass
        ++given;
        if (std::all_of(rem.begin(), rem.end(), [](double r) { return r < 0.0; })) std::fill(rem.begin(), rem.end(), 0.0);
    }
    std::vector<Share> out(n);
    std::uint32_t at = 0U;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = {at, at + count[i]};
        at += count[i];
    }
    return out;
}

// The share in elements of `unit_elems` each (columns of a projection, rows of a state).
[[nodiscard]] inline Share scaled(const Share& s, std::uint32_t unit_elems) noexcept { return {s.begin * unit_elems, s.end * unit_elems}; }

// One all-gather of the decode step: the bytes each rank sends (its share of the rows being joined).
struct Trip {
    double bytes = 0.0;
};
// Link cost of one trip: fixed latency (producer + exchange + sync, measured) plus bytes over the useful rate.
struct TripCost {
    double latency_ms = 0.0;
    double bytes_per_ms = 0.0;  // 0: latency only
    [[nodiscard]] double ms(double bytes) const noexcept { return latency_ms + (bytes_per_ms > 0.0 ? bytes / bytes_per_ms : 0.0); }
};
struct StepModel {
    double split_bytes = 0.0;       // bytes read per step that the column / head shares divide (weights, KV, recurrent state)
    double replicated_bytes = 0.0;  // bytes every rank reads (norm gains, tiny projections kept whole)
    double fixed_ms = 0.0;          // per-step work that does not shrink with the share (small replicated kernels)
    std::vector<Trip> trips;        // all-gathers per step, in order
};
struct StepEstimate {
    double one_rank_ms = 0.0;       // the fastest single rank decoding alone
    std::uint32_t one_rank = 0;
    double group_ms = 0.0;          // max over ranks of compute, plus the trips (they are on every rank's critical path)
    double compute_ms = 0.0, link_ms = 0.0;
    bool group_wins = false;
};
// `rates` in bytes/ms per rank, `fraction` the share of split_bytes each rank owns (sums to 1, from split_units).
[[nodiscard]] inline StepEstimate estimate(const StepModel& m, std::span<const double> rates, std::span<const double> fraction,
                                           const TripCost& link) {
    if (rates.empty() || rates.size() != fraction.size()) throw std::invalid_argument("tp estimate: rates and fractions differ");
    StepEstimate e;
    e.one_rank_ms = std::numeric_limits<double>::infinity();
    for (std::size_t r = 0; r < rates.size(); ++r) {
        if (!(rates[r] > 0.0)) throw std::invalid_argument("tp estimate: rank rate must be > 0");
        const double alone = (m.split_bytes + m.replicated_bytes) / rates[r] + m.fixed_ms;
        if (alone < e.one_rank_ms) { e.one_rank_ms = alone; e.one_rank = static_cast<std::uint32_t>(r); }
        e.compute_ms = std::max(e.compute_ms, (m.split_bytes * fraction[r] + m.replicated_bytes) / rates[r] + m.fixed_ms);
    }
    for (const auto& t : m.trips) e.link_ms += link.ms(t.bytes);
    e.group_ms = rates.size() > 1U ? e.compute_ms + e.link_ms : e.one_rank_ms;
    e.group_wins = rates.size() > 1U && e.group_ms < e.one_rank_ms;
    return e;
}

// Per-rank fraction of `units` (for estimate()).
[[nodiscard]] inline std::vector<double> fractions(std::span<const Share> shares) {
    std::uint32_t total = 0U;
    for (const auto& s : shares) total += s.size();
    std::vector<double> f(shares.size(), 0.0);
    for (std::size_t i = 0; i < shares.size(); ++i) f[i] = total ? static_cast<double>(shares[i].size()) / total : 0.0;
    return f;
}

// Which ranks decode, for unequal nodes. With shares in proportion to the rates, a group's compute is the split bytes
// over the sum of its rates plus the replicated bytes over its slowest rate (+ fixed), and the trips do not depend on
// who is in the group; so for every group size k the k fastest ranks are the best group. The plan tries k = 1..N
// (k <= units) and keeps the fastest step; a tie keeps the smaller group. `ranks` lists the chosen ranks fastest first
// (stable for equal rates); `est` is that group's estimate with split_units shares over `units`.
struct GroupPlan {
    std::vector<std::uint32_t> ranks;
    StepEstimate est;
};
[[nodiscard]] inline GroupPlan choose_ranks(const StepModel& m, std::span<const double> rates, std::uint32_t units,
                                            const TripCost& link) {
    if (rates.empty() || units == 0U) throw std::invalid_argument("tp choose_ranks: no ranks or no units");
    std::vector<std::uint32_t> order(rates.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::uint32_t>(i);
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return rates[a] > rates[b]; });
    GroupPlan best;
    double best_ms = std::numeric_limits<double>::infinity();
    for (std::size_t k = 1; k <= order.size() && k <= units; ++k) {
        std::vector<double> r(k);
        for (std::size_t i = 0; i < k; ++i) r[i] = rates[order[i]];
        const auto shares = split_units(units, r);
        const auto e = estimate(m, r, fractions(shares), link);
        if (e.group_ms < best_ms) {
            best_ms = e.group_ms;
            best.ranks.assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k));
            best.est = e;
        }
    }
    return best;
}

}  // namespace strix::pair::tp
