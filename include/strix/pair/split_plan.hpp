#pragma once

// How a model divides over ranks, ONE description for every regime (host only, model- and vendor-agnostic).
// A binding lists its pass as Stages in execution order: what each stage reads, on which axis it can be divided, and
// what must be exchanged after it so that every rank holds the rows the next stage needs. The planner turns that list
// into the per-regime cost of each candidate form and keeps the fastest admissible one:
//   single  - one rank runs the pass;
//   layers  - contiguous layer stages on a line of ranks (strix/pair/layer_pipeline.hpp): one row of one stream visits the
//             stages one after the other, prompt windows overlap as a flow shop (strix/pair/chain.hpp is the N-device form);
//   tensor  - every rank runs every stage on its share of the axis (strix/pair/column_split.hpp shares, strix/pair/
//             tp_layout.hpp joins), all ranks read their weights AT THE SAME TIME; joins are on every rank's critical path.
// Exactness is a property of the join, so the plan says per stage whether the split keeps the one-rank bits:
//   gather - all-gather of output shares, every element computed by one rank in the one-rank arithmetic: exact;
//   slots  - routed slots (experts selected per row) computed whole by one rank each, the slot rows exchanged unreduced
//            and combined by every rank over all slots in slot order (the one-rank combine): exact;
//   pick   - argmax candidates (value, index) folded in rank order, ties to the lowest index (tp_layout.hpp ArgPick): exact;
//   reduce - partial sums over a split K added in rank order: FAST (bits change unless the one-rank engine adds the same
//            partials); refused unless the caller allows bit-changing forms.
// Costs are measured, never assumed: per-rank read rate (bytes/ms), a per-stage efficiency (the fraction of that rate the
// stage's kernels reach), launch-bound fixed ms, and the exchange cost latency + bytes / rate (tp::TripCost). A segment
// between two joins ends with a barrier: its time is the slowest rank's, then the exchange.
// Home: shared (host shared). Bound by: qwen35 dense HIP exact TP decode (apps/hip_qwen35_decode_main.hip; rates from the
// topology declaration, trip cost measured at start-up: strix/pair/tp_inputs.hpp). Prefill windows and verify rows
// (speculation) use the same stages.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "strix/pair/column_split.hpp"

namespace strix::pair::split {

enum class Axis : std::uint8_t { replicated, heads, columns, slots, vocab };
enum class Join : std::uint8_t { none, gather, slots, pick, reduce };
[[nodiscard]] constexpr bool exact(Join j) noexcept { return j != Join::reduce; }

struct Stage {
    std::string_view name;
    std::uint32_t layer = 0;           // trunk layer the stage belongs to (layer cuts fall between layers)
    Axis axis = Axis::replicated;
    std::uint32_t units = 1;           // indivisible units on the axis: heads, 64-column blocks, vocab blocks; slots: per row
    double bytes = 0.0;                // read per pass by one rank alone, independent of rows (weights); slots: ONE slot
    double bytes_per_row = 0.0;        // read per row (KV, state), divided with the axis
    double ms_per_row = 0.0;           // compute per row by one rank alone (prefill windows); divided with the axis
    double extra_slot_bytes = 0.0;     // slots: one always-present unit (e.g. a shared expert), placed whole on rank 0
    std::uint32_t pool = 0;            // slots: bank size; R rows touch pool x (1 - (1 - units/pool)^R) distinct slots
    double efficiency = 1.0;           // fraction of the rank's read rate this stage's kernels reach (measured class)
    double fixed_ms = 0.0;             // launch-bound work per pass, on every rank whatever its share
    Join join = Join::none;            // exchange after the stage
    double join_bytes_per_row = 0.0;   // gather / reduce / pick: bytes of the joined row; slots: bytes of ONE slot row
    double extra_join_bytes = 0.0;     // slots: bytes per row of the always-present unit's output
};

enum class Regime : std::uint8_t { decode, verify, prefill };
struct Pass {
    Regime regime = Regime::decode;
    std::uint32_t rows = 1;            // rows per pass: 1 (decode), R (verify), window rows (prefill)
    std::uint32_t windows = 1;         // passes in the request (prefill windows); decode / verify: 1
};

enum class Form : std::uint8_t { single, layers, tensor };
[[nodiscard]] inline const char* form_name(Form f) noexcept {
    return f == Form::single ? "single" : f == Form::layers ? "layers" : "tensor";
}

// Distinct slots R rows touch: units for one row; for R rows the expected union over a uniform independent route, an
// UPPER bound (neighbouring rows' routes correlate). Rounded up, at most pool.
[[nodiscard]] inline std::uint32_t touched_slots(const Stage& s, std::uint32_t rows) {
    if (s.units == 0U || rows == 0U) throw std::invalid_argument("split: slot stage needs units > 0 and rows > 0");
    if (rows == 1U || s.pool == 0U) return s.units;
    if (s.units > s.pool) throw std::invalid_argument("split: slots per row exceed the pool");
    const double k = static_cast<double>(s.units), e = static_cast<double>(s.pool);
    const double u = e * (1.0 - std::pow(1.0 - k / e, static_cast<double>(rows)));
    return static_cast<std::uint32_t>(std::min(e, std::ceil(u - 1e-9)));
}

// Slots over ranks: rank 0 takes the always-present unit and s0 slots, ranks 1.. share the rest by rate (split_units);
// s0 minimises the slowest rank (ties: the smaller s0). Slot ORDER does not matter for the bits (the combine runs over all
// slots in slot order on every rank); counts are what the plan needs. Returns slots per rank.
[[nodiscard]] inline std::vector<std::uint32_t> assign_slots(std::uint32_t slots, double slot_bytes, double extra_bytes,
                                                             std::span<const double> rates) {
    if (rates.empty()) throw std::invalid_argument("split: no rank");
    std::vector<std::uint32_t> best(rates.size(), 0U);
    if (rates.size() == 1U) { best[0] = slots; return best; }
    double best_ms = std::numeric_limits<double>::infinity();
    for (std::uint32_t s0 = 0; s0 <= slots; ++s0) {
        const auto rest = tp::split_units(slots - s0, rates.subspan(1));
        double worst = (extra_bytes + static_cast<double>(s0) * slot_bytes) / rates[0];
        for (std::size_t r = 1; r < rates.size(); ++r)
            worst = std::max(worst, static_cast<double>(rest[r - 1U].size()) * slot_bytes / rates[r]);
        if (worst < best_ms) {
            best_ms = worst;
            best[0] = s0;
            for (std::size_t r = 1; r < rates.size(); ++r) best[r] = rest[r - 1U].size();
        }
    }
    return best;
}

struct Costs {
    std::vector<double> rates;         // read rate per rank, bytes/ms (measured); rank order = line order
    tp::TripCost link;                 // one exchange: latency + bytes / rate (measured on the pair transport)
    double hop_bytes_per_row = 0.0;    // layers form: residual row crossing a cut
    bool allow_fast = false;           // admit bit-changing joins (reduce)
};

struct Estimate {
    Form form = Form::single;
    bool admissible = true;
    bool exact = true;
    std::uint32_t ranks = 1, trips = 0, cut = 0;  // cut: layers form, first layer of the second stage
    double ms = 0.0;                   // whole request: windows x pass (tensor, single) or flow shop (layers)
    double compute_ms = 0.0, link_ms = 0.0;
    std::vector<double> rank_bytes;    // bytes each rank reads per pass (tensor)
};

namespace detail {
[[nodiscard]] inline double stage_ms(const Stage& s, double bytes, double row_ms, double rate) {
    if (!(s.efficiency > 0.0) || s.efficiency > 1.0) throw std::invalid_argument("split: stage efficiency must be in (0, 1]");
    return std::max(bytes / (rate * s.efficiency), row_ms) + s.fixed_ms;
}
[[nodiscard]] inline double alone_bytes(const Stage& s, std::uint32_t rows) {
    const double rowb = s.bytes_per_row * static_cast<double>(rows);
    return s.axis == Axis::slots ? static_cast<double>(touched_slots(s, rows)) * s.bytes + s.extra_slot_bytes + rowb
                                 : s.bytes + rowb;
}
inline void check(std::span<const Stage> stages, const Costs& c) {
    if (stages.empty()) throw std::invalid_argument("split: no stage");
    if (c.rates.empty()) throw std::invalid_argument("split: no rank");
    for (const double r : c.rates)
        if (!std::isfinite(r) || r <= 0.0) throw std::invalid_argument("split: every rank rate must be finite and > 0");
    for (std::size_t i = 0; i < stages.size(); ++i) {
        const auto& s = stages[i];
        if (s.axis != Axis::replicated && s.units == 0U) throw std::invalid_argument("split: a divided stage needs units");
        if (i && s.layer < stages[i - 1U].layer) throw std::invalid_argument("split: stages must be in layer order");
        if (s.bytes < 0.0 || s.bytes_per_row < 0.0 || s.ms_per_row < 0.0 || s.fixed_ms < 0.0 || s.join_bytes_per_row < 0.0)
            throw std::invalid_argument("split: negative cost");
    }
}
}  // namespace detail

// One pass on rank `r` alone (single form), ms.
[[nodiscard]] inline double single_pass_ms(std::span<const Stage> stages, const Pass& p, double rate) {
    double t = 0.0;
    for (const auto& s : stages)
        t += detail::stage_ms(s, detail::alone_bytes(s, p.rows), s.ms_per_row * static_cast<double>(p.rows), rate);
    return t;
}

// Tensor form over all ranks of `c.rates`: per stage each rank's share (split_units over the axis units; slots by
// assign_slots), a barrier + exchange at every join.
[[nodiscard]] inline Estimate tensor(std::span<const Stage> stages, const Pass& p, const Costs& c) {
    detail::check(stages, c);
    const std::size_t n = c.rates.size();
    const double rows = static_cast<double>(p.rows);
    Estimate e;
    e.form = Form::tensor;
    e.ranks = static_cast<std::uint32_t>(n);
    e.rank_bytes.assign(n, 0.0);
    std::vector<double> seg(n, 0.0);
    double pass_compute = 0.0, pass_link = 0.0;
    for (const auto& s : stages) {
        std::vector<double> frac(n, 1.0), sent(n, 0.0);
        std::vector<double> bytes(n, 0.0);
        if (s.axis == Axis::replicated) {
            for (std::size_t r = 0; r < n; ++r) bytes[r] = detail::alone_bytes(s, p.rows);
        } else if (s.axis == Axis::slots) {
            const auto touched = touched_slots(s, p.rows);
            const auto count = assign_slots(touched, s.bytes, s.extra_slot_bytes, c.rates);
            for (std::size_t r = 0; r < n; ++r) {
                frac[r] = static_cast<double>(count[r]) / static_cast<double>(touched);
                bytes[r] = static_cast<double>(count[r]) * s.bytes + (r == 0U ? s.extra_slot_bytes : 0.0) +
                           s.bytes_per_row * rows * frac[r];
                // rows of the slots this rank computed (each row's selected slots spread like the touched set)
                sent[r] = rows * (static_cast<double>(s.units) * frac[r] * s.join_bytes_per_row + (r == 0U ? s.extra_join_bytes : 0.0));
            }
        } else {
            const auto f = tp::fractions(tp::split_units(s.units, c.rates));
            for (std::size_t r = 0; r < n; ++r) {
                frac[r] = f[r];
                bytes[r] = (s.bytes + s.bytes_per_row * rows) * f[r];
            }
        }
        if (n > 1U && s.join != Join::none && s.axis != Axis::slots) {
            for (std::size_t r = 0; r < n; ++r)
                sent[r] = s.join == Join::gather ? rows * s.join_bytes_per_row * frac[r] : rows * s.join_bytes_per_row;
        }
        for (std::size_t r = 0; r < n; ++r) {
            e.rank_bytes[r] += bytes[r];
            seg[r] += detail::stage_ms(s, bytes[r], s.ms_per_row * rows * frac[r], c.rates[r]);
        }
        if (n > 1U && s.join != Join::none) {
            if (!exact(s.join)) e.exact = false;
            pass_compute += *std::max_element(seg.begin(), seg.end());
            std::fill(seg.begin(), seg.end(), 0.0);
            pass_link += c.link.ms(*std::max_element(sent.begin(), sent.end()));
            ++e.trips;
        }
    }
    pass_compute += *std::max_element(seg.begin(), seg.end());
    const double w = static_cast<double>(std::max(p.windows, 1U));
    e.compute_ms = pass_compute * w;
    e.link_ms = pass_link * w;
    e.ms = e.compute_ms + e.link_ms;
    e.admissible = e.exact || c.allow_fast;
    return e;
}

// Layers form over the first two ranks of `c.rates`: rank 0 runs the stages of layers [0, cut), rank 1 the rest; the
// cut minimises the slower stage (ties: the earliest cut). A row of one stream visits both stages in order; prompt
// windows overlap as a two-stage flow shop: a + b + (windows - 1) x max(a, b) + windows x hop.
[[nodiscard]] inline Estimate layers(std::span<const Stage> stages, const Pass& p, const Costs& c) {
    detail::check(stages, c);
    Estimate e;
    e.form = Form::layers;
    if (c.rates.size() < 2U) { e.admissible = false; return e; }
    e.ranks = 2U;
    const std::uint32_t last = stages.back().layer;
    double best = std::numeric_limits<double>::infinity();
    for (std::uint32_t cut = stages.front().layer + 1U; cut <= last; ++cut) {
        double a = 0.0, b = 0.0;
        for (const auto& s : stages) {
            const double t = detail::stage_ms(s, detail::alone_bytes(s, p.rows), s.ms_per_row * static_cast<double>(p.rows),
                                              c.rates[s.layer < cut ? 0U : 1U]);
            (s.layer < cut ? a : b) += t;
        }
        const double hop = c.link.ms(c.hop_bytes_per_row * static_cast<double>(p.rows));
        const double w = static_cast<double>(std::max(p.windows, 1U));
        const double ms = a + b + (w - 1.0) * std::max(a, b) + w * hop;
        if (std::max(a, b) < best) {
            best = std::max(a, b);
            e.cut = cut;
            e.compute_ms = a + b + (w - 1.0) * std::max(a, b);
            e.link_ms = w * hop;
            e.ms = ms;
            e.trips = 1U;
        }
    }
    if (!std::isfinite(best)) e.admissible = false;  // one layer: nothing to cut
    return e;
}

struct Plan {
    Estimate chosen;
    Estimate single, layers, tensor;
    std::uint32_t single_rank = 0;
};
// Every admissible form for this pass; the fastest wins, a tie keeps the simpler form (single, then layers).
[[nodiscard]] inline Plan plan(std::span<const Stage> stages, const Pass& p, const Costs& c) {
    detail::check(stages, c);
    Plan out;
    out.single.form = Form::single;
    out.single.ms = std::numeric_limits<double>::infinity();
    for (std::size_t r = 0; r < c.rates.size(); ++r) {
        const double ms = single_pass_ms(stages, p, c.rates[r]) * static_cast<double>(std::max(p.windows, 1U));
        if (ms < out.single.ms) {
            out.single.ms = out.single.compute_ms = ms;
            out.single_rank = static_cast<std::uint32_t>(r);
        }
    }
    out.chosen = out.single;
    out.layers = layers(stages, p, c);
    out.tensor = tensor(stages, p, c);
    if (c.rates.size() > 1U) {
        if (out.layers.admissible && out.layers.ms < out.chosen.ms) out.chosen = out.layers;
        if (out.tensor.admissible && out.tensor.ms < out.chosen.ms) out.chosen = out.tensor;
    }
    return out;
}

// A dense trunk as the exact column-split binding knows it: per layer one stage of `bytes_per_layer` on `units` column
// units, then `gathers_per_layer` all-gathers of `gather_bytes_per_row` each; then `tail_joins` picks (argmax) after the
// head. Any engine whose TP decode is "every projection split by columns, an all-gather after each" describes itself so.
[[nodiscard]] inline std::vector<Stage> uniform_layers(std::uint32_t layers, double bytes_per_layer, std::uint32_t units,
                                                       std::uint32_t gathers_per_layer, double gather_bytes_per_row,
                                                       std::uint32_t tail_joins) {
    if (layers == 0U || units == 0U || gathers_per_layer == 0U) throw std::invalid_argument("split: uniform trunk needs layers, units, gathers");
    std::vector<Stage> s;
    s.reserve(static_cast<std::size_t>(layers) * gathers_per_layer + tail_joins);
    for (std::uint32_t l = 0; l < layers; ++l)
        for (std::uint32_t g = 0; g < gathers_per_layer; ++g) {
            Stage st;
            st.name = "trunk";
            st.layer = l;
            st.axis = Axis::columns;
            st.units = units;
            st.bytes = g == 0U ? bytes_per_layer : 0.0;
            st.join = Join::gather;
            st.join_bytes_per_row = gather_bytes_per_row;
            s.push_back(st);
        }
    for (std::uint32_t t = 0; t < tail_joins; ++t) {
        Stage st;
        st.name = "pick";
        st.layer = layers - 1U;
        st.axis = Axis::vocab;
        st.units = units;
        st.join = Join::pick;
        s.push_back(st);
    }
    return s;
}

}  // namespace strix::pair::split
