#pragma once
// ONE entry for multi-device placement: plan(model, costs, request) over one resolved device::Set and one cost input
// (host only, model- and vendor-agnostic). The existing form evaluators are called, not copied:
//   single, tensor, layers      strix/pair/split_plan.hpp (single_pass_ms, tensor, layers)
//   layers_overlap              strix/pair/window_overlap.hpp (overlapped: speculative windows on a layer split)
//   role, replica               not bound yet: answered "needs binding", never silently absent.
// What this entry adds on top of the evaluators:
//   - unknown is not free: a decode form needs the MEASURED trip cost of its link (device::Set::set_measured) and the
//     measured host gap per join segment of its gather route; missing numbers refuse the form and name the key;
//   - the trip of a tensor join is priced with the gap class its segment's compute implies; when only a faster class
//     was measured it is used with +25 % uncertainty and the line says so;
//   - the host gap per join segment (wake-up + relaunch after a host wait: zero only for a route measured at zero);
//   - a GAIN BAR: a multi-device form is admissible only if ms x (1 + uncertainty) < single x (1 - margin);
//   - a reason string on every refusal and one receipt line with counts and bytes, so prediction and measurement compare.
// Prints only: callers still choose their form (an engine binds the chosen form in its own change, with its measurement).
// Host test: tests/pair_planner_tests.cpp.
// Home: shared (host shared). Bound by: none yet (engines describe their pass as split::Stage rows and print the line).
#include "strix/device/set.hpp"
#include "strix/pair/gather_link.hpp"
#include "strix/pair/split_plan.hpp"
#include "strix/pair/window_overlap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::pair::planner {

enum class Regime : std::uint8_t { prefill, decode_plain, decode_spec };
enum class Form : std::uint8_t { single, layers, layers_overlap, tensor, role, replica };
[[nodiscard]] constexpr const char* regime_name(Regime r) noexcept {
    return r == Regime::prefill ? "prefill" : r == Regime::decode_plain ? "decode_plain" : "decode_spec";
}
[[nodiscard]] constexpr const char* form_name(Form f) noexcept {
    constexpr const char* n[] = {"single", "layers", "layers_overlap", "tensor", "role", "replica"};
    return n[static_cast<std::size_t>(f)];
}

struct Costs {
    const device::Set* set = nullptr;
    // Host gap per join segment by gather route (staged, device_direct, channel), us: the empty-queue wake-up and
    // relaunch a host wait costs. Legal only when measured (e.g. the `tp link:` counters); 0 measured = no gap.
    std::array<double, 3> segment_gap_us{};
    std::array<bool, 3> gap_measured{};
    GatherRoute route = GatherRoute::staged;
    double margin = 0.05;           // gain bar: a multi-device form must beat the best single by this fraction
    bool allow_fast = false;        // admit bit-changing joins (reduce); FAST profile only
};

struct Request {
    Regime regime = Regime::decode_plain;
    std::uint32_t rows = 1;         // rows per pass: 1 (plain), R (verify window), prompt window rows (prefill)
    std::uint32_t windows = 1;      // passes in the request (prefill windows); decode: 1
    std::uint32_t a = 0, b = 1;     // the two units a two-unit form uses (line order)
    double hop_bytes_per_row = 0.0; // layers forms: residual row crossing a cut
    // decode_spec: the drafter (one unit's draft time per window) and the acceptance statistics.
    double draft_ms = 0.0;
    bool draft_reads_hidden = false;
    overlap::Acceptance acceptance{};
    bool forced = false; Form force = Form::single;   // override: still printed with its prediction
};

struct Candidate {
    Form form = Form::single;
    bool admissible = false;
    bool exact = true;
    std::string reason;             // always set when !admissible; notes (priced class) when admissible
    std::vector<std::uint32_t> units;
    std::uint32_t cut = 0;          // layers forms: first layer of the second stage
    double ms = std::numeric_limits<double>::infinity();   // whole request (prefill) / per pass or window (decode)
    double compute_ms = 0.0, link_ms = 0.0, gap_ms = 0.0;
    std::uint32_t trips = 0;
    double link_bytes = 0.0;        // bytes on the wire (largest sender per join), whole request
    double uncertainty = 0.0;       // relative
};

struct Plan {
    Candidate chosen;
    std::vector<Candidate> all;     // every form, in Form order
    double single_ms = std::numeric_limits<double>::infinity();
    std::uint64_t hash = 0;
    std::string line;               // format_line(*this)
};

namespace detail {
inline std::string fmt(const char* f, double v) {
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}
inline split::Pass pass_of(const Request& q) {
    split::Pass p;
    p.regime = q.regime == Regime::prefill ? split::Regime::prefill : q.regime == Regime::decode_spec ? split::Regime::verify
                                                                                                         : split::Regime::decode;
    p.rows = std::max(q.rows, 1U);
    p.windows = q.regime == Regime::prefill ? std::max(q.windows, 1U) : 1U;
    return p;
}
// Trip cost for a gap class: the measured class, else the nearest faster measured class with +25 % uncertainty.
struct Trip { bool known = false; tp::TripCost cost; double uncertainty = 0.0; std::string note; };
inline Trip trip_for(const device::SetLink& l, device::GapClass want, bool decode) {
    Trip t;
    for (int g = static_cast<int>(want); g >= 0; --g) {
        if (!l.cost_measured[static_cast<std::size_t>(g)]) continue;
        const auto& c = l.cost[static_cast<std::size_t>(g)];
        t.known = true;
        t.cost = tp::TripCost{c.latency_us / 1000.0, c.bytes_per_ms()};
        if (g != static_cast<int>(want)) {
            t.uncertainty = 0.25;
            t.note = std::string("trip class ") + device::gap_class_name(want) + " priced with " +
                     device::gap_class_name(static_cast<device::GapClass>(g)) + " (+25 %)";
        }
        return t;
    }
    if (!decode && l.bytes_per_ms > 0.0) {   // prefill hand-off may use the declared / nominal link with its uncertainty
        t.known = true;
        t.cost = tp::TripCost{l.rtt_us / 1000.0, l.bytes_per_ms};
        t.uncertainty = std::max(l.uncertainty, l.src == device::RateSource::measured ? 0.0 : 0.3);
        t.note = std::string("link ") + device::rate_source_name(l.src) + " (not a measured trip)";
    }
    return t;
}
inline double bytes_of(const split::Estimate& e, const tp::TripCost& c, std::uint32_t passes) {
    if (c.bytes_per_ms <= 0.0) return 0.0;
    return std::max(0.0, e.link_ms - c.latency_ms * double(e.trips) * double(passes)) * c.bytes_per_ms;
}
inline void gain_bar(Candidate& c, double single_ms, double margin) {
    if (!c.admissible) return;
    if (!std::isfinite(single_ms)) return;   // no single fits: the bar does not apply
    if (c.ms * (1.0 + c.uncertainty) < single_ms * (1.0 - margin)) return;
    c.admissible = false;
    c.reason = "refused: " + fmt("%.1f", c.ms) + " ms" + (c.uncertainty > 0.0 ? fmt(" (+%.0f %% uncertainty)", 100.0 * c.uncertainty) : "") +
               " vs single " + fmt("%.1f", single_ms) + " ms (gain bar " + fmt("%.0f", 100.0 * margin) + " %; " +
               std::to_string(c.trips) + " trips, link " + fmt("%.1f", c.link_ms) + " ms, gap " + fmt("%.1f", c.gap_ms) + " ms)" +
               (c.reason.empty() ? "" : "; " + c.reason);
}
inline std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xFFU; h *= 1099511628211ULL; }
    return h;
}
}  // namespace detail

namespace detail {
inline std::string make_line(const Plan& p, Regime regime, const device::Set& set, GatherRoute route) {
    const auto& c = p.chosen;
    std::string units;
    for (const auto u : c.units) units += (units.empty() ? "" : ",") + set.units.at(u).id;
    char hash[24];
    std::snprintf(hash, sizeof hash, "%016llx", static_cast<unsigned long long>(p.hash));
    return std::string("plan: regime=") + regime_name(regime) + " form=" + form_name(c.form) + " units=" + units +
           " exact=" + (c.exact ? "E0" : "FAST") + detail::fmt(" pred_ms=%.1f", c.ms) + detail::fmt(" single_ms=%.1f", p.single_ms) +
           detail::fmt(" ratio=%.2f", std::isfinite(p.single_ms) && c.ms > 0.0 ? p.single_ms / c.ms : 0.0) +
           detail::fmt(" compute_ms=%.1f", c.compute_ms) + detail::fmt(" link_ms=%.1f", c.link_ms) + detail::fmt(" gap_ms=%.1f", c.gap_ms) +
           " trips=" + std::to_string(c.trips) + detail::fmt(" link_bytes=%.0f", c.link_bytes) +
           (c.form == Form::tensor ? std::string(" route=") + (route == GatherRoute::staged ? "staged" : route == GatherRoute::device_direct ? "device_direct" : "channel") : "") +
           " hash=" + hash + (c.reason.empty() ? "" : " note=\"" + c.reason + "\"");
}
}  // namespace detail

// Every form for this request over the units of `costs.set`; the fastest admissible wins, ties keep the simpler form.
[[nodiscard]] inline Plan plan(std::span<const split::Stage> model, const Costs& costs, const Request& q) {
    if (costs.set == nullptr || costs.set->units.empty()) throw std::invalid_argument("planner: no device set");
    if (model.empty()) throw std::invalid_argument("planner: empty model description");
    const auto& set = *costs.set;
    const split::Pass pass = detail::pass_of(q);
    const bool decode = q.regime != Regime::prefill;
    const bool spec = q.regime == Regime::decode_spec;
    const double draft = spec ? q.draft_ms : 0.0;   // per window, one unit (the same term for single, tensor, layers)
    Plan out;
    out.all.resize(6);
    for (std::size_t f = 0; f < out.all.size(); ++f) out.all[f].form = static_cast<Form>(f);

    // single: every unit with a known read rate; the best one
    Candidate& one = out.all[static_cast<std::size_t>(Form::single)];
    std::string missing;
    for (const auto& u : set.units) {
        if (u.read_bytes_per_ms <= 0.0) { missing += (missing.empty() ? "" : ", ") + u.id; continue; }
        const double ms = split::single_pass_ms(model, pass, u.read_bytes_per_ms) * double(pass.windows) + draft;
        if (ms < one.ms) { one.ms = one.compute_ms = ms; one.units = {u.rank}; one.admissible = true; }
    }
    if (!one.admissible) one.reason = "needs nodes." + missing + ".memory_gbytes_ps";
    out.single_ms = one.ms;

    // two-unit forms share their prerequisites
    std::string two_reason;
    const device::SetLink* link = nullptr;
    if (set.units.size() < 2U) two_reason = "one unit";
    else if (q.a >= set.units.size() || q.b >= set.units.size() || q.a == q.b) two_reason = "units a / b not in the set";
    else if (set.units[q.a].read_bytes_per_ms <= 0.0 || set.units[q.b].read_bytes_per_ms <= 0.0) two_reason = "needs both units' memory_gbytes_ps";
    else if ((link = set.link(q.a, q.b)) == nullptr) two_reason = "needs links between " + set.units[q.a].id + " and " + set.units[q.b].id;
    const std::vector<double> rates = two_reason.empty() ? std::vector<double>{set.units[q.a].read_bytes_per_ms, set.units[q.b].read_bytes_per_ms}
                                                         : std::vector<double>{};

    // tensor: split::tensor + host gap per join segment of the route
    Candidate& ten = out.all[static_cast<std::size_t>(Form::tensor)];
    if (!two_reason.empty()) ten.reason = two_reason;
    else {
        // segment compute with a free link decides the gap class of its trips
        const auto free_est = split::tensor(model, pass, split::Costs{rates, tp::TripCost{}, 0.0, true});
        const double seg_us = free_est.trips > 0U ? 1000.0 * free_est.compute_ms / double(free_est.trips * pass.windows) : 0.0;
        const auto tr = detail::trip_for(*link, device::gap_class(seg_us), decode);
        const auto ri = static_cast<std::size_t>(costs.route);
        if (!tr.known) ten.reason = "needs links." + link->name + " measured trip cost (" + device::gap_class_name(device::gap_class(seg_us)) + ")";
        else if (decode && !costs.gap_measured[ri]) ten.reason = std::string("needs the measured host gap per join segment of route ") + gather_route_name(costs.route);
        else {
            const auto e = split::tensor(model, pass, split::Costs{rates, tr.cost, 0.0, costs.allow_fast});
            ten.units = {q.a, q.b};
            ten.exact = e.exact;
            ten.trips = e.trips * pass.windows;
            ten.compute_ms = e.compute_ms + draft;
            ten.link_ms = e.link_ms;
            ten.gap_ms = double(ten.trips) * costs.segment_gap_us[ri] / 1000.0;
            ten.ms = ten.compute_ms + ten.link_ms + ten.gap_ms;
            ten.link_bytes = detail::bytes_of(e, tr.cost, pass.windows);
            ten.uncertainty = tr.uncertainty;
            ten.reason = tr.note;
            ten.admissible = e.admissible;
            if (!e.admissible) ten.reason = "bit-changing join (reduce) needs allow_fast";
        }
    }

    // layers: split::layers (one hop per pass / window)
    Candidate& lay = out.all[static_cast<std::size_t>(Form::layers)];
    if (!two_reason.empty()) lay.reason = two_reason;
    else {
        const auto tr = detail::trip_for(*link, decode ? device::GapClass::in_token : device::GapClass::back_to_back, decode);
        if (!tr.known) lay.reason = "needs links." + link->name + " measured trip cost";
        else {
            const auto e = split::layers(model, pass, split::Costs{rates, tr.cost, q.hop_bytes_per_row, costs.allow_fast});
            if (!e.admissible) lay.reason = "one layer: nothing to cut";
            else {
                lay.admissible = true;
                lay.units = {q.a, q.b};
                lay.cut = e.cut;
                lay.trips = e.trips * pass.windows;
                lay.compute_ms = e.compute_ms + draft;
                lay.link_ms = e.link_ms;
                lay.ms = e.ms + draft;
                lay.link_bytes = q.hop_bytes_per_row * double(pass.rows) * double(pass.windows);
                lay.uncertainty = tr.uncertainty;
                lay.reason = tr.note;
                if (decode && lay.ms >= one.ms) {
                    lay.admissible = false;
                    lay.reason = "decode: a + hop + b " + detail::fmt("%.1f", lay.ms) + " ms >= single " + detail::fmt("%.1f", one.ms) + " ms";
                }
            }
        }
    }

    // layers_overlap: speculative windows on a layer split (decode_spec with a drafter and its statistics only)
    Candidate& ovl = out.all[static_cast<std::size_t>(Form::layers_overlap)];
    if (!spec) ovl.reason = "speculative decode only";
    else if (!two_reason.empty()) ovl.reason = two_reason;
    else if (overlap::windows(q.acceptance) == 0U) ovl.reason = "needs the acceptance histogram";
    else {
        const auto tr = detail::trip_for(*link, device::GapClass::in_token, true);
        if (!tr.known) ovl.reason = "needs links." + link->name + " measured trip cost";
        else {
            overlap::Window w;
            const std::uint32_t first = model.front().layer, last = model.back().layer;
            w.layer_ms.assign(last - first + 1U, 0.0);
            for (const auto& s : model) {
                const split::Stage one_stage[] = {s};
                w.layer_ms[s.layer - first] += split::single_pass_ms(one_stage, pass, rates[0]);
            }
            w.draft_ms = q.draft_ms;
            w.rows = pass.rows;
            w.hop_bytes_per_row = q.hop_bytes_per_row;
            w.draft_reads_hidden = q.draft_reads_hidden;
            overlap::Pair p;
            p.speed1 = rates[0] / rates[1];
            p.link = chain::Link{tr.cost.bytes_per_ms, tr.cost.bytes_per_ms, tr.cost.latency_ms};
            if (w.layer_ms.size() < 2U) ovl.reason = "one layer: nothing to cut";
            else {
                const auto e = overlap::overlapped(w, p, q.acceptance);
                ovl.units = {q.a, q.b};
                ovl.cut = e.cut;
                ovl.ms = e.ms;
                ovl.compute_ms = e.p_serial;
                ovl.uncertainty = tr.uncertainty;
                ovl.admissible = e.admissible;
                ovl.reason = e.admissible ? "per window; h " + detail::fmt("%.2f", e.h) + " >= h_min " + detail::fmt("%.2f", e.h_min) : e.reason;
            }
        }
    }
    out.all[static_cast<std::size_t>(Form::role)].reason = "needs binding (role placement is not bound yet)";
    out.all[static_cast<std::size_t>(Form::replica)].reason = "needs binding (serve replicas are not bound yet)";

    // gain bar, then the fastest admissible (Form order breaks ties: the simpler form)
    for (auto& c : out.all) if (c.form != Form::single) detail::gain_bar(c, out.single_ms, costs.margin);
    out.chosen = one;
    for (const auto& c : out.all)
        if (c.admissible && c.ms < out.chosen.ms) out.chosen = c;
    if (q.forced) {
        const auto& f = out.all[static_cast<std::size_t>(q.force)];
        out.chosen = f;
        out.chosen.reason = "forced" + (f.admissible ? std::string() : "; would be refused: " + f.reason);
    }
    std::uint64_t h = device::digest(set);
    h = detail::mix(h, static_cast<std::uint64_t>(out.chosen.form));
    h = detail::mix(h, out.chosen.cut);
    for (const auto u : out.chosen.units) h = detail::mix(h, u);
    h = detail::mix(h, out.chosen.exact ? 1U : 0U);
    out.hash = h;
    out.line = detail::make_line(out, q.regime, set, costs.route);
    return out;
}


// The receipt line (same text from every caller): counts and bytes, so prediction and measurement compare.
[[nodiscard]] inline std::string format_line(const Plan& p) { return p.line; }
// One line per form that was not chosen and is not admissible: why.
[[nodiscard]] inline std::vector<std::string> refusal_lines(const Plan& p) {
    std::vector<std::string> out;
    for (const auto& c : p.all)
        if (!c.admissible && c.form != p.chosen.form) out.push_back(std::string("plan_refused: form=") + form_name(c.form) + " reason=\"" + c.reason + "\"");
    return out;
}

}  // namespace strix::pair::planner
