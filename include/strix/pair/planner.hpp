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
//   - the host gap per join segment (wake-up + relaunch after a host wait): 0 by contract for a device-waited route
//     (pair::route_waits_on_host false), else the measured value; unmeasured on a host-waited route refuses the form;
//   - the fixed cost the stage model cannot see (launches, host gaps) enters as stages: callers that measured a one-unit
//     pass append its residual with split::with_measured_single before calling plan() (no second formula here);
//   - memory fit: Request::resident_bytes larger than a unit's known memory refuses the single form on that unit
//     ("does not fit"); a pair-only model then has no single bar and the fastest admissible two-unit form wins;
//   - a GAIN BAR: a multi-device form is admissible only if ms x (1 + uncertainty) < single x (1 - margin);
//   - a reason string on every refusal and one receipt line with counts and bytes, so prediction and measurement compare.
// Prints only: callers still choose their form (an engine binds the chosen form in its own change, with its measurement).
// Host test: tests/pair_planner_tests.cpp.
// Home: shared (host shared). Bound by: the qwen35 dense HIP tensor-parallel decode placement (measured trip + measured
// one-node pass); other pair engines describe their pass as split::Stage rows and call the same entry.
#include "strix/device/set.hpp"
#include "strix/pair/device_gain.hpp"
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
#include <string_view>
#include <utility>
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
    std::uint64_t resident_bytes = 0; // bytes one unit must hold to run the whole model alone (single form); 0 = not stated
    double hop_bytes_per_row = 0.0; // layers forms: residual row crossing a cut
    // decode_spec: the drafter (one unit's draft time per window) and the acceptance statistics.
    double draft_ms = 0.0;
    bool draft_reads_hidden = false;
    overlap::Acceptance acceptance{};   // fresh windows (drafted after the true ids and hidden rows)
    overlap::Acceptance speculative{};  // windows drafted on the guess; empty = same as fresh (ids-only drafter)
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
    Regime regime = Regime::decode_plain;
    std::uint32_t rows = 0, windows = 0;
    std::uint64_t device_hash = 0;
    double margin = 0.0;
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
    if (DeviceGain{single_ms, c.ms, c.ms * c.uncertainty}.clears(margin)) return;
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
    out.regime = q.regime;
    out.rows = pass.rows;
    out.windows = pass.windows;
    out.device_hash = device::digest(set);
    out.margin = costs.margin;
    out.all.resize(6);
    for (std::size_t f = 0; f < out.all.size(); ++f) out.all[f].form = static_cast<Form>(f);

    // single: every unit with a known read rate; the best one
    Candidate& one = out.all[static_cast<std::size_t>(Form::single)];
    std::string missing, no_fit;
    for (const auto& u : set.units) {
        if (q.resident_bytes > 0U && u.memory_bytes > 0U && u.memory_bytes < q.resident_bytes) {
            no_fit += (no_fit.empty() ? "" : ", ") + u.id + detail::fmt(" %.1f GB", double(u.memory_bytes) / 1e9);
            continue;
        }
        if (u.read_bytes_per_ms <= 0.0) { missing += (missing.empty() ? "" : ", ") + u.id; continue; }
        const double ms = split::single_pass_ms(model, pass, u.read_bytes_per_ms) * double(pass.windows) + draft;
        if (ms < one.ms) { one.ms = one.compute_ms = ms; one.units = {u.rank}; one.admissible = true; }
    }
    if (!one.admissible)
        one.reason = !missing.empty() ? "needs nodes." + missing + ".memory_gbytes_ps"
                                      : "does not fit: resident " + detail::fmt("%.1f GB", double(q.resident_bytes) / 1e9) + " > " + no_fit;
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
        else if (decode && route_waits_on_host(costs.route) && !costs.gap_measured[ri])
            ten.reason = std::string("needs the measured host gap per join segment of route ") + gather_route_name(costs.route);
        else {
            const auto e = split::tensor(model, pass, split::Costs{rates, tr.cost, 0.0, costs.allow_fast});
            ten.units = {q.a, q.b};
            ten.exact = e.exact;
            ten.trips = e.trips * pass.windows;
            ten.compute_ms = e.compute_ms + draft;
            ten.link_ms = e.link_ms;
            ten.gap_ms = route_waits_on_host(costs.route) ? double(ten.trips) * costs.segment_gap_us[ri] / 1000.0 : 0.0;
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
    else if (q.draft_reads_hidden && overlap::windows(q.speculative) == 0U)
        ovl.reason = "needs the acceptance of speculative windows (the drafter reads the target hidden)";
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
            overlap::Pair p;
            p.speed1 = rates[0] / rates[1];
            p.link = chain::Link{tr.cost.bytes_per_ms, tr.cost.bytes_per_ms, tr.cost.latency_ms};
            if (w.layer_ms.size() < 2U) ovl.reason = "one layer: nothing to cut";
            else {
                const auto e = overlap::overlapped(w, p, q.acceptance, q.speculative);
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
    std::uint64_t h = out.device_hash;
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

// An explicit transition offered by an engine. Footprint is peak additional resident bytes over the decode
// layout, including prefill-only weights, state and staging, in prefill rank order. No catalogue entry or default
// is inferred from the geometry; a launcher must implement the transition and qualify its state contract.
struct SplitPhaseOption {
    std::string_view name;
    Form prefill_form = Form::layers;
    Form decode_form = Form::tensor;
    std::array<std::uint64_t, 2> extra_resident_bytes{};
    std::uint64_t min_prompt_rows = 1;
    std::uint64_t max_prompt_rows = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t min_context_tokens = 0;
    std::uint64_t max_context_tokens = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_ready_requests = 1;
    std::uint64_t handoff_fixed_bytes = 0;
    std::uint64_t handoff_bytes_per_row = 0;
    std::uint32_t handoff_trips = 0;
    device::GapClass handoff_gap = device::GapClass::idle;
    double handoff_setup_ms = 0.0;   // state staging and validation on the critical path
    bool handoff_exact = false;
};

struct SplitPhaseRequest {
    std::string_view placement;   // required opt-in, matching an option name
    std::uint64_t prompt_rows = 0, context_tokens = 0, ready_requests = 1;
    std::array<std::uint64_t, 2> free_bytes{};   // per-rank headroom after the decode layout, in prefill rank order
};

struct SplitPhaseDecision {
    bool admissible = false, exact = false;
    Candidate prefill, decode;
    std::uint64_t handoff_bytes = 0;
    double handoff_ms = 0.0, prefill_with_handoff_ms = 0.0;
    std::string reason, line;
};

// Compose two evaluated phases. The incoming estimates keep their phase gain bars and the composite also charges the
// handoff against the prefill bar. `handoff_bytes` counts the payload on the critical link path (for each serialized
// exchange, the larger direction). A full-rate probe of the handoff route is required for transferred bytes.
[[nodiscard]] inline SplitPhaseDecision split_phase(const Costs& costs, const SplitPhaseRequest& q,
                                                   const SplitPhaseOption& option, const Plan& prefill,
                                                   const Plan& decode) {
    SplitPhaseDecision out;
    const auto refuse = [&](std::string reason) {
        out.reason = std::move(reason);
        out.line = "split_phase_refused: placement=" + std::string(q.placement) + " reason=\"" + out.reason + "\"";
        return out;
    };
    if (q.placement.empty() || q.placement != option.name) return refuse("needs explicit placement");
    if (q.prompt_rows == 0U || q.ready_requests == 0U || q.prompt_rows < option.min_prompt_rows ||
        q.prompt_rows > option.max_prompt_rows || q.context_tokens < option.min_context_tokens ||
        q.context_tokens > option.max_context_tokens || q.ready_requests > option.max_ready_requests)
        return refuse("outside the declared prompt, context or concurrency band");
    if (prefill.regime != Regime::prefill || decode.regime == Regime::prefill || prefill.rows == 0U ||
        (q.prompt_rows - 1U) / prefill.rows + 1U != prefill.windows)
        return refuse("needs matching prefill windows and a decode estimate");
    if (costs.set == nullptr || prefill.device_hash != device::digest(*costs.set) ||
        prefill.device_hash != decode.device_hash || prefill.margin != costs.margin || decode.margin != costs.margin)
        return refuse("needs both phases evaluated for the same device and gain bar");
    const auto pre = static_cast<std::size_t>(option.prefill_form);
    const auto dec = static_cast<std::size_t>(option.decode_form);
    if (pre >= prefill.all.size() || dec >= decode.all.size() || option.prefill_form == option.decode_form)
        return refuse("needs two distinct evaluated phase forms");
    out.prefill = prefill.all[pre];
    out.decode = decode.all[dec];
    if (!out.prefill.admissible) return refuse("prefill: " + out.prefill.reason);
    if (!out.decode.admissible) return refuse("decode: " + out.decode.reason);
    if (out.prefill.units.size() != 2U || out.decode.units != out.prefill.units ||
        out.prefill.units[0] == out.prefill.units[1] || out.prefill.units[0] >= costs.set->units.size() ||
        out.prefill.units[1] >= costs.set->units.size())
        return refuse("needs the same pair and rank order for both phases");
    for (std::size_t r = 0; r < option.extra_resident_bytes.size(); ++r)
        if (option.extra_resident_bytes[r] > q.free_bytes[r])
            return refuse("prefill and decode layouts exceed the declared headroom of " +
                          costs.set->units[out.prefill.units[r]].id);
    out.exact = out.prefill.exact && out.decode.exact && option.handoff_exact;
    if (!out.exact && !costs.allow_fast) return refuse("bit-changing transition needs allow_fast");
    if (option.handoff_bytes_per_row > (std::numeric_limits<std::uint64_t>::max() - option.handoff_fixed_bytes) / q.prompt_rows)
        return refuse("handoff bytes overflow");
    out.handoff_bytes = option.handoff_fixed_bytes + option.handoff_bytes_per_row * q.prompt_rows;
    if (out.handoff_bytes > 0U && option.handoff_trips == 0U) return refuse("handoff payload needs trips");
    if (!std::isfinite(option.handoff_setup_ms) || option.handoff_setup_ms < 0.0)
        return refuse("needs a finite nonnegative handoff setup cost");
    out.handoff_ms = option.handoff_setup_ms;
    if (option.handoff_trips != 0U) {
        const auto* link = costs.set->link(out.prefill.units[0], out.prefill.units[1]);
        const auto gap = static_cast<std::size_t>(option.handoff_gap);
        if (link == nullptr || gap >= device::kGapClasses || !link->cost_measured[gap] || !link->cost[gap].valid() ||
            (out.handoff_bytes > 0U && link->cost[gap].us_per_byte == 0.0))
            return refuse("needs a measured handoff route in its gap class, with a bulk rate for payload");
        const auto& trip = link->cost[gap];
        out.handoff_ms += trip.latency_ms() * static_cast<double>(option.handoff_trips) +
                          trip.us_per_byte * static_cast<double>(out.handoff_bytes) / 1000.0;
    }
    out.prefill_with_handoff_ms = out.prefill.ms + out.handoff_ms;
    const DeviceGain prefill_gain{prefill.single_ms, out.prefill_with_handoff_ms,
                                  out.prefill.ms * out.prefill.uncertainty};
    const DeviceGain decode_gain{decode.single_ms, out.decode.ms, out.decode.ms * out.decode.uncertainty};
    if (!decode_gain.clears(costs.margin)) return refuse("decode does not clear the single-device gain bar");
    if (!prefill_gain.clears(costs.margin))
        return refuse("prefill plus handoff does not clear the single-device gain bar");
    out.admissible = true;
    out.line = "split_phase: placement=" + std::string(option.name) + " prefill_form=" + form_name(out.prefill.form) +
               " decode_form=" + form_name(out.decode.form) + " exact=" + (out.exact ? "E0" : "FAST") +
               " prompt_rows=" + std::to_string(q.prompt_rows) + " context_tokens=" + std::to_string(q.context_tokens) +
               " ready_requests=" + std::to_string(q.ready_requests) + " devices=" + std::to_string(prefill.device_hash) +
               detail::fmt(" prefill_ms=%.3f", out.prefill.ms) + detail::fmt(" prefill_single_ms=%.3f", prefill.single_ms) +
               " prefill_trips=" + std::to_string(out.prefill.trips) + detail::fmt(" prefill_link_bytes=%.0f", out.prefill.link_bytes) +
               detail::fmt(" decode_ms_per_pass=%.3f", out.decode.ms) +
               detail::fmt(" decode_single_ms_per_pass=%.3f", decode.single_ms) +
               " decode_trips_per_pass=" + std::to_string(out.decode.trips) +
               detail::fmt(" decode_link_bytes_per_pass=%.0f", out.decode.link_bytes) +
               " handoff_bytes=" + std::to_string(out.handoff_bytes) + " handoff_trips=" + std::to_string(option.handoff_trips) +
               detail::fmt(" handoff_ms=%.3f", out.handoff_ms) + detail::fmt(" prefill_with_handoff_ms=%.3f", out.prefill_with_handoff_ms) +
               detail::fmt(" prefill_ratio=%.3f", prefill_gain.ratio()) +
               detail::fmt(" decode_ratio=%.3f", decode_gain.ratio()) + " evidence=prediction";
    return out;
}

}  // namespace strix::pair::planner
