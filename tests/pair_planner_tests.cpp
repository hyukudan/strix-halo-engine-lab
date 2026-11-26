// pair::planner::plan: one entry over a device::Set; measured-cost rules, gain bar, reasons, receipt line.
// Synthetic devices and links only (no GPU): a two-node USB4 pair, a same-box PCIe pair, one node.
#include "strix/pair/planner.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
namespace dv = strix::device;
namespace pl = strix::pair::planner;
namespace sp = strix::pair::split;
using strix::pair::GatherRoute;

dv::Set two_units(double gbps, const char* cls, double link_gbps, const char* host_b = "host-b") {
    dv::Set s;
    s.source = "<synthetic>";
    for (std::uint32_t r = 0; r < 2U; ++r) {
        dv::Unit u;
        u.id = r == 0U ? "a" : "b";
        u.host = r == 0U ? "host-a" : host_b;
        u.rank = r;
        u.read_bytes_per_ms = gbps * 1e6;
        u.read_src = dv::FactSource::declared;
        s.units.push_back(u);
    }
    dv::SetLink l;
    l.name = "ab";
    l.a = 0U;
    l.b = 1U;
    l.kind = cls;
    l.bytes_per_ms = link_gbps * 1e6;
    l.rtt_us = 30.0;
    l.src = dv::RateSource::nominal;
    l.uncertainty = 0.3;
    s.links.push_back(l);
    return s;
}
strix::dist::LinkCost cost(double latency_us, double gbps) {
    strix::dist::LinkCost c;
    c.latency_us = latency_us;
    c.us_per_byte = gbps > 0.0 ? 1000.0 / (gbps * 1e6) : 0.0;
    return c;
}
const pl::Candidate& form(const pl::Plan& p, pl::Form f) { return p.all[static_cast<std::size_t>(f)]; }
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// Qwen 27B dense as its TP decode app describes one pass: 64 layers, 14.0 GB read per pass, 4 gathers of 22,016 B rows
// per layer + the argmax pick (257 joins). Receipt numbers: one rank ~57.4 ms/pass at 243 GB/s; channel trip 32 us at
// 2.8 GB/s; staged 48.7 us at 2.3 GB/s.
std::vector<sp::Stage> qwen_dense() { return sp::uniform_layers(64U, 14.0e9 / 64.0, 1024U, 4U, 22016.0, 1U); }

void test_one_unit() {
    dv::Set s = two_units(243.0, "usb4", 4.58);
    s.units.pop_back();
    s.links.clear();
    const auto m = qwen_dense();
    pl::Costs c;
    c.set = &s;
    const auto p = pl::plan(m, c, pl::Request{});
    check(p.chosen.form == pl::Form::single && std::abs(p.single_ms - 14.0e9 / 243.0e6) < 1e-6, "one unit: single, bytes / rate");
    check(form(p, pl::Form::tensor).reason == "one unit" && form(p, pl::Form::layers).reason == "one unit", "two-unit forms: 'one unit'");
    check(has(p.line, "plan: regime=decode_plain form=single units=a exact=E0"), "receipt line names the form and unit");
}

void test_qwen_dense_decode() {
    dv::Set s = two_units(243.0, "usb4", 4.58);
    const auto m = qwen_dense();
    pl::Costs c;
    c.set = &s;
    c.route = GatherRoute::channel;
    // nothing measured: the decode forms refuse and say what they need
    auto p = pl::plan(m, c, pl::Request{});
    check(p.chosen.form == pl::Form::single, "no measured trip: single");
    check(has(form(p, pl::Form::tensor).reason, "needs links.ab measured trip cost (in_token)"), "tensor names the missing trip class");
    bool listed = false;
    for (const auto& l : pl::refusal_lines(p)) listed = listed || has(l, "plan_refused: form=tensor reason=\"needs links.ab");
    check(listed, "refusal lines print why");
    // measured in-token trip; the host-waited staged route still needs its measured gap per join segment
    dv::set_measured(s, 0U, 1U, dv::GapClass::in_token, cost(32.0, 2.8));
    c.route = GatherRoute::staged;
    p = pl::plan(m, c, pl::Request{});
    check(has(form(p, pl::Form::tensor).reason, "needs the measured host gap per join segment of route staged"), "staged: gap of the route needed");
    // the device-waited channel returns without a host wait: gap 0 by contract, nothing to measure
    c.route = GatherRoute::channel;
    p = pl::plan(m, c, pl::Request{});
    check(p.chosen.form == pl::Form::tensor && form(p, pl::Form::tensor).gap_ms == 0.0, "channel: no host gap needed");
    c.gap_measured = {true, false, true};
    c.segment_gap_us = {17.0, 0.0, 5.0};   // staged: 4.4 ms over 257 segments; a channel value is never priced (no host wait)
    p = pl::plan(m, c, pl::Request{});
    const auto& t = form(p, pl::Form::tensor);
    check(p.chosen.form == pl::Form::tensor && t.trips == 257U && t.exact, "channel: tensor, 257 exact joins");
    // compute 57.6 / 2 = 28.8; link 256 x (32 us + 11,008 B / 2.8 GB/s) + 32 us = 9.23 ms; gap 0
    check(std::abs(t.compute_ms - 28.81) < 0.05 && std::abs(t.link_ms - 9.23) < 0.05 && t.gap_ms == 0.0, "tensor arithmetic");
    check(std::abs(t.link_bytes - 256.0 * 11008.0) < 1.0, "link bytes = largest share per gather");
    check(has(p.line, "form=tensor units=a,b exact=E0") && has(p.line, "trips=257") && has(p.line, "route=channel"), "receipt line");
    check(has(form(p, pl::Form::layers).reason, "decode: a + hop + b"), "layers decode refused when single fits");
    check(has(form(p, pl::Form::role).reason, "needs binding") && has(form(p, pl::Form::replica).reason, "needs binding"), "role / replica not silent");
    // staged route: + 257 x 17 us of host gap
    c.route = GatherRoute::staged;
    const auto ps = pl::plan(m, c, pl::Request{});
    check(std::abs(form(ps, pl::Form::tensor).gap_ms - 257.0 * 0.017) < 1e-9, "staged: host gap per join segment");
    // only the back-to-back class measured: priced with it, +25 % uncertainty, the line says so
    dv::Set s2 = two_units(243.0, "usb4", 4.58);
    dv::set_measured(s2, 0U, 1U, dv::GapClass::back_to_back, cost(32.0, 2.8));
    c.set = &s2;
    c.route = GatherRoute::channel;
    const auto pb = pl::plan(m, c, pl::Request{});
    check(form(pb, pl::Form::tensor).uncertainty == 0.25 && has(form(pb, pl::Form::tensor).reason, "in_token priced with back_to_back"),
          "faster class stands in with its uncertainty");
    check(pb.chosen.form == pl::Form::tensor, "38 ms x 1.25 still clears the bar against 57.6 ms");
    const auto again = pl::plan(m, c, pl::Request{});
    check(again.hash == pb.hash && again.line == pb.line, "same inputs: same hash and line");
}

// 20 layers of 1 ms at 1 GB/s, one latency-only gather per layer: single 20 ms, tensor 10 ms + 20 trips.
void test_gain_bar() {
    const auto m = sp::uniform_layers(20U, 1.0e6, 1024U, 1U, 0.0, 0U);
    for (const double lat_us : {470.0, 440.0, 400.0}) {
        dv::Set s = two_units(1.0, "pcie4x16", 25.0, "host-a");   // same box
        dv::set_measured(s, 0U, 1U, dv::GapClass::in_token, cost(lat_us, 0.0));
        pl::Costs c;
        c.set = &s;
        c.gap_measured = {true, true, true};
        const auto p = pl::plan(m, c, pl::Request{});
        const auto& t = form(p, pl::Form::tensor);
        if (lat_us == 470.0) check(!t.admissible && has(t.reason, "refused: 19.4 ms vs single 20.0 ms"), "3 % gain refused by the 5 % bar");
        if (lat_us == 440.0) check(t.admissible && p.chosen.form == pl::Form::tensor, "6 % gain, measured class: admitted");
        if (lat_us == 400.0) check(t.admissible && std::abs(t.ms - 18.0) < 1e-9, "10 % gain admitted");
    }
    dv::Set s = two_units(1.0, "pcie4x16", 25.0, "host-a");
    dv::set_measured(s, 0U, 1U, dv::GapClass::back_to_back, cost(440.0, 0.0));
    pl::Costs c;
    c.set = &s;
    c.gap_measured = {true, true, true};
    const auto p = pl::plan(m, c, pl::Request{});
    check(!form(p, pl::Form::tensor).admissible && has(form(p, pl::Form::tensor).reason, "+25 % uncertainty"), "6 % gain with +25 % uncertainty refused");
}

void test_exactness() {
    auto m = sp::uniform_layers(20U, 1.0e6, 1024U, 1U, 0.0, 0U);
    m[5].join = sp::Join::reduce;
    dv::Set s = two_units(1.0, "pcie4x16", 25.0, "host-a");
    dv::set_measured(s, 0U, 1U, dv::GapClass::in_token, cost(100.0, 0.0));
    pl::Costs c;
    c.set = &s;
    c.gap_measured = {true, true, true};
    auto p = pl::plan(m, c, pl::Request{});
    check(!form(p, pl::Form::tensor).admissible && has(form(p, pl::Form::tensor).reason, "needs allow_fast"), "bit-changing join refused");
    c.allow_fast = true;
    p = pl::plan(m, c, pl::Request{});
    check(p.chosen.form == pl::Form::tensor && has(p.line, "exact=FAST"), "allow_fast admits it, labelled FAST");
}

void test_prefill_layers() {
    dv::Set s = two_units(243.0, "usb4", 4.58);   // nothing measured: prefill may use the nominal link with its uncertainty
    const auto m = qwen_dense();
    pl::Costs c;
    c.set = &s;
    pl::Request q;
    q.regime = pl::Regime::prefill;
    q.rows = 512U;
    q.windows = 16U;
    q.hop_bytes_per_row = 5120.0 * 4.0;
    const auto p = pl::plan(m, c, q);
    const auto& l = form(p, pl::Form::layers);
    check(p.chosen.form == pl::Form::layers && l.cut == 32U && l.uncertainty >= 0.3, "prefill 8K: layers over the nominal link");
    check(p.single_ms / l.ms > 1.7 && p.single_ms / l.ms < 1.9, "prefill layers ratio 1.7-1.9");
    check(!form(p, pl::Form::tensor).admissible && has(form(p, pl::Form::tensor).reason, "refused:"), "tensor prefill refused by its link time");
}

void test_spec_overlap_inputs() {
    dv::Set s = two_units(243.0, "usb4", 4.58);
    dv::set_measured(s, 0U, 1U, dv::GapClass::in_token, cost(32.0, 2.8));
    const auto m = qwen_dense();
    pl::Costs c;
    c.set = &s;
    c.gap_measured = {true, true, true};
    pl::Request q;
    q.regime = pl::Regime::decode_spec;
    q.rows = 8U;
    q.draft_ms = 5.0;
    auto p = pl::plan(m, c, q);
    check(form(p, pl::Form::layers_overlap).reason == "needs the acceptance histogram", "overlap needs the acceptance histogram");
    q.acceptance.accepted = {2, 1, 1, 1, 1, 1, 1, 2};
    q.acceptance.bonus_hits = 1;
    q.draft_reads_hidden = true;
    p = pl::plan(m, c, q);
    check(has(form(p, pl::Form::layers_overlap).reason, "needs the acceptance of speculative windows"), "hidden-reading drafter needs its histogram");
    check(p.chosen.form == pl::Form::tensor, "speculative decode on the pair: tensor");
    q.forced = true;
    q.force = pl::Form::layers;
    p = pl::plan(m, c, q);
    check(p.chosen.form == pl::Form::layers && has(p.chosen.reason, "forced; would be refused"), "force prints the refusal it overrides");
}
// A pair-only model: the single form fits no unit (resident bytes > known memory), so no gain bar applies and the
// two-unit form wins on its own price; unknown memory is not a refusal (the rate decides as before).
void test_does_not_fit() {
    dv::Set s = two_units(243.0, "usb4", 4.58);
    for (auto& u : s.units) u.memory_bytes = 120000000000ULL;
    dv::set_measured(s, 0U, 1U, dv::GapClass::in_token, cost(22.0, 2.8));
    const auto m = qwen_dense();
    pl::Costs c;
    c.set = &s;
    c.route = GatherRoute::channel;
    pl::Request q;
    q.resident_bytes = 150000000000ULL;
    const auto p = pl::plan(m, c, q);
    check(!form(p, pl::Form::single).admissible && has(form(p, pl::Form::single).reason, "does not fit: resident 150.0 GB > a 120.0 GB, b 120.0 GB"),
          "single refused: does not fit, names the units");
    check(p.chosen.form == pl::Form::tensor && !std::isfinite(p.single_ms) && has(p.line, "form=tensor") && has(p.line, "single_ms=inf"),
          "pair-only: tensor chosen without a single bar");
    bool listed = false;
    for (const auto& l : pl::refusal_lines(p)) listed = listed || has(l, "plan_refused: form=single reason=\"does not fit");
    check(listed, "refusal line for the single form");
    q.resident_bytes = 100000000000ULL;
    check(pl::plan(m, c, q).single_ms < 60.0, "fits: single priced");
    for (auto& u : s.units) u.memory_bytes = 0U;
    q.resident_bytes = 150000000000ULL;
    check(pl::plan(m, c, q).all[0].admissible, "memory unknown: single priced by its rate");
}

// The Qwen dense TP decode binding: a measured one-node pass (62.0 ms) enters as a replicated fixed stage
// (split::with_measured_single), the chained channel trip (22 us) as the in-token class; single = the measured pass,
// tensor = half the bytes + the same fixed ms + 257 trips.
void test_calibrated_qwen_dense() {
    dv::Set s = two_units(243.0, "usb4", 4.58);
    dv::set_measured(s, 0U, 1U, dv::GapClass::in_token, cost(22.0, 2.8));
    auto m = qwen_dense();
    const auto cal = sp::with_measured_single(m, sp::Pass{}, 243.0e6, 62.0);
    check(std::abs(cal.fixed_ms - (62.0 - 14.0e9 / 243.0e6)) < 1e-9, "residual = measured - stage model");
    pl::Costs c;
    c.set = &s;
    c.route = GatherRoute::channel;
    const auto p = pl::plan(m, c, pl::Request{});
    const auto& t = form(p, pl::Form::tensor);
    check(std::abs(p.single_ms - 62.0) < 1e-9, "single = the measured one-node pass");
    // 28.81 + 4.39 fixed; link 256 x (22 + 3.93) + 22 us = 6.66 ms; gap 0 (channel)
    check(p.chosen.form == pl::Form::tensor && std::abs(t.compute_ms - (28.81 + cal.fixed_ms)) < 0.05 &&
              std::abs(t.link_ms - 6.66) < 0.05 && t.gap_ms == 0.0 && t.uncertainty == 0.0,
          "calibrated tensor arithmetic, in-token trip measured");
    check(std::abs(t.ms - 39.86) < 0.1 && has(p.line, "single_ms=62.0"), "prediction ~39.9 ms vs 62.0 one node");
}

void test_split_phase() {
    auto s = two_units(243.0, "usb4", 4.58);
    dv::set_measured(s, 0U, 1U, dv::GapClass::back_to_back, cost(26.0, 4.0));
    dv::set_measured(s, 0U, 1U, dv::GapClass::in_token, cost(32.0, 2.8));
    const auto m = qwen_dense();
    pl::Costs c;
    c.set = &s;
    c.route = GatherRoute::channel;
    pl::Request pf;
    pf.regime = pl::Regime::prefill;
    pf.rows = 512U;
    pf.windows = 16U;
    pf.hop_bytes_per_row = 20480.0;
    pl::SplitPhaseOption option;
    option.name = "pipeline-tensor";
    option.extra_resident_bytes = {64U, 128U};
    option.min_prompt_rows = 2048U;
    option.max_context_tokens = 16384U;
    option.handoff_bytes_per_row = 4096U;
    option.handoff_fixed_bytes = 256U;
    option.handoff_trips = 16U;
    option.handoff_setup_ms = 0.75;
    option.handoff_exact = true;
    pl::SplitPhaseRequest q;
    q.prompt_rows = 8192U;
    q.context_tokens = 8192U;
    q.free_bytes = {64U, 128U};
    auto prefill = pl::plan(m, c, pf);
    auto decode = pl::plan(m, c, {});
    auto p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "explicit placement"), "split phase is opt-in");
    q.placement = option.name;
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "measured handoff route"), "split phase needs the measured gap class");
    dv::set_measured(s, 0U, 1U, dv::GapClass::idle, cost(80.0, 0.0));
    prefill = pl::plan(m, c, pf);
    decode = pl::plan(m, c, {});
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "bulk rate"), "latency-only handoff probe cannot price payload");
    dv::set_measured(s, 0U, 1U, dv::GapClass::idle, cost(80.0, 2.0));
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "same device"), "split phase refuses estimates from stale calibration");
    prefill = pl::plan(m, c, pf);
    decode = pl::plan(m, c, {});
    p = pl::split_phase(c, q, option, prefill, decode);
    const double transfer_ms = 0.75 + 16 * 0.08 + 33554688.0 / 2.0e6;
    check(p.admissible && p.exact && p.handoff_bytes == 33554688U && std::abs(p.handoff_ms - transfer_ms) < 1e-9,
          "split phase charges setup, serialized trips and bytes");
    check(std::abs(p.prefill_with_handoff_ms - p.prefill.ms - transfer_ms) < 1e-9 &&
          has(p.line, "prefill_form=layers decode_form=tensor") && has(p.line, "handoff_bytes=33554688 handoff_trips=16") &&
          has(p.line, "evidence=prediction"), "split phase receipt identifies phases and prediction");
    q.free_bytes[1] = 127U;
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "headroom of b"), "each rank must fit its peak co-resident layout");
    q.free_bytes[1] = 128U;
    q.ready_requests = 2U;
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "concurrency band"), "option refuses unpriced concurrency");
    q.ready_requests = 1U;
    q.context_tokens = 32768U;
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "context"), "option refuses unpriced context");
    q.context_tokens = 8192U;
    q.prompt_rows = 4096U;
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "prefill windows"), "handoff rows and prefill window count must match");
    q.prompt_rows = 8192U;
    option.handoff_exact = false;
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "allow_fast"), "handoff must preserve bits under EXACT");
    option.handoff_exact = true;
    option.handoff_fixed_bytes = std::numeric_limits<std::uint64_t>::max();
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "overflow"), "handoff byte calculation cannot wrap");
    option.handoff_fixed_bytes = 256U;
    option.handoff_bytes_per_row = 200000U;
    p = pl::split_phase(c, q, option, prefill, decode);
    check(!p.admissible && has(p.reason, "single-device gain bar"), "handoff cost can erase prefill savings");
}
}  // namespace

int main() {
    try {
        test_one_unit();
        test_qwen_dense_decode();
        test_gain_bar();
        test_exactness();
        test_prefill_layers();
        test_spec_overlap_inputs();
        test_does_not_fit();
        test_calibrated_qwen_dense();
        test_split_phase();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "pair planner tests: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("pair planner tests: all passed");
    return 0;
}
