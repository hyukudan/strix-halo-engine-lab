// pair::planner::plan: one entry over a device::Set; measured-cost rules, gain bar, reasons, receipt line.
// Synthetic devices and links only (no GPU): a two-node USB4 pair, a same-box PCIe pair, one node.
#include "strix/pair/planner.hpp"

#include <cmath>
#include <cstdio>
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
    // measured in-token trip on the channel, gap not yet measured for the route
    dv::set_measured(s, 0U, 1U, dv::GapClass::in_token, cost(32.0, 2.8));
    p = pl::plan(m, c, pl::Request{});
    check(has(form(p, pl::Form::tensor).reason, "needs the measured host gap per join segment of route channel"), "gap of the route needed");
    c.gap_measured = {true, false, true};
    c.segment_gap_us = {17.0, 0.0, 0.0};   // staged: 4.4 ms over 257 segments; channel: counter printed sync 0
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
}  // namespace

int main() {
    try {
        test_one_unit();
        test_qwen_dense_decode();
        test_gain_bar();
        test_exactness();
        test_prefill_layers();
        test_spec_overlap_inputs();
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
