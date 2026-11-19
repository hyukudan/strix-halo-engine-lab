// Overlapped speculative verify windows on a two-stage layer split (include/strix/pair/window_overlap.hpp).
#include "strix/pair/window_overlap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace ov = strix::pair::overlap;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
bool near(double a, double b, double tol = 1e-9) { return std::abs(a - b) <= tol * std::max(1.0, std::abs(b)); }

// h = bonus_hits / windows; R = 3 here (k = 0, 1, 2 proposals accepted).
ov::Acceptance acceptance(std::uint64_t full, std::uint64_t hits, std::uint64_t total) {
    ov::Acceptance a;
    a.accepted = {total - full, 0, full};
    a.bonus_hits = hits;
    return a;
}
ov::Window equal_layers(std::size_t n, double ms_each, double draft_ms) {
    ov::Window w;
    w.layer_ms.assign(n, ms_each);
    w.draft_ms = draft_ms;
    w.id_bytes_per_row = 0.0;  // latency-only links unless a case declares bytes and rates
    return w;
}
}  // namespace

int main() try {
    {  // acceptance statistics
        const ov::Acceptance a = acceptance(6, 4, 10);
        require(near(ov::hit_rate(a), 0.4), "h = bonus hits over all windows");
        require(near(ov::tokens_per_window(a), (4.0 * 1.0 + 6.0 * 3.0) / 10.0), "tokens per window = E[k + 1]");
        require(ov::hit_rate(ov::Acceptance{}) == 0.0, "no windows: h = 0 (unknown is not a hit)");
        bool threw = false;
        try { (void)ov::hit_rate(acceptance(2, 3, 10)); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "more bonus hits than full-accept windows is refused");
    }
    {  // hand-computed case: 4 x 10 ms layers, D 4, latency 1 ms, no bytes, h 0.5
        ov::Window w = equal_layers(4, 10.0, 4.0);
        ov::Pair p;
        p.link.latency_ms = 1.0;
        const ov::Acceptance a = acceptance(5, 5, 10);
        const ov::Estimate one = ov::single(w, p);
        require(near(one.ms, 44.0), "single = layers + draft");
        const ov::Estimate s = ov::split(w, p);
        require(near(s.ms, 46.0) && s.cut == 2U, "serial split = single + hop + back, balanced cut");
        require(s.ms > one.ms, "a serial layer split never decodes faster than one device");
        const ov::Estimate o = ov::overlapped(w, p, a);
        // cut 2: stage0 = 24, b = 20, ring (24 + 1 + 20 + 1) / 2 = 23 -> P_hit 24; P_ser = 4 + 40 + 2 = 46;
        // the doomed window outlasts hop + b + back (22) by w = 2 -> P_miss 48 (cut 1: E 38, cut 3: E 51)
        require(o.admissible && o.cut == 2U, "overlap admissible at cut 2");
        require(near(o.p_hit, 24.0) && near(o.p_serial, 46.0) && near(o.p_miss, 48.0), "overlap terms");
        require(near(o.ms, 36.0), "E = 0.5 x 24 + 0.5 x 48");
        require(near(o.h_min, 2.0 / 24.0), "h_min = loss / (gain + loss) = 2 / (22 + 2)");
        const ov::Choice c = ov::choose(w, p, a);
        require(c.chosen.form == ov::Form::overlap, "overlap chosen over single and serial split");
        require(near(o.tokens, 2.0) && near(o.ms_per_token, 18.0) && near(c.single.ms_per_token, 22.0),
                "ranked per committed token: T = 2 for every form");
        const ov::Estimate same = ov::overlapped(w, p, a, a);
        require(near(same.ms, o.ms) && near(same.h, 0.5) && near(same.tokens, 2.0), "speculative == fresh: the fresh model");
        // A hidden-fed drafter whose speculative windows never accept (h_S 0, T_S 1): h = 0.5 / 1.5, T = 5 / 3,
        // E = 24 / 3 + 2 x 48 / 3 = 40 -> 24 ms per token > P_ser / T_F = 23: overlap refused, one device chosen.
        const ov::Acceptance cold = acceptance(0, 0, 10);
        const ov::Estimate low = ov::overlapped(w, p, a, cold);
        require(!low.admissible && near(low.h, 1.0 / 3.0) && near(low.tokens, 5.0 / 3.0) && near(low.ms, 40.0) &&
                    low.reason.find("per committed token") != std::string::npos,
                "lower speculative acceptance flips overlap to refused");
        require(ov::choose(w, p, a, cold).chosen.form == ov::Form::single, "and one device decodes");
    }
    {  // the cut moves layers off the drafter's device: 40 x 1 ms, D 8, free link -> a + 8 = b at cut 16
        ov::Window w = equal_layers(40, 1.0, 8.0);
        ov::Pair p;
        const ov::Estimate o = ov::overlapped(w, p, acceptance(5, 5, 10));
        require(o.cut == 16U && near(o.p_hit, 24.0), "hit path balances D0 + a against b");
        require(near(o.ms, 0.5 * 24.0 + 0.5 * 48.0), "E at the shifted cut");
    }
    {  // break-even gate: a doomed window that delays the redo raises h_min above 0
        ov::Window w = equal_layers(4, 10.0, 30.0);  // stage 0 (D 30 + a) outruns the last stage: a miss waits for it
        ov::Pair p;
        p.link.latency_ms = 1.0;
        const ov::Estimate lo = ov::overlapped(w, p, acceptance(1, 1, 10));
        require(lo.h_min > 0.1, "a non-abortable doomed window has a break-even hit rate");
        require(!lo.admissible && lo.reason.find("break-even") != std::string::npos, "below break-even: refused with reason");
        const ov::Estimate hi = ov::overlapped(w, p, acceptance(9, 9, 10));
        require(hi.admissible && hi.h >= hi.h_min, "above break-even: admitted");
    }
    {  // no drafter and no memory: refused with reasons, the choice falls back
        ov::Window w = equal_layers(4, 10.0, 0.0);
        ov::Pair p;
        require(!ov::overlapped(w, p, acceptance(5, 5, 10)).admissible, "no drafter: no overlap");
        w.draft_ms = 4.0;
        w.state_bytes.assign(4, 1e6);
        w.copy_bytes_per_ms = 1e6;
        ov::Extra x;
        x.drafter_bytes = 1e9;
        p.free0 = 5e8;
        const ov::Choice c = ov::choose(w, p, acceptance(5, 5, 10), {}, x);
        require(!c.overlap.admissible && c.overlap.reason.find("memory") != std::string::npos, "memory refusal named");
        require(c.chosen.form == ov::Form::single, "falls back to one device");
        p.free0 = 2e9;
        const ov::Estimate o = ov::overlapped(w, p, acceptance(5, 5, 10), {}, x);
        // cut 1: s = 1 ms (1 MB at 1 MB/ms), stage0 = 15, b = 30 -> P_hit 30, P_miss = 44 + 1; cut 2 pays s = 2 and a
        // doomed window of 26 against b 20 (P_miss 52): the snapshot cost moves the cut towards stage 0.
        require(o.admissible && o.cut == 1U && near(o.p_hit, 30.0) && near(o.p_miss, 45.0),
                "snapshot of the stage-0 state is on stage 0's path and in the miss");
    }
    // Same planner, two link classes (illustrative costs: a 64-layer trunk, R = 4, 5120-wide FP16 residual rows).
    const double usb4_rate = 4.58e6;     // bytes/ms (4.58 GB/s), latency 0.07 ms (measured trip class)
    const double pcie_rate = 12.0e6;     // bytes/ms through pinned host memory, latency 0.03 ms (assumed)
    for (const bool usb4 : {true, false}) {
        ov::Window w = equal_layers(64, 1.0, 6.0);
        w.head_ms = 2.0;
        w.rows = 4U;
        w.hop_bytes_per_row = 5120.0 * 2.0;
        w.hidden_bytes_per_row = 5120.0 * 2.0;
        w.id_bytes_per_row = 4.0;
        ov::Pair p;
        p.link.forward_bytes_per_ms = p.link.reverse_bytes_per_ms = usb4 ? usb4_rate : pcie_rate;
        p.link.latency_ms = usb4 ? 0.07 : 0.03;
        const ov::Acceptance a = acceptance(5, 5, 10);  // h 0.5
        const ov::Choice none = ov::choose(w, p, a);
        require(none.chosen.form == ov::Form::overlap, "no tensor form: overlap beats one device at h 0.5");
        require(none.overlap.ms < 0.8 * none.single.ms, "overlap at h 0.5 saves > 20 % of the window");
        require(none.split.ms > none.single.ms, "the serial split loses on either link");
        // A tensor form with 2 joins per layer: compute halves, trips 128 x latency.
        const double tensor = (64.0 + 2.0 + 6.0) / 2.0 + 128.0 * p.link.latency_ms;
        const ov::Choice tp = ov::choose(w, p, a, {}, {}, tensor);
        require(tp.chosen.form == ov::Form::tensor, "a cheap-join tensor form wins over overlap at h 0.5");
        const double slow_tensor = (64.0 + 2.0 + 6.0) / 2.0 + 128.0 * 0.2;  // a 200 us trip
        require(ov::choose(w, p, a, {}, {}, slow_tensor).chosen.form == ov::Form::overlap,
                "a high-latency join hands the decode to overlap");
    }
    std::puts("pair_window_overlap_tests: OK");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "pair_window_overlap_tests: FAIL: %s\n", e.what());
    return 1;
}
