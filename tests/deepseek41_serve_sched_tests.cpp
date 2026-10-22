#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/deepseek41_serve_sched.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

// The serve's decode-priority scheduling rules (deepseek41_serve_sched.hpp): deferred admission decisions and the unit
// plan of an interleaved prefill (unit ends on multiples of 8, never a unit or a remainder shorter than min_rows, every
// default piece end kept, the prompt covered exactly).
namespace {

namespace ds = strix::deepseek41;
namespace sc = strix::deepseek41::sched;

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void admission() {
    sc::Options o;
    o.mode = sc::Mode::defer;
    expect(sc::admit(o, 8192, 0, 0.0, 0.0) == sc::Admit::idle, "nobody decodes: admit");
    expect(sc::admit(o, 8192, 4, 0.0, 15.5) == sc::Admit::defer, "8K next to 4 streams with 15.5 s left: defer (4 x 14.0 s > 15.5 s)");
    expect(sc::admit(o, 512, 4, 0.0, 15.5) == sc::Admit::cost, "512 next to 4 streams with 15.5 s left: admit (4 x 1.04 s <= 15.5 s)");
    expect(sc::admit(o, 8192, 4, 30.0, 15.5) == sc::Admit::starved, "waited the starvation bound: admit");
    expect(sc::admit(o, 8192, 4, 1.0, 15.5, true) == sc::Admit::draining, "draining: admit");
    const std::vector<std::uint64_t> rem{10, 232, 0};
    expect(sc::wait_estimate(rem, 0.067) > 15.5 && sc::wait_estimate(rem, 0.067) < 15.6, "W = largest remaining x step time");
    expect(sc::tdev_seconds(o, 2048) > 3.6 && sc::tdev_seconds(o, 2048) < 3.65, "T_dev(2K) = 0.17 + 2048 x 1.69 ms");
}

// Walk a prompt with next_unit_end and check the units.
void units(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block, std::uint32_t max_rows, std::uint32_t budget,
           std::uint32_t min_rows) {
    const auto ends = ds::plan_prefill_pieces(resume, shared, total, block, max_rows, min_rows);
    const std::string tag = "units(" + std::to_string(resume) + "," + std::to_string(total) + ", block " + std::to_string(block) + ", budget " +
                            std::to_string(budget) + ")";
    std::uint32_t at = resume, n = 0;
    std::vector<std::uint32_t> seen;
    while (at < total && n < 100000U) {
        const auto e = sc::next_unit_end(at, ends, budget, min_rows);
        expect(e > at && e <= total, tag + ": unit end inside the prompt");
        expect(e == total || e % 8U == 0U, tag + ": unit end on a multiple of 8");
        expect(e - at >= min_rows, tag + ": unit of at least min_rows rows");
        expect(budget == 0U || e - at <= budget || std::binary_search(ends.begin(), ends.end(), e), tag + ": unit within the budget or a default end");
        seen.push_back(e);
        at = e;
        ++n;
    }
    expect(at == total, tag + ": prompt covered");
    for (const auto d : ends) expect(std::binary_search(seen.begin(), seen.end(), d), tag + ": default end " + std::to_string(d) + " kept");
    if (budget == 0U) expect(seen == ends, tag + ": budget 0 = the default pieces");
}

}  // namespace

// STRIX_DS41_SERVE_SCHED x prefill path (P17.2: unset = interleave for the device prefill, off for rows; defer /
// interleave with rows rejected up front).
void modes() {
    const auto throws = [](const std::string& m, bool device) {
        try {
            (void)sc::mode_for(m, device);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    expect(sc::mode_for("", true) == sc::Mode::interleave, "unset + device: interleave");
    expect(sc::mode_for("", false) == sc::Mode::off, "unset + rows: off");
    for (const bool device : {true, false}) {
        expect(sc::mode_for("off", device) == sc::Mode::off && sc::mode_for("0", device) == sc::Mode::off, "off / 0: off on either path");
        expect(throws("bogus", device), "an unknown mode is rejected");
    }
    expect(sc::mode_for("interleave", true) == sc::Mode::interleave && sc::mode_for("1", true) == sc::Mode::interleave, "interleave / 1 + device");
    expect(sc::mode_for("defer", true) == sc::Mode::defer, "defer + device");
    expect(throws("interleave", false) && throws("1", false) && throws("defer", false), "interleave / defer + rows: rejected");
}

// The hello's serve identity (sched::serve_identity_text): different decode identities or DSpark setups never compare
// equal; the same configuration always does.
void identities() {
    const auto ep = ds::prefix_namespace_from_env(false).text, tp = ds::prefix_namespace_from_env(true).text;
    expect(ep != tp, "namespace: EP and expert-TP decode differ");
    expect(sc::serve_identity_text(ep, 0, 2, "controller", "full") != sc::serve_identity_text(tp, 0, 2, "controller", "full"), "identity: EP vs expert TP differ");
    expect(sc::serve_identity_text(ep, 100, 1, "controller", "full") != sc::serve_identity_text(ep, 100, 2, "controller", "full"), "identity: DSpark ranks 1 vs 2 differ");
    expect(sc::serve_identity_text(ep, 0, 2, "controller", "full") != sc::serve_identity_text(ep, 100, 2, "controller", "full"), "identity: DSpark on vs off differ");
    expect(sc::serve_identity_text(ep, 100, 2, "controller", "full") != sc::serve_identity_text(ep, 100, 2, "fixed:5", "full"), "identity: window policies differ");
    expect(sc::serve_identity_text(ep, 0, 1, "controller", "full") == sc::serve_identity_text(ep, 0, 2, "fixed:5", "auto:x"), "identity: without DSpark its options do not matter");
    expect(sc::serve_identity_text(tp, 100, 2, "controller", "full") == sc::serve_identity_text(tp, 100, 2, "controller", "full"), "identity: same config equal");
}

int main() {
    admission();
    modes();
    identities();
    for (const std::uint32_t total : {9U, 16U, 17U, 100U, 128U, 512U, 513U, 2048U, 2053U, 8192U, 8199U})
        for (const std::uint32_t budget : {0U, 16U, 24U, 256U, 4096U}) {
            units(0, 0, total, 4096, 4096, budget, 9);
            if (total > 512U) units(256, 384, total, 256, 4096, budget, 9);
            if (total > 1024U) units(0, 0, total, 1024, 1024, budget, 9);
        }
    if (failures == 0) std::printf("deepseek41 serve sched tests pass\n");
    return failures == 0 ? 0 : 1;
}
