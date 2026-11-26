// Host tests of strix/qwen35/prefill_windows.hpp (prompt windows of the qwen35 dense CLIs). Invariants:
//   - windows tile [0, P) in order, none wider than the plan width, the width is the RESOLVED one (requested 64 /
//     resolved 16 -> 16-row windows: no window the engine would refuse);
//   - default and explicit-width plans ask logits of the last window only (one LM head per prompt), ceil(P / W) windows;
//   - diagnostics keep their rows: --dump-logits alone = one-row windows, every row's logits; --rows N = N-row windows;
//     --dump-window = every row of the last wide window; --tf-rows N = the last N tokens as <= 16-row windows;
//   - engine options: untouched unless a width is given (engine defaults), 16 rows on the row path.
#include "strix/qwen35/prefill_windows.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}
namespace q = strix::qwen35;
constexpr std::uint32_t kMax = 16U;

bool tiles(const q::PrefillPlan& p, std::size_t prompt, std::uint32_t widest) {
    std::size_t at = 0;
    for (const auto& w : p.windows) {
        if (w.at != at || w.rows == 0U || w.rows > widest) return false;
        at += w.rows;
    }
    return at == prompt;
}
std::size_t logit_windows(const q::PrefillPlan& p) {
    std::size_t n = 0;
    for (const auto& w : p.windows) n += w.tops() ? 1U : 0U;
    return n;
}

void test_default_and_width() {
    const std::size_t prompts[] = {1U, 16U, 17U, 63U, 64U, 65U, 2048U, 5001U};
    for (const std::uint32_t resolved : {64U, 16U, 128U})
        for (const std::size_t P : prompts)
            for (const std::uint32_t requested : {0U, 64U}) {
                q::PrefillFlags f;
                f.width = requested;
                f.exact = true;
                const auto p = q::plan_prefill(f, P, resolved, kMax);
                check(!p.row_path && p.width == resolved && p.requested == requested, "wide path at the resolved width");
                check(tiles(p, P, resolved), "windows tile the prompt, none wider than the resolved width");
                check(p.windows.size() == (P + resolved - 1U) / resolved, "ceil(P / W) windows");
                check(logit_windows(p) == 1U && p.windows.back().want == q::LogitRows::last && p.windows.back().tops() == 1U,
                      "one LM head: the last window, last row");
            }
}

void test_row_paths() {
    for (const std::size_t P : {1U, 17U, 65U, 300U}) {
        q::PrefillFlags dump;
        dump.dump = true;
        const auto d = q::plan_prefill(dump, P, 64U, kMax);
        check(d.row_path && d.width == 1U && tiles(d, P, 1U), "--dump-logits alone: one-row windows");
        bool all = true;
        std::size_t rows = 0;
        for (const auto& w : d.windows) { all = all && w.want == q::LogitRows::all; rows += w.tops(); }
        check(all && rows == P, "--dump-logits: every prompt row's logits");
        for (const std::uint32_t r : {1U, 5U, 16U}) {
            q::PrefillFlags fr;
            fr.rows = r;
            const auto p = q::plan_prefill(fr, P, 64U, kMax);
            check(p.row_path && p.width == r && tiles(p, P, r), "--rows N: N-row windows");
            check(logit_windows(p) == 1U && p.windows.back().want == q::LogitRows::last, "--rows N: logits on the last window only");
            fr.dump = true;
            const auto pd = q::plan_prefill(fr, P, 64U, kMax);
            std::size_t n = 0;
            for (const auto& w : pd.windows) n += w.tops();
            check(pd.row_path && pd.width == r && n == P, "--rows N --dump-logits: every row");
        }
    }
    q::PrefillFlags wd;  // explicit width wins over --rows
    wd.width = 64U;
    wd.rows = 4U;
    check(!q::plan_prefill(wd, 100U, 64U, kMax).row_path, "an explicit width wins over --rows");
}

void test_dump_window_and_tail() {
    q::PrefillFlags f;
    f.width = 64U;
    f.dump = true;
    const auto last_row = q::plan_prefill(f, 130U, 64U, kMax);
    check(logit_windows(last_row) == 1U && last_row.windows.back().tops() == 1U, "wide + dump: last row of the last window");
    f.dump_window = true;
    const auto win = q::plan_prefill(f, 130U, 64U, kMax);
    check(win.windows.back().want == q::LogitRows::all && win.windows.back().tops() == 2U && logit_windows(win) == 1U,
          "wide + dump-window: every row of the last window (130 = 64 + 64 + 2)");
    q::PrefillFlags dw;  // --dump-logits --dump-window without --wide: the wide path at the resolved width, not the row path
    dw.dump = dw.dump_window = true;
    const auto res = q::plan_prefill(dw, 130U, 64U, kMax);
    check(!dw.row_path() && !res.row_path && res.width == 64U && res.windows.back().tops() == 2U && logit_windows(res) == 1U,
          "dump-window alone: the resolved wide windows, every row of the last one");
    dw.dump_window = false;
    check(dw.row_path(), "dump alone keeps the row path");
    for (const bool dump : {false, true}) {
        q::PrefillFlags t;
        t.width = 64U;
        t.dump = dump;
        t.tf_rows = 40U;
        const auto p = q::plan_prefill(t, 200U, 64U, kMax);
        check(tiles(p, 200U, 64U), "tf tail: tiles");
        std::size_t wide_rows = 0, tail_rows = 0, tail_logits = 0;
        bool tail_ok = true;
        for (const auto& w : p.windows) {
            if (w.at < 160U) { wide_rows += w.rows; check(w.want == q::LogitRows::none, "tf: wide windows ask no logits"); }
            else { tail_rows += w.rows; tail_ok = tail_ok && w.rows <= kMax; tail_logits += w.tops(); }
        }
        check(wide_rows == 160U && tail_rows == 40U && tail_ok, "tf: P - N wide rows, N tail rows in <= 16-row windows");
        check(tail_logits == (dump ? 40U : 1U), "tf tail: every row's logits when dumping, else the last row");
    }
    q::PrefillFlags all_tail;  // tf_rows >= P keeps one wide row (the first token)
    all_tail.tf_rows = 1000U;
    const auto at = q::plan_prefill(all_tail, 20U, 64U, kMax);
    check(tiles(at, 20U, 64U) && at.windows.front().rows == 1U, "tf_rows >= P: the first token stays on the wide path");
}

struct FakeOptions {
    std::uint32_t prefill_rows = 64U;
    bool wide_exact = true;
};
void test_options() {
    FakeOptions o;
    q::apply_prefill_flags(q::PrefillFlags{}, kMax, o);
    check(o.prefill_rows == 64U && o.wide_exact, "no width: engine defaults untouched");
    q::PrefillFlags w;
    w.width = 32U;
    q::apply_prefill_flags(w, kMax, o);
    check(o.prefill_rows == 32U && !o.wide_exact, "--wide 32: FAST class");
    w.exact = true;
    q::apply_prefill_flags(w, kMax, o);
    check(o.prefill_rows == 32U && o.wide_exact, "--wide-exact 32: EXACT class");
    FakeOptions r;
    q::PrefillFlags rows;
    rows.rows = 1U;
    q::apply_prefill_flags(rows, kMax, r);
    check(r.prefill_rows == kMax && r.wide_exact, "row path: 16-row scratch, class untouched");
    bool threw = false;
    try { (void)q::plan_prefill(q::PrefillFlags{}, 0U, 64U, kMax); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "empty prompt refused");
    threw = false;
    rows.rows = 17U;
    try { (void)q::plan_prefill(rows, 10U, 64U, kMax); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "--rows above the decode window limit refused");
}
}  // namespace

int main() {
    test_default_and_width();
    test_row_paths();
    test_dump_window_and_tail();
    test_options();
    if (failures) {
        std::printf("qwen35_prefill_windows_tests: %d failures\n", failures);
        return 1;
    }
    std::printf("qwen35_prefill_windows_tests: PASS\n");
    return 0;
}
