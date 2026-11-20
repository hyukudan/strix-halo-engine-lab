// Server start warm-up (strix/serve/startup_warmup.hpp): the plan bounds, the synthetic ids and the run wrapper.
// Host-only.
#include "strix/serve/startup_warmup.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
}  // namespace

int main() {
    using namespace strix::serve;
    require(!startup_warmup_lever.default_on, "warm-up preserves the lazy-start default");
    require(startup_warmup_lever.exactness == strix::lever::Exactness::bitwise, "warm-up is bitwise for later requests");

    // The tile bounds the rows; a short context leaves room for the output token.
    const auto tile = plan_startup_warmup(true, 8192U, 40960U);
    require(tile.rows == 8192U && tile.outputs == 1U && tile.reason == "tile", "tile-bounded plan");
    const auto ctx = plan_startup_warmup(true, 8192U, 4096U);
    require(ctx.rows == 4094U && ctx.reason == "context", "context-bounded plan keeps prompt + output < context");
    require(plan_startup_warmup(true, 8192U, 8194U).rows == 8192U, "exact fit: 8192 + 1 output < 8194");
    require(plan_startup_warmup(false, 8192U, 40960U).rows == 0U, "off");
    require(plan_startup_warmup(true, 0U, 40960U).rows == 0U, "no tile");
    require(plan_startup_warmup(true, 8192U, 2U).rows == 0U && plan_startup_warmup(true, 8192U, 0U).rows == 0U, "no room");
    static_assert(plan_startup_warmup(true, 16384U, 131072U).rows == 16384U, "constexpr plan");

    // Synthetic ids: deterministic, inside [1, vocabulary / 2), varied.
    const auto a = startup_warmup_prompt(8192U, 248320U);
    const auto b = startup_warmup_prompt(8192U, 248320U);
    require(a == b && a.size() == 8192U, "deterministic ids");
    std::uint32_t lo = UINT32_MAX, hi = 0U, repeats = 0U;
    for (std::size_t i = 0; i < a.size(); ++i) {
        lo = std::min(lo, a[i]);
        hi = std::max(hi, a[i]);
        if (i && a[i] == a[i - 1U]) ++repeats;
    }
    require(lo >= 1U && hi < 248320U / 2U, "ids inside [1, vocabulary / 2)");
    require(hi - lo > 100000U && repeats < 4U, "ids spread over the range");
    for (const auto id : startup_warmup_prompt(64U, 3U)) require(id == 1U, "tiny vocabulary stays valid");

    // Unset and 0 preserve lazy start; an explicit 1 enables warm-up. All other values are refused.
    namespace lever = strix::lever;
    require(!lever::select(startup_warmup_lever, {}, nullptr).on && !lever::select(startup_warmup_lever, {}, "").on,
            "unset switch: warm-up off (existing lazy start)");
    require(!lever::select(startup_warmup_lever, {}, "0").on && lever::select(startup_warmup_lever, {}, "1").on,
            "STRIX_SERVE_WARMUP=0 off, =1 opt-in on");
    bool refused = false;
    try {
        (void)lever::select(startup_warmup_lever, {}, "yes");
    } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "switch values other than 0/1 refused");
    const auto off = plan_startup_warmup(lever::select(startup_warmup_lever, {}, "0").on, 8192U, 40960U);
    const auto unset = plan_startup_warmup(lever::select(startup_warmup_lever, {}, nullptr).on, 8192U, 40960U);
    const auto enabled = plan_startup_warmup(lever::select(startup_warmup_lever, {}, "1").on, 8192U, 40960U);
    require(off.rows == 0U && off.reason == "off" && unset.rows == 0U && unset.reason == "off",
            "disabled/default: no synthetic request");
    require(enabled.rows == 8192U && enabled.outputs == 1U, "explicit opt-in warm-up");

    // The run wrapper hands the planned prompt to the engine and does nothing without a plan.
    std::size_t seen = 0;
    const auto ran = run_startup_warmup(ctx, 248320U, [&](std::vector<std::uint32_t> prompt) { seen = prompt.size(); });
    require(seen == 4094U && ran.failure.empty(), "run gets the planned rows");
    seen = 7U;
    const auto none = run_startup_warmup(off, 248320U, [&](std::vector<std::uint32_t>) { seen = 0U; });
    require(none.ms == 0.0 && none.failure.empty() && seen == 7U, "disabled: no run");

    // Failure paths: a failed warm-up REQUEST (StartupWarmupFailed) is reported and the start goes on lazily (bounded
    // fail-open); any other exception (an engine fault) still fails the start.
    bool thrown = false;
    StartupWarmupResult failed;
    try {
        failed = run_startup_warmup(tile, 248320U,
                                    [](std::vector<std::uint32_t>) { throw StartupWarmupFailed("out of memory"); });
    } catch (...) { thrown = true; }
    require(!thrown && failed.failure == "out of memory", "a failed warm-up request does not fail the start");
    thrown = false;
    try {
        (void)run_startup_warmup(tile, 248320U, [](std::vector<std::uint32_t>) { throw std::runtime_error("fault"); });
    } catch (const StartupWarmupFailed&) {
    } catch (const std::runtime_error&) { thrown = true; }
    require(thrown, "an engine fault in the warm-up fails the start");

    const auto line = startup_warmup_line("strix serve", tile, 7123.4);
    require(line == "strix serve warm-up: 8192 rows + 1 token (tile) 7123 ms before listening", "warm-up line");
    require(startup_warmup_line("strix serve", tile, StartupWarmupResult{7123.4, {}}) == line, "completed result line");
    require(startup_warmup_line("strix serve", tile, StartupWarmupResult{12.0, "out of memory"}) ==
                "strix serve warm-up: 8192 rows + 1 token (tile) FAILED after 12 ms (out of memory): lazy start",
            "failed warm-up line");
    require(startup_warmup_line("strix serve", plan_startup_warmup(false, 8192U, 40960U), 0.0) ==
                "strix serve warm-up: none (off)", "off line");
    std::puts("serve startup warm-up tests: PASS");
    return 0;
}
