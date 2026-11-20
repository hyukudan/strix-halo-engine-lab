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
    require(startup_warmup_lever.default_on, "warm-up is default ON");
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

    // The run wrapper hands the planned prompt to the engine, does nothing without a plan and propagates failure.
    std::size_t seen = 0;
    (void)run_startup_warmup(ctx, 248320U, [&](std::vector<std::uint32_t> prompt) { seen = prompt.size(); });
    require(seen == 4094U, "run gets the planned rows");
    seen = 7U;
    require(run_startup_warmup(plan_startup_warmup(false, 8192U, 40960U), 248320U,
                               [&](std::vector<std::uint32_t>) { seen = 0U; }) == 0.0 && seen == 7U,
            "no run without a plan");
    bool thrown = false;
    try {
        (void)run_startup_warmup(tile, 248320U, [](std::vector<std::uint32_t>) { throw std::runtime_error("x"); });
    } catch (const std::runtime_error&) { thrown = true; }
    require(thrown, "a failed warm-up fails the start");

    const auto line = startup_warmup_line("strix serve", tile, 7123.4);
    require(line == "strix serve warm-up: 8192 rows + 1 token (tile) 7123 ms before listening", "warm-up line");
    require(startup_warmup_line("strix serve", plan_startup_warmup(false, 8192U, 40960U), 0.0) ==
                "strix serve warm-up: none (off)", "off line");
    std::puts("serve startup warm-up tests: PASS");
    return 0;
}
