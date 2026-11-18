// Host tests of the deferred prompt MTP rows policy (include/strix/spec/mtp_defer.hpp): a deferred call runs
// before anything rewrites the block buffers it reads or reads the rows it writes, is dropped exactly when its session's
// rows are replaced, freed or restarted, and windows never touch the block's tap rows.
#include "strix/spec/mtp_defer.hpp"
#include "strix/cuda/qwen_mtp_defer.hpp"  // forwarding header: must name the same functions

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace qmd = strix::spec::mtp_defer;
static_assert(&qmd::on_use == &strix::cuda::qwen_mtp_defer::on_use && &qmd::defer_block == &strix::cuda::qwen_mtp_defer::defer_block,
              "the forwarding header aliases the shared policy");

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
constexpr qmd::Use kUses[] = {qmd::Use::engine_block, qmd::Use::async_block, qmd::Use::read_rows, qmd::Use::replace_rows,
                              qmd::Use::window};
}  // namespace

int main() {
    using A = qmd::Action;
    using U = qmd::Use;
    // nothing pending: nothing to do, whatever happens
    for (const auto u : kUses)
        for (const bool same : {false, true})
            for (const bool fresh : {false, true}) expect(qmd::on_use(false, same, u, fresh) == A::none, "no record, no action");
    // an engine-set prefill block rewrites the block buffers: every pending call runs first, except the session's own
    // when its state restarts (its rows would be recomputed by the new prompt)
    expect(qmd::on_use(true, false, U::engine_block, false) == A::settle, "other session's block: settle");
    expect(qmd::on_use(true, false, U::engine_block, true) == A::settle, "other session's fresh block: settle");
    expect(qmd::on_use(true, true, U::engine_block, false) == A::settle, "own continuation: settle");
    expect(qmd::on_use(true, true, U::engine_block, true) == A::drop, "own fresh prefill: drop");
    // an overlapped chunk has its own block buffers: only its own session matters
    expect(qmd::on_use(true, false, U::async_block, true) == A::none, "async other: none");
    expect(qmd::on_use(true, false, U::async_block, false) == A::none, "async other continue: none");
    expect(qmd::on_use(true, true, U::async_block, true) == A::drop, "async own fresh: drop");
    expect(qmd::on_use(true, true, U::async_block, false) == A::settle, "async own continue: settle");
    // readers of the rows (drafts, exports, snapshots) see complete rows
    expect(qmd::on_use(true, true, U::read_rows) == A::settle, "own read: settle");
    expect(qmd::on_use(true, false, U::read_rows) == A::none, "other read: none");
    // replaced / freed rows: never launched into them
    expect(qmd::on_use(true, true, U::replace_rows) == A::drop, "own replace: drop");
    expect(qmd::on_use(true, false, U::replace_rows) == A::none, "other replace: none");
    // a window (the first token) never waits for the deferred call: that is the point
    expect(qmd::on_use(true, true, U::window) == A::none, "own window: none");
    expect(qmd::on_use(true, false, U::window) == A::none, "other window: none");
    // only the prompt's last block is deferred, and only when enabled
    expect(qmd::defer_block(true, 0, 8076, 8076), "single block reaching the position");
    expect(qmd::defer_block(true, 7168, 908, 8076), "last chunk");
    expect(!qmd::defer_block(true, 0, 1024, 8076), "earlier chunk runs in place");
    expect(!qmd::defer_block(false, 0, 8076, 8076), "off: never deferred");
    // tap layout: window rows [0, W) and block rows [W, W + cap) are disjoint for every window up to W rows
    constexpr std::uint32_t kW = 16;
    static_assert(qmd::block_tap_first_row(kW) == kW);
    static_assert(qmd::tap_rows(kW, 8192) == 8208);
    for (std::uint32_t pad = 1; pad <= kW; ++pad) expect(pad <= qmd::block_tap_first_row(kW), "window rows end before block rows");
    if (failures) return EXIT_FAILURE;
    std::puts("qwen_mtp_defer: all checks passed");
    return EXIT_SUCCESS;
}
