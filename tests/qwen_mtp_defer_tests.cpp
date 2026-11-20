// Host tests of the deferred prompt MTP rows policy (include/strix/spec/mtp_defer.hpp): a deferred call runs
// before anything rewrites the block buffers it reads or reads the rows it writes, is dropped exactly when its session's
// rows are replaced, freed or restarted, and windows never touch the block's tap rows.
#include "strix/spec/mtp_defer.hpp"
#include "strix/cuda/qwen_mtp_defer.hpp"  // forwarding header: must name the same functions

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace qmd = strix::spec::mtp_defer;
// Same entity: two non-type template arguments naming the same function are the same specialization.
template <auto A, auto B> struct SameEntity { static constexpr bool value = false; };
template <auto A> struct SameEntity<A, A> { static constexpr bool value = true; };
static_assert(SameEntity<&qmd::on_use, &strix::cuda::qwen_mtp_defer::on_use>::value &&
              SameEntity<&qmd::defer_block, &strix::cuda::qwen_mtp_defer::defer_block>::value,
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

    // stashed calls only (reads_block = false): the next block of the same prompt rewrites the block buffers, not the stash,
    // so the record stays pending; any other session's block, a restart, a reader and a replacement act as before
    expect(qmd::on_use(true, true, U::engine_block, false, false) == A::none, "stash only, own continuation: none");
    expect(qmd::on_use(true, true, U::engine_block, true, false) == A::drop, "stash only, own fresh prefill: drop");
    expect(qmd::on_use(true, false, U::engine_block, false, false) == A::settle, "stash only, other session's block: settle");
    expect(qmd::on_use(true, true, U::async_block, false, false) == A::settle, "stash only, own async chunk: settle first");
    expect(qmd::on_use(true, true, U::read_rows, false, false) == A::settle, "stash only, own read: settle");
    expect(qmd::on_use(true, true, U::replace_rows, false, false) == A::drop, "stash only, own replace: drop");
    expect(qmd::on_use(true, true, U::window, false, false) == A::none, "stash only, window: none");
    // stash_block: earlier blocks only, when enabled and when the pairs fit after the calls already stashed
    expect(qmd::stash_block(true, 0, 8192, 32652, 0, 8192, 32651), "first of four blocks");
    expect(!qmd::stash_block(true, 24576, 8076, 32652, 24576, 8075, 32651), "last block: deferred in place, never stashed");
    expect(!qmd::stash_block(false, 0, 8192, 32652, 0, 8192, 32651), "off: never stashed");
    expect(!qmd::stash_block(true, 0, 8192, 32652, 0, 0, 32651), "no pairs: nothing to stash");
    expect(!qmd::stash_block(true, 8192, 8192, 32652, 8192, 8192, 10000), "does not fit after the stashed calls");
    expect(!qmd::stash_block(true, 0, 8192, 32652, 20000, 8192, 10000), "stashed beyond capacity: refuse");
    static_assert(qmd::stash_rows_for(0, 8076, 8192) == 0, "one block: no stash");
    static_assert(qmd::stash_rows_for(0, 32652, 8192) == 32651, "every pair but the last position");
    static_assert(qmd::stash_rows_for(8192, 32652, 8192) == 24459, "from a later block");
    static_assert(qmd::stash_bytes(1, 10240) == 20484, "BF16 wide row + next token");
    // Driver model (prefill_enqueue + mtp_settle): blocks of `block` rows over `position`; a stashed block appends its call,
    // the last block records in place, any other block settles the stash first and runs inline. The calls must run in
    // block order for every stash capacity, and with room for the stash no call runs before the last block (TTFT).
    for (const std::uint64_t capacity : {std::uint64_t{0}, std::uint64_t{5000}, std::uint64_t{8192}, std::uint64_t{20000},
                                         std::uint64_t{32651}}) {
        constexpr std::uint64_t position = 32652;
        constexpr std::uint32_t block = 8192;
        std::uint64_t order[8] = {}, stash[8] = {}, rows_used = 0;
        unsigned ran = 0, stashed = 0, before_last = 0;
        bool in_block = false;
        std::uint64_t in_block_first = 0;
        const auto settle = [&] {
            for (unsigned i = 0; i < stashed; ++i) order[ran++] = stash[i];
            stashed = 0;
            rows_used = 0;
            if (in_block) order[ran++] = in_block_first;
            in_block = false;
        };
        for (std::uint64_t p0 = 0; p0 < position; p0 += block) {
            const auto rows = static_cast<std::uint32_t>(position - p0 < block ? position - p0 : block);
            const std::uint64_t last = p0 + rows < position - 1 ? p0 + rows : position - 1;
            const auto pairs = static_cast<std::uint32_t>(last - p0);
            if (qmd::defer_block(true, p0, rows, position)) {
                in_block = true;
                in_block_first = p0;
            } else if (qmd::stash_block(true, p0, rows, position, rows_used, pairs, capacity)) {
                stash[stashed++] = p0;
                rows_used += pairs;
            } else {
                settle();
                order[ran++] = p0;
                before_last = ran;
            }
        }
        settle();  // the first draft reads the rows
        expect(ran == 4, "every block's call runs once");
        for (unsigned i = 1; i < ran; ++i) expect(order[i - 1] < order[i], "calls in block order");
        if (capacity >= qmd::stash_rows_for(0, position, block)) expect(before_last == 0, "with room, nothing runs before the TTFT");
        if (capacity == 0) expect(before_last == 3, "no stash: the earlier blocks run inline (incumbent)");
    }
    if (failures) return EXIT_FAILURE;
    std::puts("qwen_mtp_defer: all checks passed");
    return EXIT_SUCCESS;
}
