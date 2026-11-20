// Shared in-flight prefix join + short-first prefill turn (host, no GPU).
// Pieces: PrefixDirectory::inflight_matches, choose_prefix_join, prefix_join_block_boundary, PrefixJoinWait, and their
// composition in a serial prefill scheduler with a fake engine: a session state is the list of token ids whose rows it
// holds, a checkpoint is a copy of a producer's state. Invariants: every request ends with exactly its own prompt rows
// (restored or computed: same chunks, same bits), shared rows are computed once, a waiter never blocks the producer,
// any producer end releases its waiters, and a long read is overtaken by at most kMaxQueueOvertakes chunks.
#include "strix/serve/prefix_join.hpp"
#include "strix/serve/short_first.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {

namespace sv = strix::serve;
using Tokens = std::vector<std::uint32_t>;

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

Tokens prompt(std::uint32_t family, std::size_t shared, std::uint32_t salt, std::size_t tail) {
    Tokens t;
    for (std::size_t i = 0; i < shared; ++i) t.push_back(family * 100000U + static_cast<std::uint32_t>(i));
    for (std::size_t i = 0; i < tail; ++i) t.push_back(salt * 1000000U + static_cast<std::uint32_t>(i) + 7U);
    return t;
}

std::uint64_t weak_hash(std::uint64_t chain, std::span<const std::uint32_t>) noexcept { return chain; }

void test_inflight_matches() {
    for (const bool weak : {false, true}) {
        sv::PrefixDirectory d(64, 256, weak ? &weak_hash : nullptr);
        const sv::PrefixKey key{1, 2, 3}, other_key{1, 2, 4};
        const Tokens a = prompt(1, 1000, 1, 50), b = prompt(1, 600, 2, 400), c = prompt(2, 1000, 3, 10);
        const auto ea = d.announce(key, a, 0), eb = d.announce(key, b, 1), ec = d.announce(key, c, 2);
        const auto ek = d.announce(other_key, a, 3);
        const Tokens w = prompt(1, 1000, 9, 20);
        auto m = d.inflight_matches(key, w);
        expect(m.size() == 2 && m[0].entry == ea && m[0].tokens == 1000 && m[1].entry == eb && m[1].tokens == 600 &&
                   m[0].inflight && m[1].inflight,
               "inflight_matches: every same-identity producer with its common length, ascending");
        (void)ec;
        (void)ek;
        const Tokens shorter(a.begin(), a.begin() + 700);
        m = d.inflight_matches(key, shorter);
        expect(m.size() == 2 && m[0].tokens == 699 && m[1].tokens == 600, "usable length leaves one suffix token");
        const Tokens tiny(a.begin(), a.begin() + 256);
        expect(d.inflight_matches(key, tiny).empty(), "below min_tokens after the suffix token: no match");
        d.withdraw(eb);
        m = d.inflight_matches(key, w);
        expect(m.size() == 1 && m[0].entry == ea, "a withdrawn producer no longer matches");
        d.publish(ea, 1000);
        expect(d.inflight_matches(key, w).empty(), "a sealed entry is not an in-flight producer");
    }
}

void test_choose_and_boundaries() {
    sv::PrefixDirectory d(64, 256);
    const Tokens a = prompt(1, 3000, 1, 100), b = prompt(1, 2000, 2, 100), w = prompt(1, 3000, 9, 100);
    const auto ea = d.announce({}, a, 4), eb = d.announce({}, b, 5);
    // Monotone trait: the longest shared prefix wins.
    auto c = sv::choose_prefix_join(d, {}, w, 0, [](const sv::PrefixMatch& m, std::uint32_t) { return m.tokens; });
    expect(c && c.entry == ea && c.lane == 4 && c.boundary == 3000, "largest boundary wins");
    // Non-monotone trait (plan-certified): the family's boundary decides, not the common length.
    c = sv::choose_prefix_join(d, {}, w, 0,
                               [&](const sv::PrefixMatch& m, std::uint32_t) -> std::uint64_t { return m.entry == eb ? 1024U : 512U; });
    expect(c && c.entry == eb && c.boundary == 1024, "family boundary trait chooses the producer");
    c = sv::choose_prefix_join(d, {}, w, 2048, [](const sv::PrefixMatch& m, std::uint32_t) { return m.tokens; });
    expect(c && c.entry == ea, "floor drops boundaries at or below the restored rows");
    c = sv::choose_prefix_join(d, {}, w, 3000, [](const sv::PrefixMatch& m, std::uint32_t) { return m.tokens; });
    expect(!c, "nothing above the floor: no join");
    c = sv::choose_prefix_join(d, {}, w, 0, [](const sv::PrefixMatch&, std::uint32_t) { return std::uint64_t{2048}; });
    expect(c && c.entry == ea, "ties keep the older producer");

    expect(sv::prefix_join_block_boundary(6000, 8000, 8000, 0, 256) == 5888, "deepest block with its token shared");
    expect(sv::prefix_join_block_boundary(6145, 8000, 8000, 0, 256) == 6144, "token at the boundary is shared");
    expect(sv::prefix_join_block_boundary(6144, 8000, 8000, 0, 256) == 5888, "boundary token not shared: one block less");
    expect(sv::prefix_join_block_boundary(7999, 8000, 9000, 0, 256) == 7936, "waiter keeps a suffix row (n-2)");
    expect(sv::prefix_join_block_boundary(5000, 8000, 5002, 0, 256) == 4864, "producer keeps a suffix row (n-2)");
    expect(sv::prefix_join_block_boundary(6000, 8000, 8000, 5888, 256) == 0, "a passed boundary is refused");
    expect(sv::prefix_join_block_boundary(6000, 8000, 8000, 5887, 256) == 5888, "an unpassed boundary is kept");
    expect(sv::prefix_join_block_boundary(200, 8000, 8000, 0, 256) == 0, "no block shared");
}

void test_wait_record() {
    // Asynchronous capture rule (HIP Qwen bridge): available once the position has passed the boundary.
    const sv::PrefixJoinWait wait{42, 9216};
    const auto pending = [&](std::uint64_t g, bool active, bool prefilling, std::uint64_t pos) {
        return wait.pending(g, active, prefilling, pos > wait.boundary);
    };
    expect(pending(42, true, true, 1024) && pending(42, true, true, 9216), "wait keeps an incomplete producer");
    expect(!pending(43, true, true, 0), "another generation in the slot releases the waiter");
    expect(!pending(42, false, true, 1024) && !pending(42, true, false, 1024), "producer end releases the waiter");
    expect(!pending(42, true, true, 10240), "passed boundary releases the waiter");
    expect(!sv::PrefixJoinWait{}.pending(0, true, true, false), "an empty record never waits");
}

// ---- Serial prefill scheduler over a fake engine (the CUDA cohort's composition of the shared pieces) ----
struct Lane {
    Tokens prompt;
    std::uint64_t n = 0, prefilled = 0, started = 0, prefix_entry = 0, join_entry = 0, cached = 0;
    std::uint32_t join_lane = 0, overtakes = 0;
    sv::PrefixJoinWait join;
    std::vector<std::uint64_t> caps;
    Tokens state;  // token ids whose rows the session holds
    bool active = false, prefill = false, cancelled = false;
    std::uint64_t done_at = 0;
};

struct Sim {
    static constexpr std::uint32_t block = 256;
    bool join_on = true, short_first = true;
    std::uint32_t chunk = std::numeric_limits<std::uint32_t>::max();
    sv::PrefixDirectory dir{64, 256};
    std::vector<Lane> lanes;
    std::vector<Tokens> checkpoints;
    std::uint64_t computed = 0, clock = 0, admitted = 0;

    std::uint64_t restore(Lane& l) {
        std::uint64_t best = 0;
        const Tokens* from = nullptr;
        for (const auto& cp : checkpoints) {  // a checkpoint at P holds rows [0,P) and token P
            const std::uint64_t p = cp.size() - 1;
            if (p > l.n - 2 || p <= best) continue;
            if (std::equal(cp.begin(), cp.end(), l.prompt.begin())) { best = p; from = &cp; }
        }
        l.state = from ? Tokens(from->begin(), from->end() - 1) : Tokens{};
        return best;
    }
    void announce(Lane& l, std::uint32_t lane) { l.prefix_entry = dir.announce({}, l.prompt, lane); }
    void end_entries(Lane& l) {
        if (l.prefix_entry) { dir.withdraw(l.prefix_entry); l.prefix_entry = 0; }
        if (l.join_entry) { dir.release(l.join_entry); l.join_entry = 0; }
    }
    void admit(Tokens p) {
        lanes.push_back({});
        Lane& l = lanes.back();
        const auto lane = static_cast<std::uint32_t>(lanes.size() - 1);
        l.prompt = std::move(p);
        l.n = l.prompt.size();
        l.started = admitted++;
        l.active = l.prefill = true;
        l.prefilled = l.cached = restore(l);
        l.caps = {(l.n - 2) / block * block};  // end checkpoint
        if (!join_on) return announce(l, lane);
        const auto c = sv::choose_prefix_join(dir, {}, l.prompt, l.prefilled,
            [&](const sv::PrefixMatch& m, std::uint32_t pl) -> std::uint64_t {
                const Lane& p2 = lanes[pl];
                if (pl == lane || !p2.prefill || p2.prefix_entry != m.entry) return 0;
                return sv::prefix_join_block_boundary(m.tokens, l.n, p2.n, p2.prefilled, block);
            });
        if (c && dir.acquire(c.entry)) {
            auto& caps = lanes[c.lane].caps;
            const auto at = std::lower_bound(caps.begin(), caps.end(), c.boundary);
            if (at == caps.end() || *at != c.boundary) caps.insert(at, c.boundary);
            l.join_entry = c.entry;
            l.join_lane = c.lane;
            l.join = {c.entry, c.boundary};
            return;
        }
        announce(l, lane);
    }
    bool waiting(Lane& l, std::uint32_t lane) {
        if (!l.join_entry) return false;
        const Lane& p = lanes[l.join_lane];
        if (l.join.pending(p.prefix_entry, p.active, p.prefill, p.prefilled >= l.join.boundary)) return true;
        dir.release(l.join_entry);
        l.join_entry = 0;
        l.join = {};
        l.prefilled = l.cached = restore(l);
        l.caps = {(l.n - 2) / block * block};
        announce(l, lane);
        return false;
    }
    // One chunk; false when no prompt is left.
    bool step() {
        std::vector<Lane*> ready;
        for (std::uint32_t i = 0; i < lanes.size(); ++i)
            if (lanes[i].prefill && !waiting(lanes[i], i)) ready.push_back(&lanes[i]);
        std::stable_sort(ready.begin(), ready.end(), [](const Lane* a, const Lane* b) { return a->started < b->started; });
        const auto pick = sv::select_waiting(ready, short_first, [](const Lane* l) { return l->n - 1 - l->prefilled; },
                                             [](Lane* l) -> std::uint32_t& { return l->overtakes; });
        if (pick == ready.size()) {
            for (const auto& l : lanes) expect(!l.prefill, "a waiter never stalls the scheduler");
            return false;
        }
        Lane& l = *ready[pick];
        if (l.cancelled) {
            l.prefill = l.active = false;
            end_entries(l);
            return true;
        }
        std::uint64_t rows = std::min<std::uint64_t>(chunk, l.n - 1 - l.prefilled);
        const auto next = std::upper_bound(l.caps.begin(), l.caps.end(), l.prefilled);
        if (next != l.caps.end()) rows = std::min(rows, *next - l.prefilled);
        for (std::uint64_t i = 0; i < rows; ++i) l.state.push_back(l.prompt[l.prefilled + i]);
        l.prefilled += rows;
        computed += rows;
        clock += rows;
        if (std::binary_search(l.caps.begin(), l.caps.end(), l.prefilled))
            checkpoints.emplace_back(l.prompt.begin(), l.prompt.begin() + static_cast<std::ptrdiff_t>(l.prefilled + 1));
        if (l.prefilled == l.n - 1) {
            l.prefill = false;
            l.done_at = clock;
            end_entries(l);
        }
        return true;
    }
    void run() { while (step()) {} }
    void check_states(const char* what) {
        for (const auto& l : lanes)
            if (!l.cancelled) expect(l.state == Tokens(l.prompt.begin(), l.prompt.end() - 1), what);
        expect(dir.size() == 0, "every directory entry is gone at the end");
    }
};

void test_burst_join() {
    for (const bool on : {true, false}) {
        Sim s;
        s.join_on = on;
        for (std::uint32_t i = 0; i < 8; ++i) s.admit(prompt(1, 6000, i + 1, 2000));
        s.run();
        s.check_states("burst: every request ends with exactly its own prompt rows");
        if (on) {
            expect(s.computed == 8000 - 1 + 7 * (8000 - 1 - 5888), "burst: shared rows computed once (22,776 rows)");
            bool all = true;
            for (std::size_t i = 1; i < 8; ++i) all = all && s.lanes[i].cached == 5888;
            expect(all && s.lanes[0].cached == 0, "burst: each waiter restores the producer's 5,888-row checkpoint");
            expect(s.lanes[7].done_at == s.computed, "burst: the last request ends with the last computed row");
        } else {
            expect(s.computed == 8 * (8000 - 1), "burst off: every prompt prefilled whole (incumbent)");
        }
    }
}

void test_producer_cancel_and_passed() {
    Sim s;
    s.chunk = 1024;
    s.admit(prompt(1, 6000, 1, 2000));
    s.step();  // producer runs 1,024 rows
    s.admit(prompt(1, 6000, 2, 2000));
    expect(s.lanes[1].join_entry != 0 && s.lanes[1].join.boundary == 5888, "join planned beyond the producer's frontier");
    s.lanes[0].cancelled = true;
    s.run();
    expect(s.lanes[1].cached == 0 && s.lanes[1].prefilled == 7999, "cancelled producer: the waiter prefills itself");
    s.check_states("cancel: the waiter's rows are its own");

    Sim t;
    t.chunk = 1024;
    t.admit(prompt(1, 6000, 1, 2000));
    for (int i = 0; i < 6; ++i) t.step();  // producer at 6,144 > 5,888
    t.admit(prompt(1, 6000, 2, 2000));
    expect(t.lanes[1].join_entry == 0 && t.lanes[1].prefix_entry != 0, "passed boundary: no wait, the prompt serves joins");
    t.run();
    t.check_states("passed: rows are their own");
}

// Regression contract for the serving default: new policies must remain opt-in.
// Combining both opt-outs must behave exactly like the pre-PR oldest-first, non-joining scheduler.
void test_legacy_prefill_policy() {
    Sim s;
    s.join_on = false;
    s.short_first = false;
    s.chunk = 1024;
    s.admit(prompt(1, 6000, 1, 2000));
    s.step();
    s.admit(prompt(1, 6000, 2, 2000));
    expect(s.lanes[1].join_entry == 0 && s.lanes[1].cached == 0,
           "legacy: a second request never joins an in-flight prefix");
    s.run();
    expect(s.computed == 2 * (8000 - 1),
           "legacy: both requests compute the full prompt without in-flight reuse");
    expect(s.lanes[0].done_at == 8000 - 1 && s.lanes[1].done_at == 2 * (8000 - 1),
           "legacy: oldest request finishes its prompt before the next request");
    s.check_states("legacy: both opt-outs preserve each request's exact rows");
}

void test_short_first() {
    for (const bool sf : {true, false}) {
        Sim s;
        s.join_on = false;
        s.short_first = sf;
        s.chunk = 1024;
        s.admit(prompt(1, 0, 1, 32768));
        s.step();
        s.admit(prompt(2, 0, 2, 1024));
        s.run();
        s.check_states("short first: each request keeps its own chunks");
        if (sf)
            // the short prompt's two chunks (to its end checkpoint at 768, then to 1,023) overtake the long read
            expect(s.lanes[1].done_at == 1024 + 1023 && s.lanes[0].done_at == 32767 + 1023 && s.lanes[0].overtakes == 2,
                   "short first: the short prompt runs at the next chunk boundary, the long pays its rows once");
        else
            expect(s.lanes[1].done_at == 32767 + 1023, "oldest first: the short prompt waits for the whole long read");
    }
    // Starvation bound: a stream of short prompts overtakes a long read at most kMaxQueueOvertakes chunks.
    Sim s;
    s.join_on = false;
    s.chunk = 1024;
    s.admit(prompt(1, 0, 1, 16384));
    for (std::uint32_t i = 0; i < 40; ++i) {
        s.admit(prompt(2 + i, 0, 2 + i, 512));
        s.step();
    }
    s.run();
    expect(s.lanes[0].overtakes == sv::kMaxQueueOvertakes, "long read overtaken at most kMaxQueueOvertakes chunks");
    expect(s.lanes[0].done_at <= 16383 + std::uint64_t{sv::kMaxQueueOvertakes} * 511 + 511,
           "long read ends within its rows plus the bounded overtakes");
    s.check_states("stream: rows are their own");
}

}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    test_inflight_matches();
    test_choose_and_boundaries();
    test_wait_record();
    test_burst_join();
    test_producer_cancel_and_passed();
    test_short_first();
    test_legacy_prefill_policy();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return EXIT_FAILURE;
    }
    std::cout << "prefix join + short-first prefill: all passed\n";
    return EXIT_SUCCESS;
}
