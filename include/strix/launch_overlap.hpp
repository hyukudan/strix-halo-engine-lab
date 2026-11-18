// Early-launch (launch overlap) rule of a kernel chain, as data plus a checker. Backend-neutral: the CUDA core binds it with
// programmatic dependent launch (launch_pdl / pdl_wait / pdl_trigger, src/cuda_core/common.cuh), a HIP chain with its own
// launch-overlap mechanism; only kernel bodies are per backend.
//
// Model (CUDA PTX griddepcontrol semantics; each fact is exercised by the probe harness tests/cuda_core/pdl_harness.cuh):
//  * An EARLY launch may start once every block of its predecessor released it (or exited). Nothing it does before its
//    wait is ordered against kernels that are still running.
//  * The wait returns when the predecessor COMPLETED and its memory is visible. The release is not a memory release:
//    releasing before the last write never makes a post-wait read stale (probe R1).
//  * Completion is transitive through a kernel only if it waits in at least one block of every launch (or is launched
//    without early launch); a kernel in which no block waits breaks the chain (probe S3).
//  * A kernel that releases its successor only after its own wait (in every releasing block) guarantees that its
//    predecessor completed before the successor starts.
//  * A launch without early launch starts after its predecessor completed: a barrier for everything before it.
// Rule checked here: every pair of accesses to one buffer with at least one write, in kernels i < j, is ordered: the access
// of j happens after kernel i completed (before j's start for its pre-wait accesses, before j's wait for the others).
// A segment is one stream / graph sequence; segment boundaries (host sync, graph launch) are full barriers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::overlap {

enum class Access : std::uint8_t { read, write };
enum class Phase : std::uint8_t { pre_wait, post_wait };

struct Use {
    std::uint32_t buffer = 0;  // plan-local buffer id (weights, state, activations, workspaces)
    Access access = Access::read;
    Phase phase = Phase::post_wait;  // post_wait: after the wait IN THE SAME BLOCK; an access in a block that skips the
                                     // wait is pre_wait
};

struct Launch {
    std::string kernel;
    bool early = false;               // launched with early launch (CUDA: programmatic stream serialization)
    bool waits_every_launch = true;   // >= 1 block of every launch reaches the wait (rule C2)
    bool release_after_wait = false;  // every releasing block releases after its own wait
    std::vector<Use> uses;
};

struct Violation {
    std::size_t consumer = 0, producer = 0;  // launch indices, producer < consumer
    std::uint32_t buffer = 0;
    Phase phase = Phase::post_wait;  // phase of the consumer's access
};

namespace detail {
// done[i] != 0: kernel i is guaranteed complete at the point considered.
inline void mark_complete_chain(const std::vector<Launch>& l, std::size_t m, std::vector<char>& done) {
    // complete(m) => complete(m - 1) iff m is not early or waits in every launch; repeat downwards.
    for (std::size_t k = m + 1; k-- > 0;) {
        done[k] = 1;
        if (k == 0) break;
        if (l[k].early && !l[k].waits_every_launch) break;
    }
}
}  // namespace detail

// Guaranteed-complete sets for each launch: before[j] = before its start, waited[j] = before its post-wait part.
struct Order {
    std::vector<std::vector<char>> before, waited;
};

inline Order order(const std::vector<Launch>& l) {
    const std::size_t n = l.size();
    Order o;
    o.before.assign(n, std::vector<char>(n, 0));
    o.waited.assign(n, std::vector<char>(n, 0));
    for (std::size_t j = 0; j < n; ++j) {
        if (j > 0) o.before[j] = o.before[j - 1];  // start(j) >= start(j - 1)
        if (j > 0 && !l[j].early) detail::mark_complete_chain(l, j - 1, o.before[j]);
        if (j > 1 && l[j].early && l[j - 1].early && l[j - 1].release_after_wait && l[j - 1].waits_every_launch)
            detail::mark_complete_chain(l, j - 2, o.before[j]);
        o.waited[j] = o.before[j];
        if (j > 0) detail::mark_complete_chain(l, j - 1, o.waited[j]);
    }
    return o;
}

inline std::vector<Violation> check(const std::vector<Launch>& l) {
    const Order o = order(l);
    std::vector<Violation> v;
    for (std::size_t j = 0; j < l.size(); ++j)
        for (const Use& uj : l[j].uses) {
            const bool pre = l[j].early && uj.phase == Phase::pre_wait;
            const std::vector<char>& done = pre ? o.before[j] : o.waited[j];
            for (std::size_t i = 0; i < j; ++i) {
                if (done[i]) continue;
                for (const Use& ui : l[i].uses)
                    if (ui.buffer == uj.buffer && (ui.access == Access::write || uj.access == Access::write)) {
                        v.push_back({j, i, uj.buffer, pre ? Phase::pre_wait : Phase::post_wait});
                        break;
                    }
            }
        }
    return v;
}

// Kernels that may still run when kernel j starts: (i, j) with i < j not guaranteed complete before start(j).
// depth = j - i. Used for the overlap table (which pairs can overlap at depth 1, 2, ...) and for resource planning.
struct Pair {
    std::size_t first = 0, second = 0;
};
inline std::vector<Pair> overlapping(const std::vector<Launch>& l, std::size_t max_depth) {
    const Order o = order(l);
    std::vector<Pair> p;
    for (std::size_t j = 0; j < l.size(); ++j)
        for (std::size_t i = j > max_depth ? j - max_depth : 0; i < j; ++i)
            if (!o.before[j][i]) p.push_back({i, j});
    return p;
}

// Placement generated from the table: starting from the requested early flags, the consumer of an unordered PRE-wait
// access is launched without early launch (a normal stream edge restores the order). An unordered POST-wait access (or one
// in a consumer already launched normally) means the completion chain is broken between producer and consumer by an early
// launch that never waits; that launch is demoted instead (demoting the consumer would not order it). Repeated until no
// violation remains (each step demotes one early launch, so it terminates). Returns the demotions; check(l) is empty after.
inline std::size_t plan(std::vector<Launch>& l) {
    std::size_t demoted = 0;
    for (;;) {
        const std::vector<Violation> v = check(l);
        if (v.empty()) return demoted;
        const Violation& x = v.front();
        std::size_t k = x.consumer;
        if (!l[k].early || x.phase == Phase::post_wait) {
            while (--k > x.producer && !(l[k].early && !l[k].waits_every_launch)) {
            }
            if (k == x.producer) return demoted;  // unreachable: a non-early consumer is ordered unless the chain is broken
        }
        l[k].early = false;
        ++demoted;
    }
}

}  // namespace strix::overlap
