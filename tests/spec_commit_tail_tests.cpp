// strix::spec::CommitTail (include/strix/spec/commit_tail.hpp): asynchronous commit tail of a speculative step.
// Invariants: (1) the enqueue order of every device operation is the same with the tail on and off (only host
// synchronisations differ), so ids and state bits cannot change; (2) a verify window never starts on a busy stream;
// (3) a drafter graph replay never starts while work other than the commit tail is in flight; (4) switch off = the old
// behaviour (every commit synchronises, nothing pending).
#include "strix/spec/commit_tail.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

// A one-stream model: enqueued operations in order, and what is still in flight (no device work completes on its own
// here, which is the worst case for the guards).
struct FakeStream {
    std::vector<std::string> log;  // enqueue order (what decides bits)
    bool commit_inflight = false, other_inflight = false;
    std::uint32_t syncs = 0;
    bool idle() const { return !commit_inflight && !other_inflight; }
    void sync() {
        commit_inflight = other_inflight = false;
        ++syncs;
    }
};

// The backend binding as the CUDA Qwen driver does it (commit_cohort / verify_cohort / draft).
struct FakeEngine {
    FakeStream s;
    strix::spec::CommitTail tail;
    bool deferred_rows = false;  // e.g. deferred drafter cache rows enqueued by the draft's prologue
    std::uint32_t replays = 0, eager = 0, verify_on_busy = 0, replay_on_other = 0;

    void commit(std::uint32_t step) {
        s.log.push_back("commit" + std::to_string(step));
        s.commit_inflight = true;
        if (tail.after_commit()) {
            s.sync();
            tail.synced();
        }
    }
    void draft(std::uint32_t step, bool settle) {
        if (settle) {
            s.log.push_back("settle" + std::to_string(step));
            s.other_inflight = true;
            deferred_rows = true;
        }
        const bool replay = tail.draft_replay_ok(s.idle(), deferred_rows);
        if (replay) {
            ++replays;
            if (s.other_inflight) ++replay_on_other;
        } else {
            ++eager;
        }
        s.log.push_back("draft" + std::to_string(step));
        s.sync();  // the draft reads its tokens back
        tail.synced();
    }
    void verify(std::uint32_t step) {
        if (tail.verify_needs_sync()) {
            s.sync();
            tail.synced();
        }
        if (deferred_rows) {  // the driver's mtp_inflight rule
            deferred_rows = false;
            s.sync();
        }
        if (!s.idle()) ++verify_on_busy;
        s.log.push_back("verify" + std::to_string(step));
        s.sync();  // argmax read-back
    }
};

FakeEngine run(bool enabled, std::uint32_t seed, std::uint32_t steps) {
    FakeEngine e;
    e.tail.enabled = enabled;
    std::mt19937 rng(seed);
    for (std::uint32_t i = 0; i < steps; ++i) {
        const std::uint32_t r = static_cast<std::uint32_t>(rng());
        const bool drafts = (r & 3U) != 0U;  // copy-first / no-room steps verify without a draft
        const bool settle = drafts && (r & 12U) == 12U;
        if (drafts) e.draft(i, settle);
        e.verify(i);
        e.commit(i);
    }
    return e;
}
}  // namespace

int main() {
    using strix::spec::CommitTail;
    {
        CommitTail off;
        check(off.after_commit(), "off: the commit synchronises");
        check(!off.pending && !off.verify_needs_sync(), "off: nothing pending");
        check(off.draft_replay_ok(true, false) && !off.draft_replay_ok(false, false), "off: replay only on an idle stream");
        CommitTail on;
        on.enabled = true;
        check(!on.after_commit() && on.pending, "on: the commit returns with the tail pending");
        check(on.verify_needs_sync(), "on: verify synchronises a pending tail");
        check(on.draft_replay_ok(false, false), "on: the tail alone does not stop a drafter replay");
        check(!on.draft_replay_ok(false, true), "on: other in-flight work stops it");
        on.synced();
        check(!on.pending && !on.draft_replay_ok(false, false), "synced: busy stream with no tail is not vouched for");
    }
    for (std::uint32_t seed = 1; seed <= 64; ++seed) {
        const FakeEngine off = run(false, seed, 200);
        const FakeEngine on = run(true, seed, 200);
        check(off.s.log == on.s.log, "enqueue order identical on and off");
        check(off.verify_on_busy == 0 && on.verify_on_busy == 0, "no verify window on a busy stream");
        check(on.replay_on_other == 0 && off.replay_on_other == 0, "no drafter replay over unvouched work");
        check(on.s.syncs < off.s.syncs, "on: fewer host synchronisations");
        check(on.replays >= off.replays, "on: the drafter keeps its graph replays");
    }
    std::printf("%s spec_commit_tail_tests\n", failures ? "FAIL" : "PASS");
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
