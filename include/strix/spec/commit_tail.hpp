#pragma once

// Asynchronous commit tail of a speculative decode step (shared host protocol, any backend).
// Home: shared (strix::spec host policy; the backend keeps its stream / event calls and binds this state).
//
// A verify step ends with a commit: device work that folds the accepted rows into the target state (recurrent-state
// replay, conv histories) and seeds the drafter (its cache rows, the held hidden row). The host reads none of it.
// Waiting for it before returning keeps the device idle through the host's between-step work (token emission, context
// proposals, the next draft's launch). With the tail asynchronous the commit is enqueued and the call returns; the one
// stream keeps every launch in the same order on the same inputs, so every emitted id and every state bit is unchanged.
//
// Readers of pinned host inputs still need an idle stream before they rewrite those inputs:
//   - a verify window: its descriptor / position inputs are read by earlier windows' copies, so it synchronises first
//     whenever a tail is pending (verify_needs_sync);
//   - a drafter graph replay: its inputs are read only by drafter replays, and every draft synchronises before it
//     returns, so a pending commit tail (which reads no pinned input) does not stop the replay; any other in-flight work
//     the backend cannot vouch for (other_inflight) still does (draft_replay_ok).
// A device fault in the tail surfaces at the next synchronisation (the draft's read or the verify prologue) and fails
// that call, as any asynchronous device fault does.
namespace strix::spec {

struct CommitTail {
    bool enabled = false;  // the backend's switch (off: the commit synchronises, the old behaviour)
    bool pending = false;  // a commit was enqueued and the stream has not been synchronised since

    // Call after enqueueing a commit. True: the caller synchronises the stream now (switch off).
    [[nodiscard]] bool after_commit() {
        if (!enabled) return true;
        pending = true;
        return false;
    }
    // Any synchronisation of the stream retires the tail.
    void synced() { pending = false; }
    // Verify prologue: synchronise before the window's pinned inputs are rewritten.
    [[nodiscard]] bool verify_needs_sync() const { return pending; }
    // Drafter graph replay: stream_idle = the backend's stream query; other_inflight = enqueued work that is not the
    // commit tail (e.g. deferred drafter rows).
    [[nodiscard]] bool draft_replay_ok(bool stream_idle, bool other_inflight) const {
        return stream_idle || (pending && !other_inflight);
    }
};

}  // namespace strix::spec
