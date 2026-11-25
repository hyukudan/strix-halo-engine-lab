#pragma once

#include "strix/dist/transport.hpp"
#include "strix/serve/cohort_decode.hpp"

#include <cstdint>
#include <span>
#include <vector>

// Two-rank serving over the shared multi-request decoder: rank 0 owns the scheduler, rank 1 replays its steps.
//
// A model split over two nodes runs every forward on both ranks at once (each holds part of the layers; the forward
// exchanges activations by itself). The serving question is only how rank 1 learns which rows to run. The rank-0
// scheduler (CohortDecoder, or anything driving a CohortBackend) talks to a ReplicatedCohort, which is itself a
// CohortBackend: before each local call that runs a forward it sends the call (lane / groups, tokens) to rank 1, where
// follow_cohort() makes the same call on rank 1's backend. Both ranks therefore make the same backend calls with the
// same arguments in the same order, and the forward's own exchange keeps them in step.
//
// bind_lane (lane, 64-bit request id, traits) and control (engine word list) are replicated the same way as commit and
// release: rank 1's backend sees them in rank 0's order, between the same forward calls.
//
// Commit and release run no forward on most engines (rank-local state: frontiers, slot tables), so by default they are
// applied locally at once and ride on the next message to rank 1, which applies them before that message's call: one
// control message per step. An engine whose commit or release talks to the other rank sets
// ReplicaOptions::local_ops = false and each of those calls is sent as its own message.
//
// Control messages: a fixed block of kReplicaBlockWords 32-bit words, exchanged (both ranks send one) with the shared
// lockstep header (strix/pair/lockstep.hpp: round, acknowledged round, rank), so a dropped, duplicated or stale message
// or two ranks started with the same id fail with pair::ProtocolError before rank 1 touches its backend. Payload words
// beyond the block's inline area follow from rank 0 in frames of at most ReplicaOptions::frame_words words. Words are
// host order: both ranks are the same architecture.
//
// Both ranks require a link whose abort wakes the peer and check that capability before sending or receiving a frame.
//
// Pure host code: no device types. Engines bind CohortBackend on each rank; the replication lives here.
namespace strix::serve {

inline constexpr std::uint32_t kReplicaBlockWords = 64;  // 256-byte control block

struct ReplicaOptions {
    bool local_ops = true;                  // commit / release are rank-local: they ride on the next message
    std::uint32_t frame_words = 16384;      // payload words per follow-up frame (the transport's message cap)
    // Collective reads: read_final_logits, sample_final and propose are calls (rank 1 makes the same call on its backend), for an
    // engine whose final logits are gathered from both ranks' vocabulary shares or whose drafter is a two-rank
    // forward. false: both are rank 0 queries (rank 1's backend never sees them).
    bool collective_reads = false;
};

// Rank 0 side. Forwards every query to `local`; mirrors every state change to rank 1.
class ReplicatedCohort final : public CohortBackend {
public:
    ReplicatedCohort(CohortBackend& local, dist::Transport& link, ReplicaOptions options = {});
    ~ReplicatedCohort() override;
    ReplicatedCohort(const ReplicatedCohort&) = delete;
    ReplicatedCohort& operator=(const ReplicatedCohort&) = delete;

    [[nodiscard]] std::uint32_t lanes() const override;
    [[nodiscard]] std::uint32_t rows_per_forward() const override;
    [[nodiscard]] std::uint32_t max_positions() const override;
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override;
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override;
    [[nodiscard]] bool supports_final_logits() const override;
    // rank 0 decides; with collective_reads rank 1 makes the same read (its share of the gather) and discards it
    std::span<const float> read_final_logits(std::uint32_t lane) override;
    // Device sampling is a replicated trait (rank 0 query: both ranks bind the same engine). sample_final is treated
    // like read_final_logits: with collective_reads it is a call (both ranks enter the device exchange), sent with the
    // lane, the position and the sampler's policy and options, from which rank 1 builds an equal sampler (same uniforms
    // per position) for its own sample_final; rank 0's id rules and rank 1's sampler stats are discarded. Without
    // collective_reads it is a rank 0 query.
    [[nodiscard]] bool samples_on_device() const override;
    std::uint32_t sample_final(std::uint32_t lane, TokenSampler& sampler, std::uint64_t position) override;
    void commit(std::span<const std::uint32_t> accept) override;
    void release(std::uint32_t lane) override;
    // Optional hooks. Device drafts and the group capacity / route class are rank 0 queries (drafts reach rank 1 as
    // verify rows; a pair engine reports the capacity both ranks agreed, serve::agree_prefill_group); with
    // collective_reads propose is a call instead. prefill_emits_first, prefill_top and max_window are rank 0 queries
    // (the prefill's greedy id is the forward's own result on both ranks, as verify's top is). A restore that
    // found rows and a capture ride on the next message like commit / release; rank 1 must restore the same rows (its
    // retained store is a replica: the same captures in the same order), else ProtocolError. A group pass is a call.
    [[nodiscard]] bool device_drafts() const override;
    void propose(std::uint32_t lane, std::uint32_t room, std::vector<std::uint32_t>& tokens,
                 std::vector<double>& probability) override;
    [[nodiscard]] std::uint64_t restore(std::uint32_t lane, std::span<const std::uint32_t> prompt) override;
    void capture(std::uint32_t lane, std::span<const std::uint32_t> committed) override;
    [[nodiscard]] PrefillGroupCapacity prompt_group_capacity() const override;
    [[nodiscard]] std::uint32_t prompt_route_class(std::uint64_t rows) const override;
    void prefill_group(std::span<const CohortPrompt> members) override;
    [[nodiscard]] bool prefill_emits_first() const override;
    [[nodiscard]] std::uint32_t prefill_top(std::uint32_t lane) override;
    [[nodiscard]] std::uint32_t max_window(std::uint32_t lane) const override;
    // Per-request binding and engine control messages are rank-local in effect: like commit / release they ride on the
    // next message (local_ops) or go as their own message, and rank 1's backend receives the same call with the same
    // words (a 64-bit request id travels as two words) at the same point in the call sequence.
    void bind_lane(std::uint32_t lane, std::uint64_t request_id, std::span<const std::uint32_t> traits) override;
    void control(std::span<const std::uint32_t> words) override;

    // Delivers the queued commits / releases with no forward (an idle heartbeat for transports with a timeout).
    void keepalive();
    // Delivers the queued operations and ends follow_cohort() on rank 1. Further calls throw.
    void close();
    [[nodiscard]] std::uint64_t rounds() const noexcept { return round_; }  // control messages sent

private:
    void abort_replica(const char* reason) noexcept;
    void send(std::uint32_t op, std::uint32_t arg, std::uint32_t flag, std::span<const std::uint32_t> body);
    void defer(std::uint32_t kind, std::span<const std::uint32_t> head, std::span<const std::uint32_t> body);

    CohortBackend& local_;
    dist::Transport& link_;
    ReplicaOptions options_;
    std::vector<std::uint32_t> deferred_;  // encoded commit / release records not yet on the wire
    std::uint32_t deferred_records_ = 0;
    std::vector<std::uint32_t> body_;
    std::vector<CohortGroup> read_groups_;  // verify groups as both ranks run them (every group read)
    std::uint64_t round_ = 0;
    bool closed_ = false;
};

// Rank 1 side: applies rank 0's calls to `local` until close(). Returns the control messages handled.
// A protocol error or a backend failure aborts the link (dist::Transport::abort) and is rethrown.
std::uint64_t follow_cohort(CohortBackend& local, dist::Transport& link, ReplicaOptions options = {});

}  // namespace strix::serve
