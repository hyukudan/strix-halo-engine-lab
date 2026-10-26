#pragma once
// Context n-gram extension of the MTP draft chain (host only).
//
// The MTP chain stops at its first draft below the minimum probability (or where the window cost model
// says more drafts do not pay). When the request is copying text from its context (a document being
// corrected, code being edited) the continuation is often already in the prompt or the output: the
// longest suffix (2..4 tokens) of context + confident drafts that occurred before is followed by the
// tokens that followed it then. Those tokens extend the chain inside the same verify window; the target
// verifies every row, so the output is the greedy output whatever the proposal.
//
// Acceptance of the extension rows is learned per request: every cycle's extension proposal is scored
// against the tokens the verifier emitted, whether it was used or not ("shadow" scoring, host work
// only), starting from a pessimistic prior. Below the floor the extension is not proposed, so a request
// that does not copy (free prose, summaries) never sees an extension row.
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace strix {

struct MtpNgramDraftConfig {
    std::uint32_t min_suffix = 2, max_suffix = 4;  // suffix lengths matched against the context
    std::uint32_t max_rows = 7;                    // extension rows at most
    double prior = 0.2, prior_weight = 4.0;        // per-request acceptance prior (pseudo-counts)
    double floor = 0.5;                            // first extension row must reach this acceptance
    // Cross-request lookup (set_peers): the shortest suffix matched in another request's committed context.
    std::uint32_t peer_min_suffix = 4;
};

struct MtpNgramDraftStats {
    std::uint64_t cycles = 0;          // cycles with a shadow proposal (host only: no verifier row unless extended)
    std::uint64_t scored_cycles = 0;   // shadow proposals in play (their confident drafts were all emitted)
    std::uint64_t shadow_right_rows = 0; // leading shadow rows equal to the emitted tokens (would-be accepted rows)
    std::uint64_t extended_cycles = 0; // cycles whose candidates were extended
    std::uint64_t extension_rows = 0;  // extension rows proposed to the verifier
    std::uint64_t accepted_rows = 0;   // extension rows accepted by the target
    std::uint64_t peer_cycles = 0, peer_extension_rows = 0, peer_accepted_rows = 0; // the subset from set_peers
};

class MtpNgramDraft {
public:
    explicit MtpNgramDraft(MtpNgramDraftConfig config = {});
    // A new request: the prompt, then each cycle's pending token through begin_cycle.
    void reset(std::span<const std::uint32_t> prompt);
    // Cycle start: `pending` is the unconsumed token the cycle starts from (appended to the context the
    // first time), `drafts` the MTP chain and `confident` how many leading drafts are kept. Returns the
    // extension rows (possibly empty) and their learned acceptance in `acceptance`.
    std::vector<std::uint32_t> begin_cycle(std::uint32_t pending, std::span<const std::uint32_t> drafts,
        std::size_t confident, std::size_t room, std::vector<double>& acceptance);
    // Marks the extension of this cycle as used (`rows` of it were given to the verifier).
    void used(std::size_t rows);
    // Cycle end: the tokens the verifier emitted (accepted drafts and the target's next token).
    void finish_cycle(std::span<const std::uint32_t> emitted);
    [[nodiscard]] const MtpNgramDraftStats& stats() const noexcept { return stats_; }
    [[nodiscard]] double acceptance(std::uint32_t suffix, bool supported, std::size_t row) const;
    // Opt-in cross-request lookup (joint speculation of concurrent conversations): when this request's own context
    // has no match, the longest suffix (max_suffix down to peer_min_suffix) of context + confident drafts is looked
    // up in these drafters' committed contexts (most recent occurrence first, peers in order). Peer proposals have
    // their own learned acceptance cells (prior as the others), so a request whose peers do not help never gets
    // an extension from them. Pointers are borrowed: the caller refreshes them (empty = off) before each cycle.
    void set_peers(std::vector<const MtpNgramDraft*> peers) { peers_ = std::move(peers); }
    [[nodiscard]] double peer_acceptance(std::size_t row) const;
    // Longest-suffix lookup over the context followed by `extra` (exposed for tests).
    std::vector<std::uint32_t> lookup(std::span<const std::uint32_t> extra, std::size_t length,
        std::uint32_t& suffix, std::uint32_t& support) const;
private:
    struct Key {
        std::uint64_t hash; std::uint32_t n;
        bool operator==(const Key& o) const noexcept { return hash == o.hash && n == o.n; }
    };
    struct KeyHash { std::size_t operator()(const Key& k) const noexcept { return static_cast<std::size_t>(k.hash ^ (k.n * 0x9e3779b97f4a7c15ULL)); } };
    void push(std::uint32_t token);
    [[nodiscard]] static std::size_t cell(std::uint32_t suffix, bool supported, std::size_t row) noexcept;
    MtpNgramDraftConfig config_;
    std::vector<std::uint32_t> context_;
    // n-gram (by hash of its tokens) -> end positions in context_, ascending
    std::unordered_map<Key, std::vector<std::uint32_t>, KeyHash> index_;
    std::vector<double> right_, tried_;  // online acceptance table
    bool started_ = false;
    std::vector<const MtpNgramDraft*> peers_;
    // Continuation of the longest suffix of (context_ + extra) found in a peer's committed context.
    std::vector<std::uint32_t> peer_lookup(std::span<const std::uint32_t> extra, std::size_t length) const;
    [[nodiscard]] std::size_t shadow_cell(std::size_t row) const noexcept;
    struct Shadow {
        bool valid = false;
        std::vector<std::uint32_t> prefix, tokens;  // confident drafts it continues, extension rows
        std::uint32_t suffix = 0; bool supported = false; bool peer = false; std::size_t used = 0;
    } shadow_;
    MtpNgramDraftStats stats_;
};

}  // namespace strix
