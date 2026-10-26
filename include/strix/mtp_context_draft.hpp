#pragma once
// Context copy/n-gram proposals for the MTP draft chain (host only, opt-in: --mtp-ngram-draft=ctx|copy).
// The production extension (strix/mtp_ngram_draft.hpp, D-047/D-055) is unchanged and stays the default;
// this drafter replaces its criteria: exact 2..16 token suffix with a majority vote, a cold-start prior,
// proposals after any confident prefix (including none), and copy-first chains that skip predictor
// forwards. Every candidate still passes the exact target verifier.
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace strix {
struct MtpContextDraftConfig {
    std::uint32_t min_suffix = 2, max_suffix = 16;
    std::uint32_t max_rows = 7;
    // An opt-in exploration prior: proposals can reach the verifier on a cold request.
    // The posterior is an estimate, never a target probability or an acceptance decision.
    double prior = 0.75, prior_weight = 1.0;
    double floor = 0.7;
    bool copy_edit = false; // prefer long prompt/output spans even over confident MTP drafts
    // copy/edit: a span replaces MTP drafts (or skips the predictor) only from an anchor of at least this many
    // tokens whose earlier occurrences all continue alike; ambiguous anchors (records, tables) only extend.
    std::uint32_t copy_anchor = 8;
    // Wide copy verify: the minimum unanimous anchor of a wide proposal (0 = copy_anchor; at most max_suffix).
    std::uint32_t wide_anchor = 0;
    // Chain copy (observe-only use, greedy): copy rows inside the ordinary verify window from the longest exact
    // history match of the last 3 tokens; a fresh copy needs chain_fresh matching tokens, a copy that resumes the
    // previous copy's source (an edit just interrupted it) chain_resume. 0 = off.
    std::uint32_t chain_fresh = 0, chain_resume = 5;
    // Copy-first skips the predictor (~9.5 ms of a ~88 ms R8 cycle) and pays every rejected copy row, so it needs a
    // learned chain row acceptance of at least this (t36: rewrite-py-es 203/203 rows -2.4 %, json 621/701 +1.2 %).
    double chain_first_accept = 0.95;
};
struct MtpContextDraftStats {
    std::uint64_t cycles = 0;           // cycles with a primary host-only shadow proposal
    std::uint64_t scored_cycles = 0;    // primary shadows whose prefix reached emitted truth
    std::uint64_t shadow_right_rows = 0; // leading primary shadow rows equal to emitted truth
    std::uint64_t selected_cycles = 0, selected_rows = 0; // passed the proposal gate
    std::uint64_t copy_first_cycles = 0; // long-span proposals selected before any MTP forward
    std::uint64_t replaced_cycles = 0;   // copy/edit spans chosen over a confident MTP prefix
    std::uint64_t copy_first_declined = 0; // gated copy-first spans left to MTP (expected fewer accepted rows)
    std::uint64_t extended_cycles = 0;  // cycles with copy inputs actually delivered to the target
    std::uint64_t extension_rows = 0;   // delivered inputs, excludes padding and serial bonus rows
    std::uint64_t accepted_rows = 0;    // delivered copy inputs accepted by the exact target
    std::uint64_t learning_rows = 0;    // primary plus pending-only shadow observations
    // Wide copy verify (propose_wide): committed (fully accepted) wide cycles and their accepted proposal rows, and
    // abandoned wide proposals (a row was rejected: the target restored its state and nothing was committed).
    std::uint64_t wide_cycles = 0, wide_rows_accepted = 0, wide_abandoned = 0;
    // copy/edit: proposals cut before an edited token, wide proposals refused because their span contains one.
    std::uint64_t edit_cuts = 0, wide_blocked = 0;
    // Chain copy: cycles with copy rows after the kept drafts, of them copy-first (no predictor forward), the copy
    // rows delivered and accepted.
    std::uint64_t chain_cycles = 0, chain_first_cycles = 0, chain_rows = 0, chain_accepted_rows = 0;
    std::uint64_t chain_established = 0; // cycles after which the output tail matched the history by >= chain_fresh
};
class MtpContextDraft {
public:
    explicit MtpContextDraft(MtpContextDraftConfig config = {});
    void reset(std::span<const std::uint32_t> prompt);
    // copy/edit only: a long, gated context anchor can supply the whole chain
    // before GPU drafting. An unsuccessful probe does not count as a proposal.
    // `mtp_expected` (< 0 = no comparison) is the accepted rows an MTP cycle delivers on this request: the span
    // skips the predictor only when its expected accepted rows are higher, otherwise MTP drafts as usual.
    std::vector<std::uint32_t> begin_copy_cycle(std::uint32_t pending, std::size_t room,
        std::vector<double>& acceptance, double min_probability, double mtp_expected = -1.0);
    // room is the total bounded candidate capacity, including any kept MTP prefix.
    // Also probes from pending alone, so a rejected MTP prefix cannot prevent learning.
    // copy/edit replaces the confident MTP prefix with a longer context span only when the span's expected
    // accepted rows exceed `kept_expected` (the prefix's, from its draft probabilities; < 0 = no comparison).
    std::vector<std::uint32_t> begin_cycle(std::uint32_t pending, std::span<const std::uint32_t> drafts,
        std::size_t confident, std::size_t room, std::vector<double>& acceptance,
        double min_probability = 0.0, double kept_expected = -1.0);
    [[nodiscard]] std::size_t kept_drafts() const noexcept { return shadow_.prefix.size(); }
    // copy/edit only: exactly `rows` context tokens after `pending` for one wide verify forward, or empty. Only a long
    // unanimous anchor (suffix >= copy_anchor, every earlier occurrence continuing alike) whose context continuation
    // has at least `rows` tokens, and whose learned acceptance cells (suffix 8+, unanimous; rows 0, 1-2 and 3+) are all
    // at least `min_acceptance`. A committed proposal is finished by finish_cycle like any other (wide counters, same
    // learning); an abandoned one by abandon_wide.
    std::vector<std::uint32_t> propose_wide(std::uint32_t pending, std::size_t rows, double min_acceptance = 0.9);
    // The wide proposal was verified but not committed: `truth` holds the target's selections of its leading rows up
    // to and including the first rejected row. The proposal is scored as a shadow against them (its leading rows
    // right, the rejected row wrong), so the acceptance cells learn the rejection; the context is unchanged (the
    // ordinary cycle that follows commits those tokens). No effect without an open wide proposal.
    void abandon_wide(std::span<const std::uint32_t> truth);
    // Observe-only use: the proposal begin_cycle just built is not delivered; its shadows still learn at
    // finish_cycle, and no delivery counters are kept for it.
    void discard_selection() noexcept { selected_rows_ = 0; }
    // Chain copy (config.chain_fresh > 0). chain_extend: up to `rows` copy rows after `kept` (the drafts that stay
    // in the window), or empty; chain_first: when the previous cycle accepted every copy row it was given, the
    // next `rows` copy rows before any predictor forward (resume threshold), or empty. acceptance_out receives the
    // learned chain row acceptance per returned row. Either is finished by finish_cycle.
    std::vector<std::uint32_t> chain_extend(std::uint32_t pending, std::span<const std::uint32_t> kept, std::size_t rows,
        std::vector<double>& acceptance_out);
    std::vector<std::uint32_t> chain_first(std::uint32_t pending, std::size_t rows, std::vector<double>& acceptance_out);
    [[nodiscard]] bool chain_enabled() const noexcept { return config_.chain_fresh != 0; }
    // Counts come from completed verification, never from proposal construction.
    void finish_cycle(std::span<const std::uint32_t> emitted, std::size_t verified_candidates = 0,
        std::size_t accepted_candidates = 0);
    [[nodiscard]] const MtpContextDraftStats& stats() const noexcept { return stats_; }
    // Learned acceptance per (suffix length 2/3/4-7/8+, unanimous anchor, row 0/1-2/3+).
    [[nodiscard]] double acceptance(std::uint32_t suffix, bool unanimous, std::size_t row) const;
    // Longest exact suffix, then token-wise majority continuation of matching historical spans.
    // Only committed context supplies continuation tokens; hash matches are checked exactly.
    std::vector<std::uint32_t> lookup(std::span<const std::uint32_t> extra, std::size_t length,
        std::uint32_t& suffix, std::uint32_t& support, std::uint32_t* occurrences = nullptr) const;
private:
    struct Key {
        std::uint64_t hash; std::uint32_t n;
        bool operator==(const Key& o) const noexcept { return hash == o.hash && n == o.n; }
    };
    struct KeyHash { std::size_t operator()(const Key& k) const noexcept { return static_cast<std::size_t>(k.hash ^ (k.n * 0x9e3779b97f4a7c15ULL)); } };
    void push(std::uint32_t token);
    [[nodiscard]] static std::size_t cell(std::uint32_t suffix, bool unanimous, std::size_t row) noexcept;
    MtpContextDraftConfig config_;
    std::vector<std::uint32_t> context_;
    std::unordered_map<Key, std::vector<std::uint32_t>, KeyHash> index_;
    std::vector<double> right_, tried_;
    bool started_ = false, wide_ = false;
    // copy/edit: context transitions (preceding token, token) the output replaced after a long unanimous anchor (an
    // edit being applied, e.g. a renamed identifier). Copy proposals stop before them; a wide span containing one is
    // not proposed. Keyed by the pair, not the token alone, so a frequent token is only blocked where it was edited.
    std::unordered_set<std::uint64_t> edited_;
    [[nodiscard]] static std::uint64_t edit_key(std::uint32_t previous, std::uint32_t token) noexcept {
        return (static_cast<std::uint64_t>(previous) << 32U) | token;
    }
    // Index of the first span token whose transition from its predecessor (`previous` before the first) was edited.
    [[nodiscard]] std::size_t first_edit(std::uint32_t previous, std::span<const std::uint32_t> span) const;
    std::size_t selected_rows_ = 0;
    // Chain copy state: the open proposal (rows after chain_kept_ drafts), the history position its first row was
    // copied from, whether the last finished copy was fully accepted, and the source position that copy reached.
    std::size_t chain_kept_ = 0, chain_rows_ = 0, chain_source_ = 0, chain_next_ = 0;
    bool chain_open_ = false, chain_established_ = false;
    double chain_right_ = 0.9, chain_tried_ = 1.0;
    std::vector<std::uint32_t> chain_copy(std::span<const std::uint32_t> extra, std::size_t rows, bool resume_only);
    // Longest exact history match (up to 64) of the last 3 tokens of history ++ extra; returns its source position.
    std::size_t chain_match(std::span<const std::uint32_t> extra, std::size_t& source) const;
    void chain_finish(std::size_t accepted_candidates);
    struct Shadow {
        std::vector<std::uint32_t> prefix, tokens;
        std::uint32_t suffix = 0; bool unanimous = false;
    } shadow_, pending_probe_;
    // Learns from the leading proposal rows against emitted truth (until the first wrong row).
    void score(const Shadow& s, std::span<const std::uint32_t> emitted, bool primary);
    MtpContextDraftStats stats_;
};
} // namespace strix
