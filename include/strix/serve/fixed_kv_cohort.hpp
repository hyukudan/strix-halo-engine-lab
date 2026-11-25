#pragma once

#include "strix/serve/cohort_decode.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::serve {

// Bind a resident decoder with preallocated KV slots to the greedy cohort
// contract. Its decode_slots callback must execute one forward for all rows
// and publish the frontiers together, before returning the selected tokens.
// A decoder that also offers final_logits_supported() / logit_width() /
// read_slot_logits(row, out) serves sampled requests: the final scores of the
// last decode_slots() call are copied out only when a sampled lane asks.
// A decoder that offers prefill_slot_first(lane, tokens, first) -> id reads
// the lane's WHOLE prompt by prefill and decides the first id from the final
// row of each prefill call (CohortBackend::prefill_emits_first), as its own
// single-session prefill does: a served lane then runs the same prefill rows
// as that session. The call's final scores are its row 0 (read_slot_logits),
// copied out only when a sampled lane asks, like a one-lane decode window.
// A decoder that offers per-slot verify windows serves greedy lanes whose group carries drafts (CohortConfig::max_draft):
// stage_slot_window(slot, tokens, top) stages tokens.size() rows on the slot (the pending token, then the drafts) and
// writes each row's greedy pick, commit_slot_window(slot, tokens, keep) keeps the leading `keep` rows (0 discards the
// window) and max_slot_window() bounds the rows of one window. Each window is its own forward; the one-row lanes of the
// same verify run after the window as one decode_slots pass, so their score rows stay the readable ones. Such a
// decoder holds one window at a time: a verify with two windows is refused (the cohort's live-lane rule,
// CohortConfig::max_draft_by_live, keeps drafts to one lane). Greedy ids are the target's picks, so a window commits
// exactly what one-row decode would have committed. Windows are greedy-only: a window writes picks, not readable
// scores, so a drafted group passed with reads_scores is refused before any work (the scheduler drafts only lanes with
// no score consumer). A staged window is always resolved before any other backend work: commit() keeps its prefix, and
// a verify or prefill that comes first (its lane was released and the slot reused, or the commit never came), or a
// companion one-row pass that throws after staging, discards it with commit_slot_window(slot, tokens, 0), so the
// decoder's one rollback checkpoint is never overwritten.
// A decoder that offers BOTH prefill calls also declares which one it serves with at run time
// (prefill_emits_first() const -> bool, fixed for its lifetime). This is an arithmetic lever whose resolved value
// must match the CLI; it must never be selected by slot, cohort width or arrival order. Read once at construction.
template <class Decoder>

class FixedKvCohort final : public CohortBackend {
public:
    using PrefillPolicy = std::function<void(Decoder&, std::uint64_t, bool)>;

    FixedKvCohort(std::unique_ptr<Decoder> decoder, std::uint32_t context,
                  std::uint32_t rows, std::uint32_t lanes, PrefillPolicy policy = {})
        : decoder_(std::move(decoder)), context_(context), rows_(rows), lanes_(lanes), prefill_policy_(std::move(policy)) {
        if (!decoder_ || context_ == 0U || lanes_ == 0U || lanes_ > rows_)
            throw std::invalid_argument("invalid fixed KV cohort geometry");
        if constexpr (kSelectsRoute) emits_first_ = decoder_->prefill_emits_first();
        window_.reserve(lanes_);
        decoder_->allocate_slots(lanes_);
    }

    std::uint32_t lanes() const override { return lanes_; }
    std::uint32_t rows_per_forward() const override { return rows_; }
    std::uint32_t max_positions() const override { return context_; }

    [[nodiscard]] bool prefill_emits_first() const override { return emits_first_; }

    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        prefill_scoped(lane, tokens, first, 0);
    }

    void prefill_scoped(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first,
                        std::uint64_t prompt_tokens) override {
        discard_staged();  // before any backend work: a staged window's rollback state must not be overwritten
        if (prefill_policy_) prefill_policy_(*decoder_, prompt_tokens, first);
        withdraw();  // a prefill call replaces the decoder's score rows: no verify window stays readable
        if constexpr (kEmitsFirst) {
            if (emits_first_) {
                // The window is now this lane's prefill at row 0.
                prefill_top_ = decoder_->prefill_slot_first(lane, tokens, first);
                prefill_lane_ = lane;
                return;
            }
        }
        if constexpr (kPlainPrefill) decoder_->prefill_slot(lane, tokens, first);
    }

    [[nodiscard]] std::uint32_t prefill_top(std::uint32_t lane) override {
        if (!emits_first_) return CohortBackend::prefill_top(lane);
        if (lane == kNoLane || lane != prefill_lane_)
            throw std::logic_error("fixed KV cohort: first id requested for a lane not just prefilled");
        return prefill_top_;
    }

    std::uint32_t prompt_tail_rows() const override {
        if constexpr (requires { decoder_->prefill_tail_rows(); }) return decoder_->prefill_tail_rows();
        else return 0U;
    }

    // Bounded prompt slices: a decoder offering prefill_piece_rule() and prefill_call_chunk_rows(rows) states where its
    // prefill chunks may be cut without changing bits and how it chunks one prefill call (CohortBackend).
    PromptPieceRule prompt_piece_rule() const override {
        if constexpr (kPieces) return decoder_->prefill_piece_rule();
        else return {};
    }
    std::uint64_t prompt_call_chunk(std::uint64_t rows) const override {
        if constexpr (kPieces) return decoder_->prefill_call_chunk_rows(rows);
        else return rows;
    }

    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        discard_staged();
        withdraw();
        // A decoder with the score-read hook may use a pick-only form when no lane needs final scores.
        bool reads_scores = false;
        std::size_t offset = 0;
        for (const auto& group : groups) {
            if (group.rows == 1U) {
                window_.push_back(group.lane);
                window_rows_.push_back(offset);
                groups_.push_back(kOneRow);
                reads_scores = reads_scores || group.reads_scores;
            } else {
                if constexpr (!kWindows) throw std::logic_error("fixed KV cohort requires one decode row per lane");
                if (group.rows == 0U || group.rows > max_window(group.lane) || !windows_.empty())
                    throw std::logic_error("fixed KV cohort: one verify window of at most max_slot_window() rows");
                if (group.reads_scores)
                    throw std::logic_error("fixed KV cohort: a verify window is greedy-only (no readable scores)");
                windows_.push_back({group.lane, group.lane, static_cast<std::uint32_t>(offset), group.rows});
                groups_.push_back(0U);
            }
            offset += group.rows;
        }
        if (offset != tokens.size() || top.size() != tokens.size())
            throw std::invalid_argument("fixed KV cohort: verify tokens do not match the groups");
        try {
            if constexpr (kWindows) {
                for (const auto& w : windows_) {
                    const auto rows = tokens.subspan(w.offset, w.rows);
                    window_tokens_.assign(rows.begin(), rows.end());
                    decoder_->stage_slot_window(w.slot, rows, top.subspan(w.offset, w.rows));
                    staged_ = true;  // resolved by commit(), else discarded before the next backend work
                    staged_slot_ = w.slot;
                }
            }
            if (!window_.empty()) {
                if (windows_.empty()) {
                    decode(tokens, top, reads_scores);
                } else {
                    one_tokens_.resize(window_.size());
                    one_top_.resize(window_.size());
                    for (std::size_t row = 0; row < window_.size(); ++row) one_tokens_[row] = tokens[window_rows_[row]];
                    decode(one_tokens_, one_top_, reads_scores);
                    for (std::size_t row = 0; row < window_.size(); ++row) top[window_rows_[row]] = one_top_[row];
                }
            }
        } catch (...) {
            // This verify will never be committed: unwind its staged window now. The first error is the one reported;
            // a discard that fails too stays staged and is retried before the next backend work.
            try {
                discard_staged();
            } catch (...) {
            }
            withdraw();
            throw;
        }
        verify_pending_ = true;
    }

    // The most rows (pending token plus drafts) of a lane's group: the decoder's window bound, else one row.
    [[nodiscard]] std::uint32_t max_window(std::uint32_t) const override {
        if constexpr (kWindows) return std::max<std::uint32_t>(decoder_->max_slot_window(), 1U);
        else return 1U;
    }

    [[nodiscard]] bool supports_final_logits() const override {
        if constexpr (kReadsLogits) return decoder_->final_logits_supported();
        else return false;
    }

    std::span<const float> read_final_logits(std::uint32_t lane) override {
        if constexpr (kReadsLogits) {
            if (lane != kNoLane && lane == prefill_lane_) return read_row(0U);  // emits-first: the prefill's row 0
            if (verify_pending_ && lane != kNoLane)
                for (std::size_t row = 0; row < window_.size(); ++row)
                    if (window_[row] == lane) return read_row(static_cast<std::uint32_t>(row));
            throw std::logic_error("fixed KV cohort: final logits requested for a lane outside the pending window");
        } else {
            return CohortBackend::read_final_logits(lane);
        }
    }

    void commit(std::span<const std::uint32_t> keep) override {
        if (!verify_pending_) throw std::logic_error("fixed KV cohort: commit without a pending verify window");
        if (keep.size() != groups_.size())
            throw std::logic_error("fixed KV cohort: commit needs one kept count per verify group");
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            const bool window = groups_[g] != kOneRow;
            if (!window && keep[g] != 1U) throw std::logic_error("fixed KV cohort must keep each guaranteed decode row");
            if (window && (keep[g] == 0U || keep[g] > windows_.front().rows))
                throw std::logic_error("fixed KV cohort: a window keeps 1..rows rows");
        }
        if constexpr (kWindows) {
            for (std::size_t g = 0; g < groups_.size(); ++g) {
                if (groups_[g] == kOneRow) continue;
                const auto& w = windows_.front();
                // A lane released since the verify discards its window: its slot is reset by its next first prefill.
                decoder_->commit_slot_window(w.slot, std::span<const std::uint32_t>(window_tokens_),
                                             w.lane == kNoLane ? 0U : keep[g]);
                staged_ = false;
            }
        }
        withdraw();
    }

    // The lane is free: its pending results (first id, score row) are withdrawn; a released row of a pending window
    // keeps its place for commit. A subsequent first prefill resets the reused slot's frontier and history.
    void release(std::uint32_t lane) override {
        if (lane == prefill_lane_) prefill_lane_ = kNoLane;
        std::ranges::replace(window_, lane, kNoLane);
        for (auto& w : windows_)
            if (w.lane == lane) w.lane = kNoLane;
    }

private:
    static constexpr bool kReadsLogits = requires(Decoder& decoder, std::span<float> out) {
        { decoder.final_logits_supported() } -> std::convertible_to<bool>;
        { decoder.logit_width() } -> std::convertible_to<std::uint32_t>;
        decoder.read_slot_logits(std::uint32_t{}, out);
    };
    static constexpr bool kScoreReads = requires(Decoder& decoder, const std::vector<std::uint32_t>& window,
                                                 std::span<const std::uint32_t> tokens, std::span<std::uint32_t> top) {
        decoder.decode_slots(window, tokens, top, bool{});
    };
    static constexpr bool kWindows = requires(Decoder& decoder, std::span<const std::uint32_t> tokens,
                                              std::span<std::uint32_t> top) {
        decoder.stage_slot_window(std::uint32_t{}, tokens, top);
        decoder.commit_slot_window(std::uint32_t{}, tokens, std::uint32_t{});
        { decoder.max_slot_window() } -> std::convertible_to<std::uint32_t>;
    };
    static constexpr bool kPieces = requires(const Decoder& decoder) {
        { decoder.prefill_piece_rule() } -> std::convertible_to<PromptPieceRule>;
        { decoder.prefill_call_chunk_rows(std::uint64_t{}) } -> std::convertible_to<std::uint64_t>;
    };

    static constexpr bool kEmitsFirst = requires(Decoder& decoder, std::span<const std::uint32_t> tokens) {
        { decoder.prefill_slot_first(std::uint32_t{}, tokens, bool{}) } -> std::convertible_to<std::uint32_t>;
    };
    static constexpr bool kPlainPrefill = requires(Decoder& decoder, std::span<const std::uint32_t> tokens) {
        decoder.prefill_slot(std::uint32_t{}, tokens, bool{});
    };
    static_assert(kEmitsFirst || kPlainPrefill, "fixed KV cohort: the decoder offers no prefill call");
    // Both calls offered and declared: the decoder's run-time identity picks one (prefill_emits_first()); undeclared,
    // the emits-first call serves.
    static constexpr bool kSelectsRoute = kEmitsFirst && kPlainPrefill && requires(const Decoder& decoder) {
        { decoder.prefill_emits_first() } -> std::convertible_to<bool>;
    };
    static constexpr std::uint32_t kNoLane = ~std::uint32_t{0};

    void decode(std::span<const std::uint32_t> tokens, std::span<std::uint32_t> top, bool reads_scores) {
        if constexpr (kScoreReads) decoder_->decode_slots(window_, tokens, top, reads_scores);
        else decoder_->decode_slots(window_, tokens, top);
    }

    // A window staged by a verify that was never committed: discard it (keep 0) so the decoder restores the slot.
    void discard_staged() {
        if constexpr (kWindows) {
            if (!staged_) return;
            decoder_->commit_slot_window(staged_slot_, std::span<const std::uint32_t>(window_tokens_), 0U);
            staged_ = false;
        }
    }

    void withdraw() noexcept {
        window_.clear();
        window_rows_.clear();
        windows_.clear();
        groups_.clear();
        verify_pending_ = false;
        prefill_lane_ = kNoLane;
    }

    std::span<const float> read_row(std::uint32_t row) {
        logits_.resize(decoder_->logit_width());
        decoder_->read_slot_logits(row, logits_);
        return logits_;
    }

    std::unique_ptr<Decoder> decoder_;
    std::uint32_t context_, rows_, lanes_;
    PrefillPolicy prefill_policy_;
    // The last verify window (lane per decode row; kNoLane = released since), readable until commit.
    std::vector<std::uint32_t> window_;
    bool verify_pending_ = false;
    // The one-lane verify window (kWindows): its lane (kNoLane once released), slot, first token index and rows.
    struct Window {
        std::uint32_t lane, slot, offset, rows;
    };
    static constexpr std::uint32_t kOneRow = ~std::uint32_t{0};
    std::vector<std::size_t> window_rows_;      // token index of each one-row lane of window_
    std::vector<Window> windows_;               // at most one
    std::vector<std::uint32_t> groups_;         // per verify group: kOneRow, or 0 for the window
    std::vector<std::uint32_t> window_tokens_;  // the window's tokens, kept for its commit
    bool staged_ = false;                       // the decoder holds a staged window on staged_slot_
    std::uint32_t staged_slot_ = 0;
    std::vector<std::uint32_t> one_tokens_, one_top_;
    std::vector<float> logits_;
    // emits-first: the lane of the last prefill call and its first id, valid until the next backend call; kept apart
    // from the verify window, so a commit right after a prefill is refused.
    std::uint32_t prefill_lane_ = kNoLane, prefill_top_ = 0;
    bool emits_first_ = kEmitsFirst;  // the served route (kSelectsRoute: the decoder's identity)
};

} // namespace strix::serve
