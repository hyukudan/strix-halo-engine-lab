#pragma once

#include "strix/serve/cohort_decode.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <memory>
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
// A decoder that also offers prefill_group_first(slots, prompts, firsts), prefill_group_capacity() and
// prefill_route_class(rows) runs waiting whole prompts as ONE grouped work on the emits-first route (the scheduler's
// plan_prefill_group): each member's state and first id equal its prefill_slot_first run, and its final scores are the
// group's row of its position among the members (read_slot_logits(row)).
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
// decoder's one rollback checkpoint is never overwritten. A decoder whose window checkpoint is shared with its eager
// one-row appends declares slot_window_alone() const -> true: a verify mixing its window with one-row lanes is then
// refused before any work.
// A decoder that offers the MIXED pass (stage_slot_rows / commit_slot_rows / max_mixed_rows / max_slot_window) verifies
// the windows of several lanes in ONE forward: stage_slot_rows(lanes, tokens, top) stages lanes[i].rows rows on slot
// lanes[i].slot for every i (tokens in lane order; ragged rows: sum of the rows, at most max_mixed_rows()) and writes
// each row's greedy pick; commit_slot_rows(lanes, tokens, keep) keeps the leading keep[i] rows of each lane (0 discards
// the lane's rows). Each lane holds its own rollback state (session::RowAppendState::stage_batch: a checkpoint per
// lane), so the picks and the kept state of a lane are those of its lone window. A verify with at least one window runs
// every drafted lane and every one-row lane without a score consumer in that one staged pass (a staged one-row lane
// keeps its row); a one-row lane that reads scores runs after it in a decode_slots pass, so its score row stays the
// readable one. A verify of one-row lanes only keeps the decode_slots pass (no staging, the incumbent programs).
struct SlotRows {
    std::uint32_t slot = 0;
    std::uint32_t rows = 1;
};

// A decoder that offers BOTH prefill calls also declares which one it serves with at run time
// (prefill_emits_first() const -> bool, fixed for its lifetime). This is an arithmetic lever whose resolved value
// must match the CLI; it must never be selected by slot, cohort width or arrival order. Read once at construction.
// An engine's lone-lane speculation pieces (CohortConfig::lane_speculation) are bound with bind_lanes(): the binding
// answers CohortBackend's lane hooks as they are asked (the model drafter of a lane, the speculator's data and seeds,
// whether a lane's drafter inputs are complete). Without a binding the hooks keep their defaults (copy drafts only).
// Optional slot hooks (each one detected on the decoder; absent = the CohortBackend default, so a decoder that offers
// none behaves as before). Every hook that may touch device state first resolves a staged window (discard, as a
// prefill does) and withdraws the readable score rows, so the one rollback checkpoint and the score rows stay sound:
//  - release_slot(slot): the lane is free (CohortBackend::release). While a window is staged the call is deferred
//    until the stage is resolved (commit or discard), so a staged slot is never released under its checkpoint.
//  - propose_slot(slot, room, tokens, probability): the device drafter's proposal for the slot's next window
//    (CohortBackend::device_drafts / propose).
//  - restore_slot(slot, prompt) -> rows and capture_slot(slot, committed): the decoder's retained prefixes
//    (CohortBackend::restore / capture). Through this adapter a restore covers at most prompt.size() - 1 rows: the
//    lane's last prompt row is always read by a prefill call (a larger answer is refused as a contract error).
//  - prefill_group_capacity() -> PrefillGroupCapacity, prefill_route_class(rows) -> class and prefill_slots(members):
//    one grouped prompt pass (CohortBackend::prompt_group_capacity / prompt_route_class / prefill_group). Offered only
//    on the plain prefill route: with emits-first the capacity reads one member (no grouping).
//  - samples_on_device() const -> bool and sample_slot_row(row, sampler, position) -> id: device sampling of the
//    row read_slot_logits(row, ...) would return (CohortBackend::samples_on_device / sample_final).
//  - propose_slots(proposals): one drafter pass for the round's lanes (CohortBackend::propose_round), before the
//    round's propose_slot calls; offered with propose_slot only.
//  - samples_in_round() const -> bool and sample_slot_rows(samples): the round's sampled rows in one call
//    (CohortBackend::samples_in_round / sample_round); each entry's lane field names the score row sample_slot_row
//    would read. Offered with sample_slot_row only.
//  - bind_slot(slot, request_id, traits) and control(words): per-request words and engine control messages.
class FixedKvLaneBinding {
public:
    virtual ~FixedKvLaneBinding() = default;
    [[nodiscard]] virtual spec::ChainDrafter* chain_drafter(std::uint32_t lane, std::uint32_t anchor,
                                                            std::uint64_t position) = 0;
    virtual void tune(spec::BlockConfig& /*config*/) const {}
    virtual void seed(spec::BlockDrafter& /*drafter*/) const {}
    [[nodiscard]] virtual bool ready(std::uint32_t lane) const = 0;
};

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

    // Grouped whole prompts (emits-first decoders offering prefill_group_first): the decoder's capacity and route class,
    // else the backend defaults (no grouping, the solo loop).
    [[nodiscard]] PrefillGroupCapacity prompt_group_capacity() const override {
        if constexpr (kGroups) {
            if (emits_first_) {
                auto capacity = decoder_->prefill_group_capacity();
                capacity.members = std::min<std::uint64_t>(capacity.members, lanes_);
                return capacity;
            }
        }
        return CohortBackend::prompt_group_capacity();
    }
    [[nodiscard]] std::uint32_t prompt_route_class(std::uint64_t rows) const override {
        if constexpr (kGroups) {
            if (emits_first_) return decoder_->prefill_route_class(rows);
        }
        return CohortBackend::prompt_route_class(rows);
    }
    void prefill_group(std::span<const CohortPrompt> members) override {
        if constexpr (kGroups) {
            if (emits_first_) {
                discard_staged();
                withdraw();
                group_slots_.clear();
                group_prompts_.clear();
                for (const auto& member : members) {
                    if (!member.first) throw std::logic_error("fixed KV cohort: a group member must start its prompt");
                    if (prefill_policy_) prefill_policy_(*decoder_, member.tokens.size(), true);
                    group_slots_.push_back(member.lane);
                    group_prompts_.push_back(member.tokens);
                }
                group_tops_.assign(members.size(), 0U);
                decoder_->prefill_group_first(group_slots_, group_prompts_, group_tops_);
                group_lanes_ = group_slots_;  // the member order is the decoder's score row order
                return;
            }
        }
        CohortBackend::prefill_group(members);
    }

    [[nodiscard]] std::uint32_t prefill_top(std::uint32_t lane) override {
        if (!emits_first_) return CohortBackend::prefill_top(lane);
        if (const auto row = group_row(lane); row != kNoLane) return group_tops_[row];
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
        if constexpr (kMixed) {
            if (std::ranges::any_of(groups, [](const CohortGroup& g) { return g.rows != 1U; })) {
                verify_mixed(groups, tokens, top);
                return;
            }
        }
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
        if constexpr (kWindowAlone) {
            // A decoder whose staged window shares its rollback checkpoint with the eager one-row appends runs a window
            // only alone; refused before any work (the by-live draft cap keeps windows to one live lane).
            if (!windows_.empty() && !window_.empty() && decoder_->slot_window_alone())
                throw std::logic_error("fixed KV cohort: this decoder runs a verify window alone");
        }
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
        if constexpr (kWindows || kMixed) return std::max<std::uint32_t>(decoder_->max_slot_window(), 1U);
        else return 1U;
    }

    // A mixed decoder runs a rectangle (or any ragged set) of windows in one pass: its row bound.
    [[nodiscard]] std::uint32_t max_batch_verify_rows() const override {
        if constexpr (kMixed) return decoder_->max_mixed_rows();
        else return 0U;
    }

    [[nodiscard]] bool supports_final_logits() const override {
        if constexpr (kReadsLogits) return decoder_->final_logits_supported();
        else return false;
    }

    std::span<const float> read_final_logits(std::uint32_t lane) override {
        if constexpr (kReadsLogits) {
            return read_row(final_row(lane));
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
            if (window && (keep[g] == 0U || keep[g] > windows_.at(groups_[g]).rows))
                throw std::logic_error("fixed KV cohort: a window keeps 1..rows rows");
        }
        if constexpr (kMixed) {
            if (mixed_staged_) {
                mixed_keep_.assign(mixed_lanes_.size(), 0U);
                for (std::size_t g = 0; g < groups_.size(); ++g)
                    if (groups_[g] != kOneRow && windows_[groups_[g]].lane != kNoLane)
                        mixed_keep_[groups_[g]] = keep[g];  // a lane released since the verify discards its rows
                decoder_->commit_slot_rows(std::span<const SlotRows>(mixed_lanes_),
                                           std::span<const std::uint32_t>(window_tokens_),
                                           std::span<const std::uint32_t>(mixed_keep_));
                mixed_staged_ = false;
                staged_ = false;
                withdraw();
                flush_releases();
                return;
            }
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
        flush_releases();
    }

    // Lone-lane speculation: the bound engine pieces (owned here, destroyed before the decoder they may reference).
    void bind_lanes(std::unique_ptr<FixedKvLaneBinding> binding) { lanes_binding_ = std::move(binding); }
    [[nodiscard]] spec::ChainDrafter* lane_chain_drafter(std::uint32_t lane, std::uint32_t anchor,
                                                         std::uint64_t position) override {
        return lanes_binding_ ? lanes_binding_->chain_drafter(lane, anchor, position) : nullptr;
    }
    void tune_lane_speculation(spec::BlockConfig& config) const override {
        if (lanes_binding_) lanes_binding_->tune(config);
    }
    void seed_lane_speculation(spec::BlockDrafter& drafter) const override {
        if (lanes_binding_) lanes_binding_->seed(drafter);
    }
    [[nodiscard]] bool lane_speculation_ready(std::uint32_t lane) const override {
        return lanes_binding_ ? lanes_binding_->ready(lane) : CohortBackend::lane_speculation_ready(lane);
    }

    // The lane is free: its pending results (first id, score row) are withdrawn; a released row of a pending window
    // keeps its place for commit. A subsequent first prefill resets the reused slot's frontier and history.
    void release(std::uint32_t lane) override {
        if (lane == prefill_lane_) prefill_lane_ = kNoLane;
        std::ranges::replace(group_lanes_, lane, kNoLane);
        std::ranges::replace(window_, lane, kNoLane);
        for (auto& w : windows_)
            if (w.lane == lane) w.lane = kNoLane;
        if constexpr (kReleases) {
            if (staged_) pending_releases_.push_back(lane);  // released once the stage is resolved
            else decoder_->release_slot(lane);
        }
    }

    // A decoder with propose_slot may still decline at run time (device_drafts() const -> bool, e.g. no drafter loaded).
    [[nodiscard]] bool device_drafts() const override {
        if constexpr (kProposes && requires(const Decoder& view) { { view.device_drafts() } -> std::convertible_to<bool>; })
            return decoder_->device_drafts();
        else return kProposes;
    }
    void propose(std::uint32_t lane, std::uint32_t room, std::vector<std::uint32_t>& tokens,
                 std::vector<double>& probability) override {
        if constexpr (kProposes) {
            before_work();
            decoder_->propose_slot(lane, room, tokens, probability);
        } else {
            CohortBackend::propose(lane, room, tokens, probability);
        }
    }
    void propose_round(std::span<const CohortProposal> lanes) override {
        if constexpr (kRoundProposes) {
            before_work();
            decoder_->propose_slots(lanes);
        } else {
            CohortBackend::propose_round(lanes);
        }
    }

    [[nodiscard]] std::uint64_t restore(std::uint32_t lane, std::span<const std::uint32_t> prompt) override {
        if constexpr (kRetains) {
            before_work();
            const std::uint64_t rows = decoder_->restore_slot(lane, prompt);
            if (!prompt.empty() && rows >= prompt.size())
                throw std::logic_error("fixed KV cohort: a restore must leave the last prompt row to a prefill call");
            return rows;
        } else {
            return CohortBackend::restore(lane, prompt);
        }
    }
    void capture(std::uint32_t lane, std::span<const std::uint32_t> committed) override {
        if constexpr (kRetains) {
            before_work();
            decoder_->capture_slot(lane, committed);
        } else {
            CohortBackend::capture(lane, committed);
        }
    }

    [[nodiscard]] PrefillGroupCapacity prompt_group_capacity() const override {
        if constexpr (kGroups) {
            if (!emits_first_) return decoder_->prefill_group_capacity();
        }
        return CohortBackend::prompt_group_capacity();
    }
    [[nodiscard]] std::uint32_t prompt_route_class(std::uint64_t rows) const override {
        if constexpr (kGroups) return decoder_->prefill_route_class(rows);
        else return CohortBackend::prompt_route_class(rows);
    }
    void prefill_group(std::span<const CohortPrompt> members) override {
        if constexpr (kGroups) {
            if (emits_first_) throw std::logic_error("fixed KV cohort: no grouped prompt pass on the emits-first route");
            before_work();
            decoder_->prefill_slots(members);
        } else {
            CohortBackend::prefill_group(members);
        }
    }

    [[nodiscard]] bool samples_on_device() const override {
        if constexpr (kSamples) return decoder_->samples_on_device();
        else return false;
    }
    std::uint32_t sample_final(std::uint32_t lane, TokenSampler& sampler, std::uint64_t position) override {
        if constexpr (kSamples) {
            return decoder_->sample_slot_row(final_row(lane), sampler, position);
        } else {
            return CohortBackend::sample_final(lane, sampler, position);
        }
    }
    [[nodiscard]] bool samples_in_round() const override {
        if constexpr (kRoundSamples) return decoder_->samples_in_round();
        else return false;
    }
    void sample_round(std::span<CohortSample> lanes) override {
        if constexpr (kRoundSamples) {
            round_samples_.assign(lanes.begin(), lanes.end());
            for (auto& s : round_samples_) s.lane = final_row(s.lane);
            decoder_->sample_slot_rows(std::span<CohortSample>(round_samples_));
            for (std::size_t i = 0; i < lanes.size(); ++i) lanes[i].token = round_samples_[i].token;
        } else {
            CohortBackend::sample_round(lanes);
        }
    }

    void bind_lane(std::uint32_t lane, std::uint64_t request_id, std::span<const std::uint32_t> traits) override {
        if constexpr (kBinds) {
            before_work();
            decoder_->bind_slot(lane, request_id, traits);
        } else {
            CohortBackend::bind_lane(lane, request_id, traits);
        }
    }
    void control(std::span<const std::uint32_t> words) override {
        if constexpr (kControls) {
            before_work();
            decoder_->control(words);
        } else {
            CohortBackend::control(words);
        }
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
    // Optional: slot_window_alone() const -> bool, true when a window may not share its verify with one-row lanes.
    static constexpr bool kWindowAlone = kWindows && requires(const Decoder& decoder) {
        { decoder.slot_window_alone() } -> std::convertible_to<bool>;
    };
    static constexpr bool kMixed = requires(Decoder& decoder, std::span<const SlotRows> lanes,
                                            std::span<const std::uint32_t> tokens, std::span<std::uint32_t> top,
                                            std::span<const std::uint32_t> keep) {
        decoder.stage_slot_rows(lanes, tokens, top);
        decoder.commit_slot_rows(lanes, tokens, keep);
        { decoder.max_mixed_rows() } -> std::convertible_to<std::uint32_t>;
        { decoder.max_slot_window() } -> std::convertible_to<std::uint32_t>;
    };
    static constexpr bool kPieces = requires(const Decoder& decoder) {
        { decoder.prefill_piece_rule() } -> std::convertible_to<PromptPieceRule>;
        { decoder.prefill_call_chunk_rows(std::uint64_t{}) } -> std::convertible_to<std::uint64_t>;
    };

    static constexpr bool kGroups = requires(Decoder& decoder, const Decoder& view, std::span<const std::uint32_t> slots,
                                             std::span<const std::span<const std::uint32_t>> prompts,
                                             std::span<std::uint32_t> firsts) {
        decoder.prefill_group_first(slots, prompts, firsts);
        { view.prefill_group_capacity() } -> std::convertible_to<PrefillGroupCapacity>;
        { view.prefill_route_class(std::uint64_t{}) } -> std::convertible_to<std::uint32_t>;
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
    static constexpr bool kReleases = requires(Decoder& decoder) { decoder.release_slot(std::uint32_t{}); };
    static constexpr bool kProposes = requires(Decoder& decoder, std::vector<std::uint32_t>& tokens,
                                               std::vector<double>& probability) {
        decoder.propose_slot(std::uint32_t{}, std::uint32_t{}, tokens, probability);
    };
    static constexpr bool kRetains = requires(Decoder& decoder, std::span<const std::uint32_t> tokens) {
        { decoder.restore_slot(std::uint32_t{}, tokens) } -> std::convertible_to<std::uint64_t>;
        decoder.capture_slot(std::uint32_t{}, tokens);
    };
    static constexpr bool kGroups = requires(Decoder& decoder, const Decoder& view,
                                             std::span<const CohortPrompt> members) {
        { view.prefill_group_capacity() } -> std::convertible_to<PrefillGroupCapacity>;
        { view.prefill_route_class(std::uint64_t{}) } -> std::convertible_to<std::uint32_t>;
        decoder.prefill_slots(members);
    };
    static constexpr bool kSamples = requires(Decoder& decoder, const Decoder& view, TokenSampler& sampler) {
        { view.samples_on_device() } -> std::convertible_to<bool>;
        { decoder.sample_slot_row(std::uint32_t{}, sampler, std::uint64_t{}) } -> std::convertible_to<std::uint32_t>;
    };
    static constexpr bool kRoundProposes = kProposes && requires(Decoder& decoder, std::span<const CohortProposal> lanes) {
        decoder.propose_slots(lanes);
    };
    static constexpr bool kRoundSamples = kSamples && requires(Decoder& decoder, const Decoder& view,
                                                               std::span<CohortSample> rows) {
        { view.samples_in_round() } -> std::convertible_to<bool>;
        decoder.sample_slot_rows(rows);
    };
    static constexpr bool kBinds = requires(Decoder& decoder, std::span<const std::uint32_t> traits) {
        decoder.bind_slot(std::uint32_t{}, std::uint64_t{}, traits);
    };
    static constexpr bool kControls = requires(Decoder& decoder, std::span<const std::uint32_t> words) {
        decoder.control(words);
    };
    static constexpr std::uint32_t kNoLane = ~std::uint32_t{0};

    // Before an optional hook that may touch device state: a staged window is resolved (discarded) and the score rows
    // of the last call stop being readable, as before a prefill.
    void before_work() {
        discard_staged();
        withdraw();
    }

    // The score row of `lane` that read_slot_logits / sample_slot_row serve: the emits-first prefill's row 0, else the
    // lane's row of the pending one-row window.
    [[nodiscard]] std::uint32_t final_row(std::uint32_t lane) const {
        if (lane != kNoLane && lane == prefill_lane_) return 0U;  // emits-first: the prefill's row 0
        if (const auto row = group_row(lane); row != kNoLane) return row;  // a group member's row
        if (verify_pending_ && lane != kNoLane)
            for (std::size_t row = 0; row < window_.size(); ++row)
                if (window_[row] == lane) return static_cast<std::uint32_t>(row);
        throw std::logic_error("fixed KV cohort: final logits requested for a lane outside the pending window");
    }

    // Deferred releases (kReleases) reach the decoder once no window is staged.
    void flush_releases() {
        if constexpr (kReleases) {
            if (staged_) return;
            auto lanes = std::move(pending_releases_);
            pending_releases_.clear();
            for (const auto lane : lanes) decoder_->release_slot(lane);
        }
    }

    void decode(std::span<const std::uint32_t> tokens, std::span<std::uint32_t> top, bool reads_scores) {
        if constexpr (kScoreReads) decoder_->decode_slots(window_, tokens, top, reads_scores);
        else decoder_->decode_slots(window_, tokens, top);
    }

    // A window staged by a verify that was never committed: discard it (keep 0) so the decoder restores the slot.
    void discard_staged() {
        discard_stage();
        flush_releases();
    }

    void discard_stage() {
        if constexpr (kMixed) {
            if (mixed_staged_) {
                mixed_keep_.assign(mixed_lanes_.size(), 0U);
                decoder_->commit_slot_rows(std::span<const SlotRows>(mixed_lanes_),
                                           std::span<const std::uint32_t>(window_tokens_),
                                           std::span<const std::uint32_t>(mixed_keep_));
                mixed_staged_ = false;
                staged_ = false;
                return;
            }
        }
        if constexpr (kWindows) {
            if (!staged_) return;
            decoder_->commit_slot_window(staged_slot_, std::span<const std::uint32_t>(window_tokens_), 0U);
            staged_ = false;
        }
    }

    // The mixed pass (kMixed, at least one window): every window and every one-row lane without a score consumer staged
    // in ONE decoder pass; the one-row lanes that read scores run after it (decode_slots), their rows readable.
    void verify_mixed(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                      std::span<std::uint32_t> top) {
        mixed_lanes_.clear();
        window_tokens_.clear();
        std::size_t offset = 0, staged_rows = 0;
        for (const auto& group : groups) {
            if (group.rows == 0U || group.rows > max_window(group.lane))
                throw std::logic_error("fixed KV cohort: a verify window of 1..max_slot_window() rows");
            if (offset + group.rows > tokens.size())
                throw std::invalid_argument("fixed KV cohort: verify tokens do not match the groups");
            if (group.rows == 1U && group.reads_scores) {
                window_.push_back(group.lane);
                window_rows_.push_back(offset);
                groups_.push_back(kOneRow);
            } else {
                if (group.reads_scores)
                    throw std::logic_error("fixed KV cohort: a verify window is greedy-only (no readable scores)");
                groups_.push_back(static_cast<std::uint32_t>(windows_.size()));
                windows_.push_back({group.lane, group.lane, static_cast<std::uint32_t>(offset), group.rows});
                mixed_lanes_.push_back({group.lane, group.rows});
                const auto rows = tokens.subspan(offset, group.rows);
                window_tokens_.insert(window_tokens_.end(), rows.begin(), rows.end());
                staged_rows += group.rows;
            }
            offset += group.rows;
        }
        if (offset != tokens.size() || top.size() != tokens.size())
            throw std::invalid_argument("fixed KV cohort: verify tokens do not match the groups");
        if (staged_rows > decoder_->max_mixed_rows())
            throw std::logic_error("fixed KV cohort: a mixed verify of more than max_mixed_rows() rows");
        try {
            mixed_top_.assign(staged_rows, 0U);
            decoder_->stage_slot_rows(std::span<const SlotRows>(mixed_lanes_),
                                      std::span<const std::uint32_t>(window_tokens_), std::span(mixed_top_));
            mixed_staged_ = true;  // resolved by commit(), else discarded before the next backend work
            staged_ = true;
            std::size_t at = 0;
            for (const auto& w : windows_) {
                std::copy_n(mixed_top_.begin() + static_cast<std::ptrdiff_t>(at), w.rows,
                            top.begin() + static_cast<std::ptrdiff_t>(w.offset));
                at += w.rows;
            }
            if (!window_.empty()) {
                one_tokens_.resize(window_.size());
                one_top_.resize(window_.size());
                for (std::size_t row = 0; row < window_.size(); ++row) one_tokens_[row] = tokens[window_rows_[row]];
                decode(one_tokens_, one_top_, true);
                for (std::size_t row = 0; row < window_.size(); ++row) top[window_rows_[row]] = one_top_[row];
            }
        } catch (...) {
            try {
                discard_staged();
            } catch (...) {
            }
            withdraw();
            throw;
        }
        verify_pending_ = true;
    }

    void withdraw() noexcept {
        window_.clear();
        window_rows_.clear();
        windows_.clear();
        groups_.clear();
        verify_pending_ = false;
        prefill_lane_ = kNoLane;
        group_lanes_.clear();
    }

    // Position of the lane among the last group's members (its score row), or kNoLane.
    [[nodiscard]] std::uint32_t group_row(std::uint32_t lane) const noexcept {
        if (lane == kNoLane) return kNoLane;
        const auto at = std::ranges::find(group_lanes_, lane);
        return at == group_lanes_.end() ? kNoLane : static_cast<std::uint32_t>(at - group_lanes_.begin());
    }

    std::span<const float> read_row(std::uint32_t row) {
        logits_.resize(decoder_->logit_width());
        decoder_->read_slot_logits(row, logits_);
        return logits_;
    }

    std::unique_ptr<Decoder> decoder_;
    std::unique_ptr<FixedKvLaneBinding> lanes_binding_;  // after decoder_: destroyed first
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
    // kMixed: the staged lanes of the last mixed pass (their tokens in window_tokens_), kept for its commit or discard.
    std::vector<SlotRows> mixed_lanes_;
    std::vector<std::uint32_t> mixed_top_, mixed_keep_;
    bool mixed_staged_ = false;
    std::vector<std::uint32_t> pending_releases_;  // kReleases: lanes released while a window was staged
    std::vector<CohortSample> round_samples_;      // kRoundSamples: the round's entries with score rows for lanes
    std::vector<float> logits_;
    // emits-first: the lane of the last prefill call and its first id, valid until the next backend call; kept apart
    // from the verify window, so a commit right after a prefill is refused.
    std::uint32_t prefill_lane_ = kNoLane, prefill_top_ = 0;
    // The last grouped prefill: member lanes in score row order (kNoLane = released since) and their first ids.
    std::vector<std::uint32_t> group_lanes_, group_tops_, group_slots_;
    std::vector<std::span<const std::uint32_t>> group_prompts_;
    bool emits_first_ = kEmitsFirst;  // the served route (kSelectsRoute: the decoder's identity)
};

} // namespace strix::serve
