#pragma once

#include "strix/serve/cohort_decode.hpp"
#include "strix/spec/block_drafter.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// Lone-lane speculation of the shared cohort decoder (CohortConfig::lane_speculation): one cohort lane runs the same
// speculator step as the command line (spec::BlockSpeculator over a spec::BlockDrafter built by
// spec::make_block_drafter, the shared Arbiter) against the cohort backend. Engine-neutral: the backend's optional
// hooks supply the model drafter, the tuning and the seeds. Pure host code.
namespace strix::serve {

// One lane of a CohortBackend as a spec::VerifyTarget: verify(tokens) is backend.verify of ONE group (the lane, all
// tokens, reads_scores false: a greedy lane has no score consumer) with the greedy ids into `greedy`; commit(k) is
// backend.commit({k}). The window stays open between the two (a one-row window may already be committed inside verify,
// as the CohortBackend contract allows; commit({1}) still follows).
class CohortLaneTarget final : public spec::VerifyTarget {
public:
    CohortLaneTarget(CohortBackend& backend, std::uint32_t lane) : backend_(backend), lane_(lane) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override;
    void commit(std::uint32_t k) override;
    // A verified window waits for its commit (never true between two speculator steps).
    [[nodiscard]] bool open() const noexcept { return rows_ != 0U; }
    [[nodiscard]] std::uint32_t lane() const noexcept { return lane_; }

private:
    CohortBackend& backend_;
    std::uint32_t lane_;
    std::uint32_t rows_ = 0;  // rows of the open window (0 = none)
};

// The speculator of one lane for one request. start() begins (or resumes) speculating, step() runs one verify +
// commit, suspend() hands the lane to the round. The drafter keeps its learned walls and acceptance across a suspend.
class LaneSpeculator {
public:
    // max_rows: the widest window (anchor + drafts) the drafter may buy; validated by spec::make_block_drafter.
    LaneSpeculator(CohortBackend& backend, std::uint32_t lane, const CohortConfig::LaneSpeculation& config,
                   std::uint32_t max_rows);
    LaneSpeculator(const LaneSpeculator&) = delete;
    LaneSpeculator& operator=(const LaneSpeculator&) = delete;

    // stream = the request's prompt then its emitted ids; its last token is `anchor`, the lane's pending token (not
    // committed: the lane holds stream.size() - 1 positions). The lookup index receives the tokens of `stream` it has
    // not seen. The first start attaches the engine's model drafter (CohortBackend::lane_chain_drafter) when
    // lane_speculation_ready(lane); a resume restarts it only when ready, else the lane continues lookup-only for the
    // rest of its request (lookup_only()).
    void start(std::uint32_t anchor, std::span<const std::uint32_t> stream);
    // One speculator step (spec::BlockSpeculator::step): appends the emitted ids to `out`. Requires active().
    spec::StepRecord step(std::vector<std::uint32_t>& out, const spec::StepParams& params);
    // The lane takes the round from the next step boundary (the window of the last step is already committed).
    void suspend() noexcept { active_ = false; }

    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] bool started() const noexcept { return drafter_.has_value(); }
    // The model drafter was dropped (its inputs were not complete at a start): copy proposals only from then on.
    [[nodiscard]] bool lookup_only() const noexcept { return lookup_only_; }
    // A model drafter is attached (false: lookup-only, or the engine has none).
    [[nodiscard]] bool chained() const noexcept { return chain_.engine() != nullptr; }
    // Model drafter attachments (the first start and every restart on a resume).
    [[nodiscard]] std::uint64_t chain_starts() const noexcept { return chain_starts_; }
    [[nodiscard]] const spec::BlockDrafter& drafter() const;
    [[nodiscard]] const CohortLaneTarget& target() const noexcept { return target_; }

private:
    // Forwards every ChainDrafter call to the engine's drafter of the current attachment, so a resume can restart the
    // engine's drafter without rebuilding the BlockDrafter (whose chain pointer is fixed at construction).
    class Chain final : public spec::ChainDrafter {
    public:
        void attach(spec::ChainDrafter* target);
        [[nodiscard]] spec::ChainDrafter* engine() const noexcept { return engine_; }
        void set_row_rule(const spec::RowRule& rule) override;
        void set_first_position(std::uint64_t position) override;
        void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override;
        bool draft_prelaunch(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out,
                             const spec::PrelaunchFn& launch) override;
        void accept(std::span<const std::uint32_t> emitted) override;
        [[nodiscard]] std::span<const std::uint32_t> runner_ups() const override;
        [[nodiscard]] std::span<const float> runner_up_gaps() const override;
        void accept_path(std::span<const std::uint32_t> emitted, std::span<const std::uint32_t> rows) override;

    private:
        spec::ChainDrafter* engine_ = nullptr;
        std::optional<spec::RowRule> rule_;  // the last rule set (applied to a drafter attached later)
    };
    void build(bool with_chain);  // (re)builds the drafter and the speculator; the lookup index starts empty

    CohortBackend& backend_;
    std::uint32_t lane_;
    CohortConfig::LaneSpeculation config_;
    std::uint32_t max_rows_;
    CohortLaneTarget target_;
    Chain chain_;
    std::optional<spec::BlockDrafter> drafter_;
    std::optional<spec::BlockSpeculator> speculator_;
    std::uint64_t seen_ = 0;  // stream tokens the drafter's lookup index holds
    std::uint64_t chain_starts_ = 0;
    bool active_ = false, lookup_only_ = false;
};

// Validates a lane speculation configuration against a backend (the drafter it would build); throws
// std::invalid_argument. Returns the widest window: min(max_rows, rows_per_forward).
std::uint32_t validate_lane_speculation(const CohortBackend& backend, const CohortConfig::LaneSpeculation& config);

}  // namespace strix::serve
