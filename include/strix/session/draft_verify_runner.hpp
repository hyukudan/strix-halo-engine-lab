#pragma once

#include "strix/exec/draft_runner.hpp"
#include "strix/session/ring_plan.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::session {

class VerifyWindowPlan {
public:
    static constexpr std::uint32_t max_width = 8;

    explicit VerifyWindowPlan(const RingPlan& ring, exec::ProgramKey target_key = {});

    [[nodiscard]] std::span<const std::uint32_t> widths() const noexcept { return widths_; }
    // Finite warm set for exec::Compiler::warm(). Bucket 0 is reserved for
    // fixed-grid programs, so these keys use 1 + spec::context_bin(frontier).
    [[nodiscard]] std::span<const exec::ProgramKey> keys() const noexcept { return keys_; }
    [[nodiscard]] const exec::ProgramKey& key(std::uint32_t width, std::uint64_t frontier) const;
    [[nodiscard]] std::uint32_t row_limit(std::uint64_t frontier, std::uint32_t max_rows) const noexcept;

    [[nodiscard]] exec::StepBlock describe(std::uint32_t width, std::uint64_t frontier, std::uint32_t slot,
                                           std::uint32_t anchor, std::span<const std::uint32_t> drafts) const;

private:
    std::vector<std::uint32_t> widths_;
    std::vector<exec::ProgramKey> keys_;
};

struct DraftVerifyStep {
    spec::Decision decision{.rows = 0};
    std::array<exec::RowOut, VerifyWindowPlan::max_width> picks{};
    std::uint32_t kept = 0;
    bool stopped = false;

    [[nodiscard]] std::span<const exec::RowOut> committed_picks() const noexcept {
        return std::span(picks).first(kept);
    }
};

namespace detail {

[[nodiscard]] spec::Arbiter chance_arbiter(const VerifyWindowPlan& plan, spec::ChanceWindowTables tables);
void draft_confident_block(exec::DraftRunner& draft, std::uint32_t anchor, std::uint64_t frontier,
                           std::uint32_t k, std::vector<std::uint32_t>& ids, std::vector<float>& confidence);
void accept_verify(DraftVerifyStep& result, std::span<const std::uint32_t> drafts,
                   std::span<std::uint32_t> emitted);

}  // namespace detail

// CPU reference for one serialized greedy lane. The ring may contain a
// committed prompt. The anchor is the next input at ring.committed().
template <class WindowRow, class ConvolutionRow>
class DraftVerifyRunner {
public:
    using Ring = PositionRingReference<WindowRow, ConvolutionRow>;
    // Read committed state from the ring and earlier rows from the pending
    // outputs. Write one pick per physical row and one state image per valid
    // row. A stop pick ends publication at that row. Padding must be masked.
    using Verify = std::function<void(const exec::Program&, const exec::StepBlock&, const Ring&,
                                      std::span<exec::RowOut>, std::span<WindowRow>, std::span<ConvolutionRow>)>;

    DraftVerifyRunner(Ring& ring, std::uint32_t anchor, std::uint32_t slot, exec::DraftRunner& draft,
                      exec::Compiler& target, spec::ChanceWindowTables tables, Verify verify,
                      exec::ProgramKey target_key = {})
        : ring_(ring), anchor_(anchor), slot_(slot), draft_(draft), target_(target), plan_(ring.plan(), target_key),
          arbiter_(detail::chance_arbiter(plan_, std::move(tables))), verify_(std::move(verify)) {
        if (!verify_ || draft_.declaration().kind != spec::DraftKind::block ||
            draft_.declaration().confidence.empty() || anchor_ == spec::kNoToken)
            throw std::invalid_argument("draft verify requires a target, an anchor and a confident P1 block");
    }

    DraftVerifyRunner(const DraftVerifyRunner&) = delete;
    DraftVerifyRunner& operator=(const DraftVerifyRunner&) = delete;

    [[nodiscard]] const VerifyWindowPlan& plan() const noexcept { return plan_; }

    [[nodiscard]] DraftVerifyStep step(std::uint32_t remaining, std::uint32_t max_rows = VerifyWindowPlan::max_width) {
        if (stopped_) return {.stopped = true};
        const auto frontier = ring_.committed();
        const auto limit = plan_.row_limit(frontier, max_rows);
        if (remaining == 0 || limit == 0) return {};
        if (ring_.staged_rows() != 0) throw std::logic_error("draft verify requires an idle target ring");

        draft_.start(anchor_, frontier);
        std::vector<std::uint32_t> proposals;
        std::vector<float> confidence;
        const auto depth = (remaining < limit ? remaining : limit) - 1U;
        detail::draft_confident_block(draft_, anchor_, frontier, depth, proposals, confidence);
        DraftVerifyStep result;
        spec::StepInput input{};
        input.remaining = remaining;
        input.max_rows = limit;
        input.context = frontier;
        input.model_confidence = confidence;
        result.decision = arbiter_.decide(input);
        const auto drafts = std::span(proposals).first(result.decision.k);
        const auto width = result.decision.rows;
        const auto desc = plan_.describe(width, frontier, slot_, anchor_, drafts);
        const auto& program = target_.get(plan_.key(width, frontier));
        const auto valid = drafts.size() + 1;
        std::array<WindowRow, VerifyWindowPlan::max_width> window{};
        std::array<ConvolutionRow, VerifyWindowPlan::max_width> convolution{};
        const auto window_rows = std::span(window).first(valid);
        const auto convolution_rows = std::span(convolution).first(valid);
        auto picks = std::span(result.picks).first(width);
        for (auto& pick : picks) pick = {spec::kNoToken, 0.0F, 0.0F, exec::RowStatus::masked};
        verify_(program, desc, ring_, picks, window_rows, convolution_rows);

        std::array<std::uint32_t, VerifyWindowPlan::max_width> emitted{};
        detail::accept_verify(result, drafts, emitted);
        const auto kept_ids = std::span(emitted).first(result.kept);
        ring_.stage(window_rows, convolution_rows);
        try {
            // P1 may need to catch up its own state. If it fails, the target's
            // old live range remains published and the same anchor can retry.
            draft_.accept(kept_ids);
        } catch (...) {
            ring_.rollback_staged();
            throw;
        }
        ring_.commit_staged(result.kept);
        anchor_ = kept_ids.back();
        stopped_ = result.stopped;
        arbiter_.observe({.source = result.decision.source, .proposed = result.decision.k,
                          .accepted = result.kept - 1U, .rows = width, .context = frontier}, kept_ids);
        return result;
    }

private:
    Ring& ring_;
    std::uint32_t anchor_;
    std::uint32_t slot_;
    exec::DraftRunner& draft_;
    exec::Compiler& target_;
    const VerifyWindowPlan plan_;
    spec::Arbiter arbiter_;
    Verify verify_;
    bool stopped_ = false;
};

}  // namespace strix::session
