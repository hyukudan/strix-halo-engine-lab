// Lone-lane speculation of the shared cohort decoder (strix/serve/lane_speculation.hpp).
#include "strix/serve/lane_speculation.hpp"

#include "strix/spec/block_setup.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix::serve {

void CohortLaneTarget::verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) {
    if (tokens.empty()) throw std::logic_error("cohort lane target: a verify needs the anchor row");
    if (rows_ != 0U) throw std::logic_error("cohort lane target: verify over an uncommitted window");
    const CohortGroup group{lane_, static_cast<std::uint32_t>(tokens.size()), false};
    greedy.assign(tokens.size(), 0U);
    backend_.verify(std::span<const CohortGroup>(&group, 1U), tokens, greedy);
    rows_ = group.rows;
}

void CohortLaneTarget::commit(std::uint32_t k) {
    if (rows_ == 0U || k < 1U || k > rows_) throw std::logic_error("cohort lane target: commit outside the verified window");
    const std::uint32_t keep[1] = {k};
    backend_.commit(keep);
    rows_ = 0;
}

void LaneSpeculator::Chain::attach(spec::ChainDrafter* target) {
    engine_ = target;
    if (engine_ && rule_) engine_->set_row_rule(*rule_);
}

void LaneSpeculator::Chain::set_row_rule(const spec::RowRule& rule) {
    rule_ = rule;
    if (engine_) engine_->set_row_rule(rule);
}

void LaneSpeculator::Chain::set_first_position(std::uint64_t position) {
    if (engine_) engine_->set_first_position(position);
}

void LaneSpeculator::Chain::draft(std::span<const std::uint32_t> prefix, std::uint32_t k,
                                  std::vector<std::uint32_t>& out) {
    if (!engine_) throw std::logic_error("lane speculator: chain draft without a model drafter");
    engine_->draft(prefix, k, out);
}

bool LaneSpeculator::Chain::draft_prelaunch(std::span<const std::uint32_t> prefix, std::uint32_t k,
                                            std::vector<std::uint32_t>& out, const spec::PrelaunchFn& launch) {
    if (!engine_) throw std::logic_error("lane speculator: chain draft without a model drafter");
    return engine_->draft_prelaunch(prefix, k, out, launch);
}

void LaneSpeculator::Chain::accept(std::span<const std::uint32_t> emitted) {
    if (engine_) engine_->accept(emitted);
}

std::span<const std::uint32_t> LaneSpeculator::Chain::runner_ups() const {
    return engine_ ? engine_->runner_ups() : std::span<const std::uint32_t>();
}

std::span<const float> LaneSpeculator::Chain::runner_up_gaps() const {
    return engine_ ? engine_->runner_up_gaps() : std::span<const float>();
}

void LaneSpeculator::Chain::accept_path(std::span<const std::uint32_t> emitted, std::span<const std::uint32_t> rows) {
    if (engine_) engine_->accept_path(emitted, rows);
}

namespace {
// The cohort's block setup: the engine's tuning and seeds, the configured probe-loss keys, learned prices (the
// command line's auto mode without a persisted cost table), greedy rows.
spec::BlockSetup lane_setup(const CohortBackend& backend, const CohortConfig::LaneSpeculation& config,
                            std::uint32_t max_rows, spec::ChainDrafter* chain) {
    spec::BlockSetup setup;
    setup.max_rows = max_rows;
    setup.engine = config.engine;
    setup.backend = config.backend;
    setup.tune = [&backend](spec::BlockConfig& c) { backend.tune_lane_speculation(c); };
    setup.chain = [chain](spec::BlockConfig&) { return chain; };
    setup.seed = [&backend](spec::BlockDrafter& d) { backend.seed_lane_speculation(d); };
    return setup;
}
}  // namespace

std::uint32_t validate_lane_speculation(const CohortBackend& backend, const CohortConfig::LaneSpeculation& config) {
    if (config.max_live == 0U) throw std::invalid_argument("cohort decoder: lane speculation needs max_live >= 1");
    if (config.max_rows < 2U) throw std::invalid_argument("cohort decoder: lane speculation needs max_rows >= 2");
    const std::uint32_t rows = std::max(std::min(config.max_rows, backend.rows_per_forward()), 1U);
    std::optional<spec::BlockDrafter> probe;
    (void)spec::make_block_drafter(probe, lane_setup(backend, config, rows, nullptr));
    return rows;
}

LaneSpeculator::LaneSpeculator(CohortBackend& backend, std::uint32_t lane, const CohortConfig::LaneSpeculation& config,
                               std::uint32_t max_rows)
    : backend_(backend), lane_(lane), config_(config), max_rows_(max_rows), target_(backend, lane) {}

const spec::BlockDrafter& LaneSpeculator::drafter() const {
    if (!drafter_) throw std::logic_error("lane speculator: not started");
    return *drafter_;
}

void LaneSpeculator::build(bool with_chain) {
    speculator_.reset();
    drafter_.reset();
    (void)spec::make_block_drafter(drafter_, lane_setup(backend_, config_, max_rows_, with_chain ? &chain_ : nullptr));
    speculator_.emplace(*drafter_, target_);
    seen_ = 0;
}

void LaneSpeculator::start(std::uint32_t anchor, std::span<const std::uint32_t> stream) {
    if (stream.empty() || stream.back() != anchor)
        throw std::invalid_argument("lane speculator: the stream must end with the anchor");
    if (target_.open()) throw std::logic_error("lane speculator: start over an uncommitted window");
    const std::uint64_t position = stream.size() - 1U;  // committed positions; the anchor is the next row
    // The engine's model drafter, restarted at (anchor, position) when its inputs for the lane are complete.
    const auto restart = [&]() -> spec::ChainDrafter* {
        if (!backend_.lane_speculation_ready(lane_)) return nullptr;
        return backend_.lane_chain_drafter(lane_, anchor, position);
    };
    if (!drafter_) {
        if (!backend_.lane_speculation_ready(lane_)) lookup_only_ = true;
        spec::ChainDrafter* const engine = lookup_only_ ? nullptr : backend_.lane_chain_drafter(lane_, anchor, position);
        chain_.attach(engine);
        if (engine) ++chain_starts_;
        build(engine != nullptr);
    } else if (chain_.engine() != nullptr) {
        if (spec::ChainDrafter* const engine = restart()) {
            chain_.attach(engine);
            ++chain_starts_;
        } else {
            // The BlockDrafter's chain is fixed at construction: a lookup-only lane gets a drafter without one (the
            // lookup index is rebuilt from the whole stream below).
            lookup_only_ = true;
            chain_.attach(nullptr);
            build(false);
        }
    }
    if (seen_ > stream.size()) throw std::logic_error("lane speculator: the stream is shorter than the one it saw");
    drafter_->commit(stream.subspan(static_cast<std::size_t>(seen_)));
    seen_ = stream.size();
    speculator_->start(anchor);
    active_ = true;
}

spec::StepRecord LaneSpeculator::step(std::vector<std::uint32_t>& out, const spec::StepParams& params) {
    if (!active_ || !speculator_) throw std::logic_error("lane speculator: step while not active");
    const std::size_t before = out.size();
    const auto record = speculator_->step(out, params);
    seen_ += out.size() - before;  // the drafter committed the emitted ids (the last one is the new anchor)
    return record;
}

}  // namespace strix::serve
