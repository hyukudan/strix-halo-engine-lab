#include "strix/session/draft_verify_runner.hpp"
#include "strix/session/linear_verify_step.hpp"

#include "strix/spec/greedy_prefix.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

namespace strix::session {
namespace {

constexpr auto max_position = std::numeric_limits<std::uint32_t>::max();

}  // namespace

VerifyWindowPlan::VerifyWindowPlan(const RingPlan& ring, exec::ProgramKey target_key) {
    if (target_key.graph_id != 0 || target_key.tree || target_key.sampled)
        throw std::invalid_argument("draft verify requires a linear greedy target key");
    target_key.phase = exec::Phase::verify;
    for (std::uint32_t width = 1; width <= max_width && width <= ring.max_window(); width *= 2) {
        widths_.push_back(width);
        target_key.rows = width;
        for (std::size_t bucket = 1; bucket <= spec::context_bins; ++bucket) {
            target_key.ctx_bucket = static_cast<std::uint8_t>(bucket);
            keys_.push_back(target_key);
        }
    }
}

const exec::ProgramKey& VerifyWindowPlan::key(std::uint32_t width, std::uint64_t frontier) const {
    const auto match = std::ranges::find(widths_, width);
    if (match == widths_.end()) throw std::invalid_argument("unsupported draft verify width");
    const auto index = static_cast<std::size_t>(match - widths_.begin());
    return keys_[index * spec::context_bins + spec::context_bin(frontier)];
}

std::uint32_t VerifyWindowPlan::row_limit(std::uint64_t frontier, std::uint32_t max_rows) const noexcept {
    if (frontier >= max_position) return 0;
    const auto room = max_position - static_cast<std::uint32_t>(frontier);
    return std::bit_floor(std::min({room, max_rows, widths_.back()}));
}

exec::StepBlock VerifyWindowPlan::describe(std::uint32_t width, std::uint64_t frontier, std::uint32_t slot,
                                           std::uint32_t anchor, std::span<const std::uint32_t> drafts) const {
    if (!std::has_single_bit(width) || width > row_limit(frontier, max_width) || drafts.size() >= width)
        throw std::invalid_argument("draft verify rows exceed program width or position room");
    if (anchor == spec::kNoToken || std::ranges::find(drafts, spec::kNoToken) != drafts.end())
        throw std::invalid_argument("draft verify requires token IDs");

    std::array<std::uint32_t, max_width> ids{anchor};
    std::ranges::copy(drafts, ids.begin() + 1);
    const LinearVerifyWindow window{width, static_cast<std::uint32_t>(frontier), slot, 0U,
                                    LinearVerifyRows::exclusive_pending};
    return linear_verify_step(window, std::span(ids).first(drafts.size() + 1U));
}

namespace detail {

spec::Arbiter chance_arbiter(const VerifyWindowPlan& plan, spec::ChanceWindowTables tables) {
    spec::ArbiterConfig config;
    config.mode = spec::ArbiterMode::ChanceWindow;
    config.windows.assign(plan.widths().begin(), plan.widths().end());
    config.chance_window = std::move(tables);
    return spec::Arbiter(std::move(config));
}

void draft_confident_block(exec::DraftRunner& draft, std::uint32_t anchor, std::uint64_t frontier,
                           std::uint32_t k, std::vector<std::uint32_t>& ids, std::vector<float>& confidence) {
    const auto& decl = draft.declaration();
    k = std::min(k, decl.max_k);
    if (k == 0) return;
    const spec::BlockPolicy layout{
        .block = decl.block, .fill_token = decl.noise_token,
        .mask = decl.attention == spec::DraftAttention::block_noncausal ? spec::BlockMask::bidirectional
                                                                      : spec::BlockMask::causal,
        .anchor_proposes = decl.anchor_proposes,
    };
    if (spec::rows_needed(layout, 0, k) > max_position - frontier) return;
    std::vector<std::uint32_t> input;
    spec::block_ids(layout, anchor, {}, k, input);
    draft.forward({input, frontier, 0, k}, ids, confidence);
    if (ids.size() != k || confidence.size() != k)
        throw std::logic_error("P1 block returned the wrong proposal or confidence count");
}

void accept_verify(DraftVerifyStep& result, std::span<const std::uint32_t> drafts,
                   std::span<std::uint32_t> emitted) {
    const auto valid = drafts.size() + 1;
    std::uint32_t first_stop = result.decision.rows;
    for (std::uint32_t i = 0; i < result.decision.rows; ++i) {
        const auto& pick = result.picks[i];
        if (i >= valid) {
            if (pick.status != exec::RowStatus::masked)
                throw std::runtime_error("draft verify returned a pick for padding");
            continue;
        }
        const bool stop = pick.status == exec::RowStatus::stop;
        if ((!stop && pick.status != exec::RowStatus::picked) || pick.id == spec::kNoToken)
            throw std::runtime_error("draft verify did not return a valid pick");
        if (stop) first_stop = std::min(first_stop, i);
        emitted[i] = pick.id;
    }
    const auto matches = spec::greedy_matching_prefix(drafts, emitted.first(valid));
    const auto last = std::min(static_cast<std::uint32_t>(matches), first_stop);
    result.kept = last + 1U;
    result.stopped = last == first_stop;
}

}  // namespace detail
}  // namespace strix::session
