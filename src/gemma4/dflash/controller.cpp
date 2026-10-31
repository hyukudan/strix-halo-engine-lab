#include "device.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::gemma4::dflash {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}
CycleResult run_device_cycle(Target& target, KvTxn& transaction, DeviceDrafter& drafter,
    std::uint32_t anchor, std::uint32_t k, std::size_t budget, std::span<const std::uint32_t> eos,
    Acceptance* stats, DeviceTimings* timings, std::uint64_t stream, std::uint64_t generation) {
    CycleResult result;
    result.next_anchor = anchor;
    if (timings) *timings = {};
    if (!budget) return result;
    const auto position = transaction.frontier();
    if (drafter.frontier() != position)
        throw std::runtime_error("dflash: target/draft frontier mismatch");
    k = admit_window(k, position, budget, enabled_from_environment(), false);
    std::array<std::uint32_t, max_drafts> tokens{};
    if (k) {
        tokens = drafter.draft(anchor, position);
        if (timings) {
            timings->block_ms = drafter.timings().block_ms;
            timings->head_ms = drafter.timings().head_ms;
        }
    }
    std::vector<RowPlan> plan(k + 1);
    for (std::uint32_t r = 0; r <= k; ++r)
        plan[r] = {stream, generation, position + r, r ? tokens[r - 1] : anchor, &transaction, r, true};
    auto start = Clock::now();
    const auto forward = target.forward_rows(plan);
    if (timings) timings->verify_ms = elapsed(start);
    if (forward.rows != plan.size() || !forward.vocab || forward.tap_width != drafter.shape().tap_width() ||
        forward.logits.size() != forward.rows * forward.vocab ||
        forward.taps.size() != forward.rows * forward.tap_width)
        throw std::runtime_error("dflash: invalid device target result");
    std::vector<std::uint32_t> greedy(plan.size());
    for (std::size_t r = 0; r < plan.size(); ++r) {
        float best = -std::numeric_limits<float>::infinity();
        for (std::uint32_t id = 0; id < forward.vocab; ++id) {
            const float x = forward.logits[r * forward.vocab + id];
            if (!std::isfinite(x)) throw std::runtime_error("dflash: nonfinite target logit");
            if (better(x, id, best, greedy[r])) { best = x; greedy[r] = id; }
        }
    }
    Acceptance staged = stats ? *stats : Acceptance{};
    result.verdict = verify_greedy(std::span<const std::uint32_t>(tokens).first(k), greedy, budget, eos, &staged);
    const auto retained = std::span(forward.taps).first(std::size_t{result.verdict.processed} * forward.tap_width);
    result.committed_taps.assign(retained.begin(), retained.end());
    for (std::uint32_t r = 0; r < result.verdict.processed; ++r)
        result.committed_positions.push_back(position + r);
    drafter.stage_append(result.committed_taps, result.committed_positions);
    if (timings) timings->append_ms = drafter.timings().append_ms;
    start = Clock::now();
    try {
        transaction.commit(result.verdict.processed);
        drafter.publish_append();
    } catch (...) { drafter.discard_append(); throw; }
    if (timings) timings->commit_ms = elapsed(start);
    if (stats) *stats = staged;
    result.next_anchor = result.verdict.emitted.back();
    return result;
}
} // namespace strix::gemma4::dflash
