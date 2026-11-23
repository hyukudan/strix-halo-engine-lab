#include "strix/graph/program_segments.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix::graph {
namespace {

bool glue_op(Op op) {
    switch (op) {
    case Op::combine: case Op::norm: case Op::mix: case Op::elementwise: case Op::clear: return true;
    default: return false;
    }
}

const Instance& instance_at(const LaunchPlan& plan, std::uint32_t index) {
    if (index >= plan.instances.size() || plan.instances[index].decl == nullptr)
        throw std::invalid_argument("program segments: launch names an undeclared instance");
    return plan.instances[index];
}

bool same_instances(const LaunchPlan& plan, const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto& x = instance_at(plan, a[i]);
        const auto& y = instance_at(plan, b[i]);
        if (x.decl != y.decl || x.format != y.format) return false;
    }
    return true;
}

// Signed distance from launch `self` to `target`; a missing edge maps to a value no real distance takes.
std::int64_t back(std::uint32_t self, std::uint32_t target) {
    if (target == no_launch) return std::numeric_limits<std::int64_t>::min();
    return std::int64_t{self} - std::int64_t{target};
}

bool same_launch(const LaunchPlan& plan, std::uint32_t ia, std::uint32_t ib) {
    const Launch& a = plan.launches[ia];
    const Launch& b = plan.launches[ib];
    if (a.entry != b.entry || a.kernels != b.kernels || a.rows != b.rows || a.jobs != b.jobs || a.pack != b.pack ||
        a.stream != b.stream || a.sidecar != b.sidecar || a.dependent != b.dependent || a.during != b.during ||
        a.prologue_chain != b.prologue_chain || a.job_prologue_chain != b.job_prologue_chain ||
        a.epilogue_chain != b.epilogue_chain || a.job_keys != b.job_keys || a.publish_frames != b.publish_frames ||
        a.regime.phase != b.regime.phase || a.regime.rows != b.regime.rows || a.regime.pos != b.regime.pos)
        return false;
    if (!same_instances(plan, a.stages, b.stages) || !same_instances(plan, a.prologue, b.prologue) ||
        !same_instances(plan, a.epilogue, b.epilogue))
        return false;
    if (back(ia, a.after) != back(ib, b.after) || a.wait_for.size() != b.wait_for.size()) return false;
    for (std::size_t i = 0; i < a.wait_for.size(); ++i)
        if (back(ia, a.wait_for[i]) != back(ib, b.wait_for[i])) return false;
    return true;
}

struct Group {
    std::uint32_t first = 0, count = 0;
    std::int32_t layer = -1;
};

bool same_group(const LaunchPlan& plan, const Group& a, const Group& b) {
    if (a.count != b.count) return false;
    for (std::uint32_t i = 0; i < a.count; ++i)
        if (!same_launch(plan, a.first + i, b.first + i)) return false;
    return true;
}

std::uint32_t u32(std::size_t v) {
    if (v > no_launch) throw std::length_error("program segments: launch count exceeds 32 bits");
    return static_cast<std::uint32_t>(v);
}

}  // namespace

bool glue_launch(const LaunchPlan& plan, const Launch& launch) {
    for (const auto* list : {&launch.stages, &launch.prologue, &launch.epilogue})
        for (const auto index : *list)
            if (!glue_op(instance_at(plan, index).decl->op)) return false;
    return !launch.stages.empty();
}

SegmentPlan segment_plan(const LaunchPlan& plan, const SegmentOptions& options) {
    SegmentPlan out;
    const std::uint32_t total = u32(plan.launches.size());
    out.launches = total;
    out.kernels = plan.kernel_count();
    std::vector<std::uint8_t> glue(total, 0);
    for (std::uint32_t i = 0; i < total; ++i) {
        glue[i] = glue_launch(plan, plan.launches[i]) ? std::uint8_t{1} : std::uint8_t{0};
        out.glue += std::uint32_t{glue[i]};
    }

    // Trunk: from the first to the last launch with a layer; a layer-less launch inside joins the preceding layer.
    std::uint32_t first = total, last = 0;
    for (std::uint32_t i = 0; i < total; ++i)
        if (plan.launches[i].layer >= 0) { first = std::min(first, i); last = i; }
    std::vector<Group> groups;
    if (first < total) {
        for (std::uint32_t i = first; i <= last; ++i) {
            const std::int32_t layer = plan.launches[i].layer;
            if (groups.empty() || (layer >= 0 && layer != groups.back().layer)) {
                if (!groups.empty() && layer != groups.back().layer + 1) {  // non-consecutive or decreasing layers
                    last = i - 1;
                    break;
                }
                groups.push_back({i, 0, layer});
            }
            ++groups.back().count;
        }
    }

    // Longest run of whole periods; ties go to the shorter period, then the earlier start.
    std::uint32_t best_start = 0, best_period = 0, best_repeats = 0;
    const auto n = u32(groups.size());
    for (std::uint32_t period = 1; period <= options.max_period && 2 * period <= n; ++period) {
        std::uint32_t start = 0;
        while (start + 2 * period <= n) {
            std::uint32_t matched = 0;  // groups k in [start, start + matched) with k + period also matching
            while (start + matched + period < n && same_group(plan, groups[start + matched], groups[start + matched + period]))
                ++matched;
            const std::uint32_t repeats = matched / period + 1;
            if (repeats >= 2 && repeats * period > best_repeats * best_period) {
                best_start = start; best_period = period; best_repeats = repeats;
            }
            start += std::max<std::uint32_t>(matched, 1);
        }
    }

    if (best_repeats >= 2) {
        const Group& a = groups[best_start];
        const Group& end = groups[best_start + best_period - 1];
        out.core = {a.first, end.first + end.count - a.first, best_repeats, a.layer, best_period};
        for (std::uint32_t k = best_start; k < best_start + best_period; ++k) {
            std::uint32_t g = 0, kernels = 0;
            for (std::uint32_t i = groups[k].first; i < groups[k].first + groups[k].count; ++i) {
                g += std::uint32_t{glue[i]};
                kernels += plan.launches[i].kernels;
            }
            out.core_glue_per_layer = std::max(out.core_glue_per_layer, g);
            out.core_kernels_per_layer = std::max(out.core_kernels_per_layer, kernels);
        }
        out.prelude = {0, a.first, 1, -1, 0};
    } else {
        out.core = {0, 0, 0, -1, 0};
        out.prelude = {0, total, 1, -1, 0};
    }
    const std::uint32_t core_end = out.core.repeats ? out.core.first + out.core.launches * out.core.repeats : total;
    out.coda = {core_end, total - core_end, 1, -1, 0};
    if (out.prelude.launches + out.core.launches * out.core.repeats + out.coda.launches != total)
        throw std::logic_error("program segments: segments do not cover the launch list");
    return out;
}

BudgetVerdict check_budget(const SegmentPlan& s, const LaunchBudget& budget) {
    if (budget.kernels_per_step && s.kernels > budget.kernels_per_step)
        return {false, "launch budget: " + std::to_string(s.kernels) + " kernels per step > budget " +
                           std::to_string(budget.kernels_per_step)};
    if (budget.glue_per_core_layer && s.core.repeats && s.core_glue_per_layer > budget.glue_per_core_layer)
        return {false, "launch budget: " + std::to_string(s.core_glue_per_layer) +
                           " glue launches in one core layer > budget " + std::to_string(budget.glue_per_core_layer)};
    return {};
}

std::string segment_receipt(const SegmentPlan& s) {
    return "segments prelude=" + std::to_string(s.prelude.launches) + " core=" + std::to_string(s.core.launches) + "x" +
           std::to_string(s.core.repeats) + " (period " + std::to_string(s.core.layers) + " layers from layer " +
           std::to_string(s.core.first_layer) + ") coda=" + std::to_string(s.coda.launches) +
           " launches=" + std::to_string(s.launches) + " kernels=" + std::to_string(s.kernels) +
           " glue=" + std::to_string(s.glue) + " glue_per_core_layer=" + std::to_string(s.core_glue_per_layer) +
           " kernels_per_core_layer=" + std::to_string(s.core_kernels_per_layer);
}

}  // namespace strix::graph
