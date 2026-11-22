#pragma once
// Program segments and launch budget (host only, backend neutral).
// A decode step is a launch list: a prelude (embedding, before-trunk stages), a core body of `period` layers repeated
// `repeats` times, and a coda (after-trunk stages, head, sampler). The segments re-describe the SAME launches: they
// never add, remove, merge or reorder a launch, so they change no bits. They give
//   - backends one pre-bound core body to issue `repeats` times (HIP bound-launch array, CUDA graph naming);
//   - the receipt the counts per step and per core layer (launches, kernels, glue launches);
//   - a compile-time launch budget: glue launches (no weight, cache or table bytes in body, prologue or epilogue) per
//     core layer and kernels per step, refused with a reason when a bound program keeps nodes the fusion rules remove.
// Budget values are model data (TOML [fusion]); zero means unbounded. Nothing of a laboratory is compiled in.
// Home: shared (host shared / plan).
#include "strix/graph/launch_plan.hpp"

#include <cstdint>
#include <string>

namespace strix::graph {

struct Segment {
    std::uint32_t first = 0;      // first launch index
    std::uint32_t launches = 0;   // launches in the segment (core: in one period)
    std::uint32_t repeats = 1;    // core: number of periods; prelude / coda: 1
    std::int32_t first_layer = -1;
    std::uint32_t layers = 0;     // core: layers per period
};

struct SegmentOptions {
    std::uint32_t max_period = 8;  // longest layer period searched (hybrid stacks repeat every few layers)
};

struct SegmentPlan {
    Segment prelude, core, coda;
    std::uint32_t launches = 0;              // per step
    std::uint32_t kernels = 0;               // per step (sum of Launch::kernels)
    std::uint32_t glue = 0;                  // glue launches per step
    std::uint32_t core_glue_per_layer = 0;   // most glue launches of one layer inside the core period
    std::uint32_t core_kernels_per_layer = 0;  // most kernels of one layer inside the core period
};

struct LaunchBudget {
    std::uint32_t kernels_per_step = 0;     // 0: unbounded
    std::uint32_t glue_per_core_layer = 0;  // 0: unbounded
};

struct BudgetVerdict {
    bool ok = true;
    std::string reason = {};  // empty when ok
};

// True when no stage of the launch (body, prologue, epilogue) moves weight, cache, table or host bytes or picks:
// every instance is a norm, mix, combine, elementwise or clear stage.
[[nodiscard]] bool glue_launch(const LaunchPlan& plan, const Launch& launch);

// Splits the plan. The core is the longest run of whole layer periods (period <= max_period, at least two repeats)
// whose launches match one by one: entry, kernels, rows, jobs, pack, stream, chains, the declaration and format of
// every instance, and the relative distance of every dependency edge. Everything before is prelude, after is coda.
// Invariant: prelude.launches + core.launches * core.repeats + coda.launches == plan.launches.size().
[[nodiscard]] SegmentPlan segment_plan(const LaunchPlan& plan, const SegmentOptions& options = {});

[[nodiscard]] BudgetVerdict check_budget(const SegmentPlan& segments, const LaunchBudget& budget);

// One line for receipts and effective-config output.
[[nodiscard]] std::string segment_receipt(const SegmentPlan& segments);

}  // namespace strix::graph
