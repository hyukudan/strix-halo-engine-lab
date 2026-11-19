#pragma once
// Launch planner (host only, backend neutral): stage graph -> ordered launches of one decode step.
// Fusion::today (CHECK MODE): the engine's present launch list, derived from the declaration plus the engine's present
//   capabilities given as data (FusionPolicy: where same-input projections share a launch, which producer entries take
//   which epilogue forms). A model's declaration is accepted only when this list equals the engine's hand-wired one.
// Fusion::fused: the rules of the shared plan, independent of today's kernels:
//   1. linear stages with the same input value and format share one launch within a layer (decode_projection_group.hpp
//      Segments; members move up to the first member when their other inputs are ready there);
//   2. a single-input (or tile-local) elementwise / store / copy / combine stage becomes an epilogue of the launch that
//      produces its last input, when every other input is produced by an earlier launch;
//   3. a reduction (norm, weighted merge) or a stage whose inputs come from different members of one grouped launch
//      becomes a prologue of every consumer launch whose head is linear / head / seam; otherwise it opens a launch and its
//      elementwise consumers fold into it (rule 2); two adjacent independent norm chains of one layer share that launch;
//   4. a hyper-connection expand followed by the next site's mix (and the collapse, by rule 2) is one seam launch;
//   5. a clear becomes an epilogue of the previous launch.
// Rows: the plan of verify / window rows is the same list; rows is recorded per launch (rows_plan.hpp prices it).
// Home: shared (host shared / plan).
#include "strix/graph/stage_graph.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace strix::graph {

enum class Fusion : std::uint8_t { today, fused };

struct EpilogueCap {
    std::string_view entry = {}, form = {};  // the producer entry takes this elementwise form in its store today
};
struct FusionPolicy {
    Fusion mode = Fusion::today;
    // today: adjacent linears with one input share a launch when their format is listed and their site is listed
    // (empty site list = every site).
    std::span<const Format> group_formats = {};
    std::span<const std::string_view> group_sites = {};
    std::span<const EpilogueCap> epilogues = {};
};

struct Launch {
    std::string_view entry = {};               // entry of the head stage
    std::vector<std::uint32_t> stages = {};    // instance indices computed by the launch body (head first, group members)
    std::vector<std::uint32_t> prologue = {};  // instance indices recomputed / applied on the launch input
    std::vector<std::uint32_t> epilogue = {};  // instance indices applied in the launch store
    std::uint32_t kernels = 1;            // today: kernels the entry call issues (fused: 1)
    std::int32_t layer = -1;
    std::uint32_t rows = 1;
};

struct LaunchPlan {
    std::vector<Instance> instances = {};
    std::vector<Launch> launches = {};
    // Kernel launches (sum of kernels), in total and for one layer (-1: before / after the trunk).
    [[nodiscard]] std::uint32_t kernel_count() const;
    [[nodiscard]] std::uint32_t kernel_count(std::int32_t layer) const;
};

[[nodiscard]] LaunchPlan launch_plan(const StageGraph& g, std::uint32_t slot, std::uint32_t rows, const FusionPolicy& policy);

}  // namespace strix::graph
