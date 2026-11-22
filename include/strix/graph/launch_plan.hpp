#pragma once
// Launch planner (host only, backend neutral): stage graph -> ordered launches of one phase and position regime.
// Fusion::today (CHECK MODE): the engine's present launch list, derived from the declaration plus the engine's present
//   capabilities given as data (FusionPolicy: where same-input projections share a launch, which producer entries take
//   which epilogue forms). A model's declaration is accepted only when this list equals the engine's hand-wired one.
// Fusion::fused through FusionPolicy uses the original five rules. PlanRequest also applies rules 6-16:
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
#include "strix/graph/chain.hpp"
#include "strix/graph/stage_graph.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace strix::device { struct Profile; }
namespace strix::exec { class Catalog; }
namespace strix::profile { enum class Name : unsigned; }

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

enum class Phase : std::uint8_t { decode, prefill, verify, state_only };
struct Regime {
    Phase phase = Phase::decode;
    Rows rows = Rows::per_row;
    Pos pos = Pos::any;
};

inline constexpr std::uint32_t no_launch = std::numeric_limits<std::uint32_t>::max();

struct Launch {
    std::string_view entry = {};               // entry of the head stage
    std::vector<std::uint32_t> stages = {};    // instance indices computed by the launch body (head first, group members)
    std::vector<std::uint32_t> prologue = {};  // instance indices recomputed / applied on the launch input
    std::vector<std::uint32_t> epilogue = {};  // instance indices applied in the launch store
    std::uint32_t kernels = 1;            // today: kernels the entry call issues (fused: 1)
    std::int32_t layer = -1;
    std::uint32_t rows = 1;
    ChainId prologue_chain = 0;               // shared by every job, or empty when job chains differ
    std::array<ChainId, 8> job_prologue_chain = {};
    std::array<ChainId, 8> epilogue_chain = {};
    std::array<std::string_view, 8> job_keys = {};
    std::array<std::uint8_t, 8> publish_frames = {};
    std::uint8_t jobs = 1;
    Pack pack = Pack::none;
    Regime regime = {};
    std::uint8_t stream = 0;
    std::uint32_t after = no_launch;           // previous launch of this stream, or a dependent-launch edge
    std::vector<std::uint32_t> wait_for = {};  // data/hazard predecessors on other streams
    std::string_view during = {};             // exchange ID for a triggered sidecar
    bool sidecar = false;
    bool dependent = false;                   // catalog pipe fallback using a dependent launch
};

struct LaunchPlan {
    std::vector<Instance> instances = {};
    std::vector<Launch> launches = {};
    ChainTable chains = {};
    // Kernel launches (sum of kernels), in total and for one layer (-1: before / after the trunk).
    [[nodiscard]] std::uint32_t kernel_count() const;
    [[nodiscard]] std::uint32_t kernel_count(std::int32_t layer) const;
};

// Host bindings projected from kernel capabilities for the requested device and tier. They
// identify catalog entries by name. The declaration and all binding strings outlive the plan.
struct PlanStage {
    const StageDecl* stage = nullptr;
    std::string_view key = {};
    Link form = {};
    Form operand_store = Form::none;          // store for the first input's activation class
    Link input_transform = {};               // first-input transform, with an arg slot shared by producer and consumers
    Link head_emit = {};                      // emit.argmax, emit.top2 or emit.topn
    std::uint32_t tiles = 0;                   // zero means grid size is unknown
    std::string_view wave_key = {};
    std::string_view split_k_key = {};        // deterministic ordered fold
    std::uint32_t pool_keys = 0;               // known select pool bound
    std::uint8_t publish_frames = 1;
    bool paired_tile = false;                // same-input grouped operands share an output tile
    bool sample = false;                     // sampler stage following a pick
    // Opt in a unary tail to a concrete catalogue variant (rule 2). Both
    // keys come from the backend's exact capability projection.
    std::string_view exact_epilogue_head = {}, exact_epilogue_entry = {};
    bool require_exact_epilogue = false;
};

struct PlanPack {
    Pack pack = Pack::none;                   // fixed sum or pipe instance
    std::string_view key = {};                // empty pipe key requests separate launches
    std::span<const std::string_view> jobs = {};
};

struct PlanSidecar {
    std::string_view parent_key = {}, key = {}, during = {};  // during names an exchange join or host wait stage
    std::uint8_t stream = 1;
};

struct PlanOverlap {
    std::string_view main_key = {}, branch_key = {};
    double serial_us = 0, concurrent_us = 0;  // measured on the requested device
    std::uint8_t stream = 1;
};

struct PlanRequest {
    Phase phase = Phase::decode;
    std::uint32_t rows = 1;
    Pos pos = Pos::any;
    profile::Name tier = {};
    const exec::Catalog* catalog = nullptr;   // opaque until the catalog projects the bindings here
    const device::Profile* device = nullptr;
    std::optional<std::uint32_t> slot = {};   // omitted: use a representative slot for pos
    std::span<const PlanStage> stages = {};
    std::span<const PlanPack> packs = {};
    std::span<const PlanSidecar> sidecars = {};
    std::span<const PlanOverlap> overlap = {};
    std::uint32_t waves = 2;
};

[[nodiscard]] LaunchPlan launch_plan(const StageGraph& g, std::uint32_t slot, std::uint32_t rows, const FusionPolicy& policy);
[[nodiscard]] LaunchPlan launch_plan(const StageGraph& g, const PlanRequest& request);
void validate_plan(const LaunchPlan& plan);
[[nodiscard]] std::uint64_t plan_identity(const LaunchPlan& plan);

}  // namespace strix::graph
