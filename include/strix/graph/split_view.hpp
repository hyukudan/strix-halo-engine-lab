#pragma once

#include "strix/graph/model_decl.hpp"
#include "strix/pair/split_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace strix::graph {

enum class StageTable : std::uint8_t { before, layer, after };

struct SplitStageKey {
    std::string_view id;
    StageTable table;
    std::optional<std::uint32_t> logical_visit;
};

// Costs and join geometry projected from a priced one-device program. An
// indivisible unit can be a head or a block of columns (unit_width > 1).
struct SplitStageCost {
    std::uint64_t units = 1, unit_width = 1, bytes = 0, bytes_per_row = 0;
    double ms_per_row = 0.0;
    pair::split::Join join = pair::split::Join::none;
    std::uint64_t join_bytes_per_row = 0;
    Format wire_format = Format::none;
    // Legacy routed-slot accounting uses a fractional mean per slot.
    std::optional<double> mean_slot_bytes;
    double extra_slot_bytes = 0.0, extra_join_bytes = 0.0;
    std::uint32_t pool = 0;
    double efficiency = 1.0, fixed_ms = 0.0;
};

struct PricedSplitStage {
    SplitStageKey key;
    SplitStageCost cost;
};

struct SplitStage {
    std::string_view id;
    StageTable table;
    std::optional<LayerVisit> visit;
    pair::split::Axis axis;
    Split by;
    bool join_in_rank_order;
    SplitStageCost cost;

    [[nodiscard]] std::int32_t layer() const noexcept;
};

// The program supplies the selected instances in execution order. Stage ids
// borrow from the model. An empty graph id selects the primary graph.
[[nodiscard]] std::vector<SplitStage> split_view(
    const ModelDecl&, std::span<const PricedSplitStage>, std::string_view graph_id = {});

struct SplitRange {
    std::uint64_t first = 0, count = 0;
    friend bool operator==(const SplitRange&, const SplitRange&) = default;
};

struct JoinSlice {
    SplitRange local;
    std::uint64_t destination = 0;
};

// Element offsets for an already selected join layout, including any padding.
struct RankJoinLayout {
    std::uint64_t slot_elements = 0, joined_elements = 0;
    std::vector<JoinSlice> slices;
};

struct RankShard {
    std::vector<SplitRange> units;  // in rank-local order
    std::optional<RankJoinLayout> join_layout;
};

struct TensorPartition {
    std::size_t stage;
    std::vector<RankShard> ranks;   // one per rank; omitted stage partitions replicate
};
struct TensorSplit { std::vector<TensorPartition> partitions; };

struct LayerSegment {
    std::uint32_t rank;
    SplitRange stages;  // in program order; a cut can fall inside a layer
};
struct LayersSplit { std::vector<LayerSegment> segments; };

struct ContextSplit {
    std::uint64_t positions;
    std::vector<SplitRange> ranks;  // one position range per rank
};

struct SplitPlan {
    std::uint32_t ranks;
    std::variant<TensorSplit, LayersSplit, ContextSplit> form;
};

struct RankStage {
    std::size_t stage;
    RankShard shard;
    bool partitioned;
};

struct RankView {
    std::uint32_t rank, ranks;
    std::vector<RankStage> stages;  // indices into the split-stage input
    std::optional<SplitRange> context;
};

[[nodiscard]] RankView split_view(std::span<const SplitStage>, const SplitPlan&, std::uint32_t rank);

}  // namespace strix::graph
