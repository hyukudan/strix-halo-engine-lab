#include "strix/graph/split_view.hpp"
#include "strix/graph/value.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace strix::graph {
namespace {

std::int64_t reference_number(std::string_view text, std::string_view reference) {
    std::int64_t number = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), number);
    if (ec != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("invalid layer in value reference " + std::string(reference));
    return number;
}

pair::split::Axis axis_of(Split by) {
    using pair::split::Axis;
    switch (by) {
    case Split::none:
    case Split::replicate: return Axis::replicated;
    case Split::by_heads: return Axis::heads;
    case Split::by_experts: return Axis::slots;
    case Split::by_k: return Axis::columns;
    case Split::by_vocab: return Axis::vocab;
    }
    throw std::invalid_argument("unknown stage split");
}

struct TableView {
    std::string_view name;
    std::span<const StageDecl> stages;
};

TableView stage_table(const StageGraph& graph, StageTable table) {
    switch (table) {
    case StageTable::before: return {"before", graph.before};
    case StageTable::layer: return {"layer", graph.layer};
    case StageTable::after: return {"after", graph.after};
    }
    throw std::invalid_argument("unknown stage table");
}

bool rank_order(const StageFields& row) {
    const auto field = row.fields.find("join_order");
    if (field == row.fields.end()) return false;
    const auto* word = std::get_if<std::string>(&field->second.data);
    if (word && *word == "rank") return true;
    throw std::invalid_argument("unknown join order for " + row.id);
}

void check_cost(const SplitStageCost& cost, std::string_view id) {
    const auto nonnegative = [](double x) { return std::isfinite(x) && x >= 0.0; };
    if (cost.units == 0 || cost.unit_width == 0 ||
        !nonnegative(cost.ms_per_row) || !nonnegative(cost.fixed_ms) ||
        !nonnegative(cost.extra_slot_bytes) || !nonnegative(cost.extra_join_bytes) ||
        (cost.mean_slot_bytes && !nonnegative(*cost.mean_slot_bytes)) ||
        !nonnegative(cost.efficiency) || cost.efficiency == 0.0 || cost.efficiency > 1.0)
        throw std::invalid_argument("invalid split cost for " + std::string(id));
}

void check_range(SplitRange range, std::uint64_t extent) {
    if (range.count == 0 || range.first > extent || range.count > extent - range.first)
        throw std::invalid_argument("invalid split range");
}

void check_ranges(std::vector<SplitRange>& ranges, std::uint64_t extent, bool cover) {
    std::ranges::sort(ranges, {}, &SplitRange::first);
    std::uint64_t end = 0;
    for (const auto range : ranges) {
        check_range(range, extent);
        if (range.first < end || (cover && range.first != end))
            throw std::invalid_argument("overlapping split ranges or gap in partition");
        end = range.first + range.count;
    }
    if (cover && end != extent) throw std::invalid_argument("incomplete split partition");
}

void check_tensor_partition(const TensorPartition& partition, const SplitStage& stage) {
    std::vector<SplitRange> units, destinations;
    const auto& geometry = partition.ranks.front().join_layout;
    for (const auto& shard : partition.ranks) {
        units.insert(units.end(), shard.units.begin(), shard.units.end());
        if (!geometry && !shard.join_layout) continue;
        if (stage.cost.join == pair::split::Join::none || !geometry || !shard.join_layout)
            throw std::invalid_argument("inconsistent tensor join layout");
        const auto& join = *shard.join_layout;
        if (join.slot_elements == 0 || join.joined_elements == 0 ||
            join.slot_elements != geometry->slot_elements || join.joined_elements != geometry->joined_elements)
            throw std::invalid_argument("inconsistent join extents");
        std::vector<SplitRange> local;
        for (const auto& slice : join.slices) {
            local.push_back(slice.local);
            destinations.push_back({slice.destination, slice.local.count});
        }
        check_ranges(local, join.slot_elements, false);
    }
    check_ranges(units, stage.cost.units, true);
    if (geometry) check_ranges(destinations, geometry->joined_elements, false);
}

RankStage whole_stage(std::size_t i, const SplitStage& stage) {
    return {i, {{{0, stage.cost.units}}, std::nullopt}, false};
}

}  // namespace

ValueScope parse_value_scope(std::string_view word) {
    if (word == "step") return ValueScope::step;
    if (word == "visit") return ValueScope::visit;
    if (word == "pass") return ValueScope::pass;
    if (word == "model") return ValueScope::model;
    throw std::invalid_argument("unknown value scope " + std::string(word));
}

ValueRef parse_value_ref(std::string_view text) {
    const auto separator = text.find_first_of("@[");
    const auto id = text.substr(0, separator);
    if (id.empty() || id.find_first_of("] \t\r\n") != std::string_view::npos)
        throw std::invalid_argument("invalid value identifier " + std::string(text));
    if (separator == std::string_view::npos) return {id};
    const auto selector = text.substr(separator);
    if (selector == "[L]") return {id, ValueSelector::model_layer_slice};
    if (selector == "@L") return {id, ValueSelector::relative_layer};
    if (selector.starts_with("@L-"))
        return {id, ValueSelector::relative_layer, reference_number(selector.substr(2), text)};
    const bool relative = selector.starts_with("@L+");
    const auto number = selector.substr(relative ? 3U : 1U);
    if (selector.front() != '@' || number.empty() || number.front() < '0' || number.front() > '9')
        throw std::invalid_argument("invalid value reference " + std::string(text));
    return {id, relative ? ValueSelector::relative_layer : ValueSelector::absolute_layer, reference_number(number, text)};
}

ResolvedValueRef resolve_value_ref(const ValueRef& ref, std::span<const ValueDecl> values,
                                   std::span<const LayerVisit> schedule, const LayerVisit* current) {
    const auto value = std::ranges::find(values, ref.id, &ValueDecl::id);
    if (value == values.end()) throw std::invalid_argument("unknown value " + std::string(ref.id));
    if (value->streams == 0 || value->width == 0) throw std::invalid_argument("empty value shape");
    ResolvedValueRef result{*value, std::nullopt, std::nullopt};
    switch (ref.selector) {
    case ValueSelector::current:
        if (current) result.logical_visit = current->logical_layer;
        return result;
    case ValueSelector::model_layer_slice:
        if (!current || value->scope != ValueScope::model ||
            std::ranges::find(schedule, current->physical_layer, &LayerVisit::physical_layer) == schedule.end())
            throw std::invalid_argument("a model value slice requires a model-scope value and layer");
        result.model_slice = current->physical_layer;
        return result;
    case ValueSelector::absolute_layer:
    case ValueSelector::relative_layer:
        break;
    default:
        throw std::invalid_argument("unknown value selector");
    }

    std::int64_t target = ref.layer;
    constexpr auto max_layer = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    if (ref.selector == ValueSelector::relative_layer) {
        if (!current) throw std::invalid_argument("relative value reference outside a layer");
        const auto layer = static_cast<std::int64_t>(current->physical_layer);
        if (target < -layer || target > max_layer - layer)
            throw std::invalid_argument("value reference outside physical layers");
        target += layer;
    }
    if (target < 0 || target > max_layer) throw std::invalid_argument("value reference outside physical layers");

    // Same visit ordinal as the consumer at this layer, clamped to the last
    // visit if the producer's layer is visited fewer times (D1 schedule rule).
    auto ordinal = current ? current->pass : 0U;
    for (const auto& visit : schedule) {
        if (visit.physical_layer != static_cast<std::uint32_t>(target)) continue;
        result.logical_visit = visit.logical_layer;
        if (ordinal == 0) break;
        --ordinal;
    }
    if (!result.logical_visit) throw std::invalid_argument("value reference outside physical layers");
    return result;
}

std::int32_t SplitStage::layer() const noexcept {
    return visit ? static_cast<std::int32_t>(visit->physical_layer) : -1;
}

std::vector<SplitStage> split_view(const ModelDecl& model, std::span<const PricedSplitStage> program,
                                  std::string_view graph_id) {
    const auto graphs = model.graphs();
    const auto graph = graph_id.empty() ? graphs.begin() : std::ranges::find(graphs, graph_id, &DeclGraph::id);
    if (graph == graphs.end()) throw std::invalid_argument("unknown split graph " + std::string(graph_id));
    std::vector<SplitStage> result;
    result.reserve(program.size());
    std::set<std::tuple<StageTable, std::string_view, std::optional<std::uint32_t>>> seen;
    for (const auto& priced : program) {
        const auto& [id, table, logical_visit] = priced.key;
        const auto stages = stage_table(graph->stages, table);
        const auto decl = std::ranges::find(stages.stages, id, &StageDecl::id);
        const auto row = std::ranges::find_if(graph->rows, [&](const StageFields& detail) {
            return detail.id == id && detail.table == stages.name;
        });
        if (decl == stages.stages.end() || row == graph->rows.end())
            throw std::invalid_argument("unknown program stage " + std::string(id));
        if (!seen.emplace(table, decl->id, logical_visit).second)
            throw std::invalid_argument("duplicate program stage " + std::string(id));
        if ((table == StageTable::layer) != logical_visit.has_value())
            throw std::invalid_argument("wrong visit for program stage " + std::string(id));
        std::optional<LayerVisit> visit;
        if (logical_visit) {
            const auto active = std::ranges::lower_bound(row->active_visits, *logical_visit, {},
                                                        &LayerVisit::logical_layer);
            if (active == row->active_visits.end() || active->logical_layer != *logical_visit ||
                active->physical_layer > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
                throw std::invalid_argument("invalid visit for program stage " + std::string(id));
            visit = *active;
        }
        check_cost(priced.cost, id);
        result.push_back({decl->id, table, visit, axis_of(decl->split), decl->split, rank_order(*row), priced.cost});
    }
    return result;
}

RankView split_view(std::span<const SplitStage> stages, const SplitPlan& plan, std::uint32_t rank) {
    if (rank >= plan.ranks) throw std::invalid_argument("rank outside split plan");
    RankView result{rank, plan.ranks, {}, std::nullopt};
    if (const auto* tensor = std::get_if<TensorSplit>(&plan.form)) {
        std::vector<const TensorPartition*> by_stage(stages.size());
        for (const auto& part : tensor->partitions) {
            if (part.stage >= stages.size() || by_stage[part.stage] || part.ranks.size() != plan.ranks ||
                stages[part.stage].axis == pair::split::Axis::replicated)
                throw std::invalid_argument("invalid tensor partition");
            check_tensor_partition(part, stages[part.stage]);
            by_stage[part.stage] = &part;
        }
        for (std::size_t i = 0; i < stages.size(); ++i) {
            const auto* part = by_stage[i];
            result.stages.push_back(part ? RankStage{i, part->ranks[rank], true} : whole_stage(i, stages[i]));
        }
    } else if (const auto* layers = std::get_if<LayersSplit>(&plan.form)) {
        std::uint64_t end = 0;
        for (const auto& segment : layers->segments) {
            check_range(segment.stages, stages.size());
            if (segment.rank >= plan.ranks || segment.stages.first != end)
                throw std::invalid_argument("invalid layer segment");
            end += segment.stages.count;
            if (segment.rank == rank) {
                for (auto i = static_cast<std::size_t>(segment.stages.first); i < end; ++i)
                    result.stages.push_back(whole_stage(i, stages[i]));
            }
        }
        if (end != stages.size()) throw std::invalid_argument("incomplete layer partition");
    } else {
        const auto& context = std::get<ContextSplit>(plan.form);
        if (context.positions == 0 || context.ranks.size() != plan.ranks)
            throw std::invalid_argument("invalid context partition");
        auto ranges = context.ranks;
        check_ranges(ranges, context.positions, true);
        result.context = context.ranks[rank];
        for (std::size_t i = 0; i < stages.size(); ++i)
            result.stages.push_back(whole_stage(i, stages[i]));
    }
    return result;
}

}  // namespace strix::graph
