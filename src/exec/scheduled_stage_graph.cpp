#include "scheduled_stage_graph.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix::exec {

ScheduledStageGraph::ScheduledStageGraph(const graph::ModelDecl& declaration)
    : layer_(declaration.stages.layer.begin(), declaration.stages.layer.end()),
      format_visits_(declaration.stages.formats.size()), stages_(declaration.stages) {
    if (declaration.graphs().empty()) throw std::invalid_argument("scheduled graph has no declaration");
    const auto& rows = declaration.graphs().front().rows;
    const auto& schedule = declaration.schedule;
    stages_.layers.assign(schedule.size(), graph::Traits{});
    for (std::size_t i = 0; i < schedule.size(); ++i) {
        if (schedule[i].logical_layer != i || schedule[i].physical_layer >= declaration.stages.layers.size())
            throw std::invalid_argument("invalid logical visit in scheduled graph");
    }

    unsigned guard_count = 0;
    for (auto& stage : layer_) {
        const auto details = std::ranges::find_if(rows, [&](const auto& row) {
            return row.table == "layer" && row.id == stage.id;
        });
        if (details == rows.end() || details->active_visits.empty())
            throw std::invalid_argument("missing visit guard for " + std::string(stage.id));
        stage.when_all = 0;
        stage.when_none = 0;
        if (details->active_visits.size() == schedule.size()) continue;
        if (guard_count == std::numeric_limits<graph::Traits>::digits)
            throw std::invalid_argument("too many guards in scheduled graph");
        stage.when_all = graph::Traits{1} << guard_count++;
        for (const auto& visit : details->active_visits) {
            if (schedule.at(visit.logical_layer) != visit)
                throw std::invalid_argument("invalid guarded visit in scheduled graph");
            stages_.layers[visit.logical_layer] |= stage.when_all;
        }
    }
    stages_.layer = layer_;

    for (std::size_t i = 0; i < stages_.formats.size(); ++i) {
        auto rule = stages_.formats[i];
        if (!rule.layers.empty()) {
            auto& visits = format_visits_[i];
            for (const auto& visit : schedule) {
                if (std::ranges::find(rule.layers, visit.physical_layer) != rule.layers.end())
                    visits.push_back(visit.logical_layer);
            }
            if (visits.empty()) continue;
            rule.layers = visits;
        }
        formats_.push_back(rule);
    }
    stages_.formats = formats_;
}

}  // namespace strix::exec
