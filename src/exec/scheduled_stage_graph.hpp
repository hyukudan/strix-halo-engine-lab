#pragma once

#include "strix/graph/model_decl.hpp"

namespace strix::exec {

// D3's StageGraph indexes independent trunk layers. Project a D1 graph
// whose guards are phase-invariant onto its ordered logical visits.
// The ModelDecl must outlive this view and any plan compiled from it.
class ScheduledStageGraph {
public:
    explicit ScheduledStageGraph(const graph::ModelDecl& declaration);
    ScheduledStageGraph(const ScheduledStageGraph&) = delete;
    ScheduledStageGraph& operator=(const ScheduledStageGraph&) = delete;

    [[nodiscard]] const graph::StageGraph& stages() const noexcept { return stages_; }

private:
    std::vector<graph::StageDecl> layer_;
    std::vector<graph::FormatRule> formats_;
    std::vector<std::vector<std::uint32_t>> format_visits_;
    graph::StageGraph stages_;
};

}  // namespace strix::exec
