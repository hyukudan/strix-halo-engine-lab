#pragma once

#include "strix/exec/catalog.hpp"
#include "strix/graph/launch_plan.hpp"

namespace strix::exec {

struct ChainStageContract {
    std::string_view entry;
    std::string_view numeric;
};

struct RowEpilogueStep {
    ChainStageContract incumbent;
    graph::Link link;
};

// One backend/regime capability for the entire ordered sequence. An exact
// fixture compares this fused catalogue entry with the declared incumbents.
// Only project FAST capabilities admitted for the requested regime.
struct RowEpilogueVariant {
    ChainStageContract head;
    graph::Format head_format;
    std::uint64_t output_width, input_width;  // head operand geometry
    std::uint32_t rows;
    device::ArchMask arch;
    std::span<const RowEpilogueStep> steps;
    KernelKey key;
    std::string_view exact_fixture = {};
};

struct RowEpilogueRequest {
    graph::FusionPolicy incumbent{};
    std::uint32_t slot = 0;
    std::uint32_t rows = 1;
    profile::Name tier = {};
    device::ArchMask arch = device::ArchMask::none;
    std::span<const RowEpilogueVariant> variants{};
};

struct RowEpilogueSelection {
    const KernelEntry* entry = nullptr;
    std::string_view exact_fixture = {};
};

struct RowEpiloguePlan {
    graph::LaunchPlan graph;
    // Indexed in launch order. A null entry keeps the incumbent launch.
    std::vector<RowEpilogueSelection> selections;
};

// First attachment rule: consecutive, single-job row epilogues in one layer
// site, with private intermediates. The graph declares all uses of those
// values. Declaration, catalogue and fixture names outlive the plan.
[[nodiscard]] RowEpiloguePlan plan_row_epilogues(const graph::StageGraph&, const Catalog&,
                                                const RowEpilogueRequest&);

}  // namespace strix::exec
