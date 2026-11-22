#include "strix/graph/launch_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    using namespace strix::graph;
    // Indexed attention has two projections consuming the same in-place query norm.
    constexpr StageDecl layer[] = {
        {.id = "query_a", .op = Op::linear, .entry = "query_a", .in = {"input"}, .out = "qa",
         .operand = weights("qa.weight", 64, 64)},
        {.id = "query_norm", .op = Op::norm, .form = "norm.rms", .entry = "query_norm", .in = {"qa"}, .out = "qa",
         .reduce = true},
        {.id = "query_b", .op = Op::linear, .entry = "query_b", .in = {"qa"}, .out = "q",
         .operand = weights("qb.weight", 64, 64)},
        {.id = "index_query", .op = Op::linear, .entry = "index_query", .in = {"qa"}, .out = "iq",
         .operand = weights("iq.weight", 64, 64), .when_all = trait::attn_indexed, .pos = Pos::prefix_select},
    };
    constexpr FormatRule formats[] = {{.format = Format::f32}};
    const StageGraph graph{.layer = layer, .layers = {trait::attn_indexed}, .formats = formats, .select_after = 2048U};
    validate(graph);

    constexpr std::uint32_t slot = 3000U;
    const auto legacy = launch_plan(graph, slot, 1U, {.mode = Fusion::fused});
    const auto bound = launch_plan(graph, PlanRequest{.slot = slot});
    for (const auto* plan : {&legacy, &bound}) {
        if (plan->launches.size() != 2U || plan->kernel_count() != 2U ||
            plan->launches[0].stages != std::vector{0U} || !plan->launches[0].prologue.empty() ||
            plan->launches[1].stages != std::vector{2U, 3U} || plan->launches[1].prologue != std::vector{1U}) {
            std::fputs("indexed fused plan has incorrect launches or prologue\n", stderr);
            return 1;
        }
    }
    std::puts("indexed_fusion PASS launches=2 prologues=1");
}
