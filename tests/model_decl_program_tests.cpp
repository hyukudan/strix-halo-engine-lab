#include "strix/exec/compiler.hpp"
#include "strix/graph/model_decl.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <stdexcept>
#include <vector>

void test_inkling_program_formats(const strix::graph::ModelDecl& decl) {
    namespace exec = strix::exec;
    namespace graph = strix::graph;

    // The compatibility graph describes pass zero. Repeat-only rows belong to the visit schedule.
    auto stages = decl.stages;
    std::vector<graph::StageDecl> active_rows;
    std::ranges::copy_if(stages.layer, std::back_inserter(active_rows), [&](const graph::StageDecl& stage) {
        return std::ranges::any_of(stages.layers, [&](graph::Traits traits) {
            return (traits & stage.when_all) == stage.when_all && (traits & stage.when_none) == 0;
        });
    });
    stages.layer = active_rows;
    graph::validate(stages);

    const auto plan = graph::launch_plan(stages, 0, 1, graph::FusionPolicy{});
    const exec::Compiler compiler(stages,
        [](const exec::ProgramKey&, const graph::Instance&) { return exec::BufferLayout{exec::BufferStorage::external}; },
        [](const exec::LaunchBindingRequest& request) {
            exec::KernelBinding binding;
            binding.fn = [](std::span<const std::byte>, const exec::Grid&, void*) {};
            binding.identity = request.launch.entry + "/host-fixture";
            return binding;
        });
    const auto program = compiler.compile({}, plan);
    std::array<unsigned, 2> gate_up{}, down{};
    for (const auto& launch : program.launches) {
        for (const auto& operand : launch.stages) {
            const auto& stage = program.stages.at(operand.instance);
            if (stage.id != "expert_gate_up" && stage.id != "expert_down") continue;
            const bool last_layers = launch.layer >= 20;
            const auto expected = last_layers ? graph::Format::mxfp8 : graph::Format::nvfp4;
            if (launch.layer != stage.layer || stage.format != expected)
                throw std::runtime_error("Inkling bound expert format does not match its layer");
            auto& counts = stage.id == "expert_gate_up" ? gate_up : down;
            ++counts[last_layers];
        }
    }
    if (gate_up != std::array{20U, 4U} || down != gate_up)
        throw std::runtime_error("Inkling must bind both expert projections for all NVFP4 and MXFP8 layers");
}
