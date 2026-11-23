#include "strix/receipt/step_profile_report.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

using namespace strix;
using namespace strix::exec;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

const serve::Json& field(const serve::Json& object, std::string_view name) {
    const auto* found = object.find(name);
    if (!found) throw std::runtime_error("missing JSON field " + std::string(name));
    return *found;
}

void write(const std::filesystem::path& path, const serve::Json& json) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << json.dump() << '\n';
    file.close();
    require(file.good(), "could not write profile fixture");
}

void run(const std::filesystem::path& directory) {
    const KernelEntry projection{
        .key = {graph::Op::linear, "expert.projection", {"synthetic", 1}, OperandClass::fp4_mma, RowsClass::r1},
        .arch = device::ArchMask::sm120, .identity = "expert-v1", .exact = lever::Exactness::bitwise,
        .cost = nullptr, .workspace = nullptr, .bind = nullptr, .launcher = nullptr,
    };
    Program program{};
    program.key.rows = 1;
    program.identity = 5;
    for (int layer = 0; layer < 3; ++layer) {
        auto& bound = program.launches.emplace_back();
        bound.layer = layer;
        bound.identity = layer < 2 ? "expert-v1" : "router-native";
        bound.entry = bound.identity;
        bound.grid = {{2, 3, 1}, {256, 1, 1}, 64};
        program.stages.push_back({"synthetic/" + std::to_string(layer), layer, graph::Format::none});
        bound.stages.push_back({static_cast<std::uint32_t>(layer), {}, 0});
    }

    constexpr double roof = 2;
    const std::array work{LaunchWork{3000, 2000, 8000, &projection},
        LaunchWork{3500, 500, 4000, &projection}, LaunchWork{}};
    const std::array mean_ms{0.004, 0.003, 0.001};
    const auto profile = make_step_profile(program, {work, roof, 4}, {mean_ms, 0.007, 0.002, 0, {}});
    const std::array costs{step_node_cost("moe_experts", {0, 3000, 2000, 4000, 1, 1000}),
        step_node_cost("moe_experts", {0, 3500, 500, 2000, 1, 2000}),
        step_node_cost("router", {0, 0, 0, 0, 1, 40})};
    StepTimingSeries samples(program.launches.size());
    const auto observer = samples.observer();
    observer.observe(observer.context, std::array{8.F, 4.F, 1.F}, 9, 25);
    observer.observe(observer.context, std::array{1.F, 4.F, 0.F}, 6, 15);
    observer.observe(observer.context, std::array{4.F, 2.F, 2.F}, 8, 20);
    observer.observe(observer.context, std::array{2.F, 2.F, 1.F}, 5, 20);
    receipt::Receipt receipt;
    receipt.identity.engine = "synthetic";
    receipt.resolution.emplace().tier = receipt::Tier::base;
    receipt.devices.emplace().emplace_back().bytes_per_second = roof * 1e9;

    constexpr std::string_view classes[]{"moe_experts", "router", "shared_dense"};
    const StepReportContext context{37, "plain", "clobbered", "catalogue", classes};
    const auto report = step_profile_report(program, profile, costs, samples, roof, context, receipt);
    require(field(report, "position").as_int() == 37 && field(report, "recipe").as_string() == "plain" &&
            field(report, "state_after_profile").as_string() == "clobbered" &&
            field(report, "dram_roof_source").as_string() == "catalogue", "diagnostic metadata");
    const auto& rows = field(report, "launches").as_array();
    require(rows.size() == 3, "launch count");
    const auto& first = rows.front();
    require(field(first, "family").as_string() == "expert.projection" &&
            field(field(first, "key"), "weight").as_string() == "synthetic", "catalogue identity");
    require(field(field(first, "grid"), "blocks").as_array()[1].as_int() == 3 &&
            field(field(first, "grid"), "shared_bytes").as_int() == 64, "launch dimensions");
    require(field(first, "gpu_us_median").as_double() == 3 && field(first, "gpu_us_min").as_double() == 1,
            "node median and minimum");
    require(field(first, "achieved_gbs").as_double() == 2 && field(first, "byte_roof_pct").as_double() == 100,
            "modelled bandwidth and roof percentage");
    require(field(first, "modelled_bytes").as_int() == 6000 && field(first, "modelled_flops").as_int() == 8000,
            "read plus write traffic and FLOPs");
    const auto& unknown = rows.back();
    require(field(unknown, "key").as_string() == "unknown" &&
            field(unknown, "modelled_bytes").as_string() == "unknown" &&
            field(unknown, "modelled_flops").as_string() == "unknown" &&
            field(unknown, "achieved_gbs").is_null(), "unknown catalogue and traffic are explicit");
    const auto& family = field(field(report, "totals_by_family"), "expert.projection");
    const auto& experts = field(field(report, "totals_by_layer_class"), "moe_experts");
    const auto& router = field(field(report, "totals_by_layer_class"), "router");
    const auto& absent = field(field(report, "totals_by_layer_class"), "shared_dense");
    require(family.dump() == experts.dump() && field(experts, "nodes").as_int() == 2 &&
            field(experts, "known_bytes_read").as_int() == 9000 && field(experts, "known_bytes").as_int() == 12000 &&
            field(experts, "known_flops").as_int() == 12000, "family and layer-class aggregation");
    require(field(router, "known_bytes_written").as_int() == 40 && field(router, "byte_priced_nodes").as_int() == 0 &&
            field(absent, "nodes").as_int() == 0,
            "partial cost tracked separately");
    const auto& comparison = field(report, "timing_comparison");
    require(field(comparison, "graph_wall_us_median").as_double() == 7 &&
            field(comparison, "eager_wall_minus_graph_wall_us").as_double() == 13 &&
            field(comparison, "summed_node_median_minus_graph_wall_us").as_double() == 0, "profiling overhead");

    std::filesystem::create_directories(directory);
    write(directory / "step-profile.json", report);
    write(directory / "receipt.json", field(report, "receipt"));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected fixture directory");
        run(argv[1]);
        std::cout << "x6_step_profile_host: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
