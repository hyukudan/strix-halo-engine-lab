#include "b8_route_cases.hpp"
#include "moe_core/route_plan_r1.hpp"

#include <algorithm>
#include <bit>
#include <iostream>
#include <numeric>

namespace {

namespace b8 = strix::tests::b8;
namespace moe = strix::moe;
namespace r1 = strix::moe_core::route_plan_r1;
using r1::MatrixFormat;

void require(bool test, const char* message) {
    if (!test) throw std::runtime_error(message);
}

void geometry() {
    const auto baseline = r1::make_layer_plan(MatrixFormat::nvfp4, 288U, 144U);
    require(baseline && baseline->columns_gate_up == 1U && baseline->columns_down == 1U, "default geometry");
    const auto packed = r1::make_layer_plan(MatrixFormat::nvfp4, 288U, 144U, {4U, 2U});
    require(packed && packed->columns_gate_up == 4U && packed->columns_down == 2U, "admitted geometry");
    const auto small = r1::make_layer_plan(MatrixFormat::nvfp4, 288U, 16U, {4U, 4U});
    require(small && small->columns_gate_up == 2U && small->columns_down == 4U, "shape fallback");
    require(!r1::make_layer_plan(MatrixFormat::mxfp8, 288U, 144U, {4U, 2U}), "format alignment");
    const auto fp8 = r1::make_layer_plan(MatrixFormat::mxfp8, 288U, 160U, {2U, 4U});
    require(fp8 && fp8->format == MatrixFormat::mxfp8 && fp8->columns_gate_up == 2U &&
                fp8->columns_down == 4U, "different layer descriptor");
    require(!r1::make_layer_plan(static_cast<MatrixFormat>(-1), 288U, 160U), "invalid format");
    require(!r1::make_layer_plan(MatrixFormat::nvfp4, 0U, 144U), "empty width");
    require(!r1::make_layer_plan(MatrixFormat::nvfp4, 288U, 144U, {3U, 4U}), "invalid device limit");
    require(!r1::valid({MatrixFormat::nvfp4, 288U, 16U, 4U, 1U}), "invalid plan geometry");
    std::cout << "B8 host geometry and layer format: PASS\n";
}

void route(bool tied) {
    const auto args = b8::router_inputs(tied);
    const auto plan = r1::make_layer_plan(MatrixFormat::nvfp4, b8::hidden, 144U).value();
    const auto actual = r1::reference<b8::Policy, b8::Math>(plan, args.input, args.weights, args.bias);

    std::array<float, b8::experts> scores, choices;
    for (unsigned e = 0; e < b8::experts; ++e) {
        const auto row = std::span{args.weights}.subspan(std::size_t{e} * b8::hidden, b8::hidden);
        const auto pair = moe::route_project_sigmoid_ref<moe::RouteStridedFmaXor32, b8::Math>(
            args.input, row, args.bias[e]);
        scores[e] = pair.unbiased;
        choices[e] = pair.biased;
    }
    std::array<std::uint32_t, b8::top_k> ids, slots;
    std::array<float, b8::top_k> weights;
    moe::route_score_planes_ref<b8::Policy>(moe::RouteScorePlanes<1U>{scores.data(), choices.data()}, ids, weights);
    std::iota(slots.begin(), slots.end(), 0U);
    std::ranges::sort(slots, [&ids](auto a, auto b) { return ids[a] < ids[b]; });

    require(actual.ids == ids, "route IDs differ from separate projection and selection");
    for (unsigned rank = 0; rank < b8::top_k; ++rank) {
        const auto slot = slots[rank];
        require(std::bit_cast<std::uint32_t>(weights[slot]) == std::bit_cast<std::uint32_t>(actual.weights[slot]),
                "route weight bits differ");
        require(actual.group_by_slot[slot] == rank, "singleton group order differs");
    }
    if (tied) require(actual.ids == b8::tied_ids && actual.group_by_slot == b8::tied_groups, "ascending ties");
}

}  // namespace

int main() {
    try {
        geometry();
        route(true);
        route(false);
        std::cout << "B8 host route IDs, weights and compact groups: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
