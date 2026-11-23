#include "cuda/core/chain/launch.cuh"
#include "cuda/core/route_softmax.cuh"
#include "strix/cuda/amd_math_tables.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

namespace am = strix::cuda::amdmath;
namespace ch = strix::cuda::core::chain;
namespace moe = strix::moe;
constexpr float neg_inf = -std::numeric_limits<float>::infinity();
constexpr float qnan = std::numeric_limits<float>::quiet_NaN();

#include "f3g_route_frozen.inc"

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

constexpr unsigned route_rows = 64U;
constexpr unsigned experts = moe::kGfx1151RouteExperts;
constexpr unsigned max_slots = route_rows * moe::kGfx1151RouteMaxTopK;

struct RouteOutput {
    std::uint32_t ids[max_slots + 1U];
    float weights[max_slots + 1U];
};

struct RouteData {
    float logits[route_rows * experts];
    RouteOutput frozen_block;
    RouteOutput frozen_warp;
    RouteOutput shared;
    moe::RouteStatus status[route_rows + 1U];
};

void compare_route(const RouteOutput& actual, const RouteOutput& expected, unsigned k, bool normalize,
                   const char* arm) {
    for (unsigned i = 0; i < route_rows * k; ++i) {
        const auto weight = std::bit_cast<std::uint32_t>(actual.weights[i]);
        const auto expected_weight = std::bit_cast<std::uint32_t>(expected.weights[i]);
        if (actual.ids[i] == expected.ids[i] && weight == expected_weight) continue;
        std::fprintf(stderr, "%s row %u slot %u K=%u renorm=%d: got (%u,%08x), expected (%u,%08x)\n",
                     arm, i / k, i % k, k, static_cast<int>(normalize), actual.ids[i], weight,
                     expected.ids[i], expected_weight);
        throw std::runtime_error("route bits differ");
    }
}

void test_route(const am::Tables& tables) {
    RouteData* data = nullptr;
    check(cudaMallocManaged(&data, sizeof(RouteData)));
    std::mt19937 random(0xF351U);
    for (float& logit : data->logits) logit = static_cast<float>(static_cast<int>(random() % 4096U) - 2048) / 64.0F;
    const auto row_logits = [&](unsigned row) { return data->logits + row * experts; };
    std::fill_n(row_logits(0), experts, neg_inf);
    std::fill_n(row_logits(1), experts, qnan);
    row_logits(2)[319] = qnan;
    std::fill_n(row_logits(3), experts, neg_inf);
    constexpr std::uint32_t tied_ids[]{31U, 32U, 255U, 256U, 511U};
    for (const auto id : tied_ids) row_logits(3)[id] = 4.0F;
    std::fill_n(row_logits(4), experts, -0.0F);
    row_logits(4)[256] = 0.0F;
    row_logits(5)[511] = -neg_inf;
    std::fill_n(row_logits(6), experts, neg_inf);
    row_logits(6)[256] = 1.0F;

    for (unsigned k = 1U; k <= moe::kGfx1151RouteMaxTopK; ++k) {
        for (const bool normalize : {false, true}) {
            const moe::RoutePolicy policy{.score = moe::RouteScore::softmax, .experts = experts, .top_k = k,
                                         .renormalize = normalize, .renorm_eps = 0.0F,
                                         .math = moe::RouteMath::gfx1151_v1};
            const unsigned end = route_rows * k;
            data->shared.ids[end] = 0xfeedU;
            data->shared.weights[end] = 123.0F;
            std::fill_n(data->status, route_rows + 1U, static_cast<moe::RouteStatus>(17U));
            const float* logits = data->logits;
            frozen::route_select_kernel<<<route_rows, moe::kGfx1151RouteLanes>>>(
                logits, k, normalize, data->frozen_block.ids, data->frozen_block.weights, tables);
            check(cudaGetLastError());
            frozen::route_select_warp_kernel<<<route_rows, moe::kGfx1151RouteLanes>>>(
                logits, k, normalize, data->frozen_warp.ids, data->frozen_warp.weights, tables);
            check(cudaGetLastError());
            check(strix::cuda::core::launch_gfx1151_softmax_route(
                policy, logits, data->shared.ids, data->shared.weights, data->status, route_rows, tables, nullptr));
            check(cudaDeviceSynchronize());
            compare_route(data->shared, data->frozen_block, k, normalize, "frozen block");
            compare_route(data->shared, data->frozen_warp, k, normalize, "frozen warp");
            for (unsigned row = 0; row < route_rows; ++row) {
                const bool masked = row < 3U || row == 5U;
                require(data->status[row] == (masked ? moe::RouteStatus::invalid_route : moe::RouteStatus::ok),
                        "unexpected invalid_route status");
                if (masked) {
                    for (unsigned slot = 0; slot < k; ++slot) {
                        const auto i = row * k + slot;
                        require(data->shared.ids[i] == experts, "missing expert-count sentinel");
                        const float weight = data->shared.weights[i];
                        require(normalize ? std::isnan(weight) : weight == neg_inf, "unexpected empty-slot weight");
                    }
                }
            }
            for (unsigned slot = 0; slot < k; ++slot) {
                const auto sparse_id = slot < 5U ? tied_ids[slot] : slot - 5U;
                require(data->shared.ids[3U * k + slot] == sparse_id, "sparse tie ordering changed");
                require(data->shared.ids[4U * k + slot] == slot, "uniform tie ordering changed");
            }
            require(data->shared.ids[end] == 0xfeedU && data->shared.weights[end] == 123.0F &&
                    data->status[route_rows] == static_cast<moe::RouteStatus>(17U), "route wrote outside valid rows");
        }
    }
    check(cudaFree(data));
    std::puts("F3G route: 2048 row/case comparisons to both frozen selectors EXACT, invalid_route PASS");
}

using Folded = ch::List<ch::ActivationStep<ch::Form::gate_sigmoid_mul, 0, ch::Math::gfx1151_v1,
                                          ch::Fold::descending_upper_first>>;
using Side = ch::List<ch::ActivationStep<ch::Form::gate_sigmoid_mul, 0, ch::Math::gfx1151_v1,
                                        ch::Fold::descending_upper_first_side_output>>;
static_assert(ch::chain_scope<Side> == ch::Scope::row && ch::static_key<Side> != ch::static_key<Folded>);
constexpr unsigned chain_rows = 8U;
constexpr unsigned chain_width = 513U;
constexpr unsigned features = 769U;
constexpr unsigned pitch = features + 7U;
constexpr float canary = 123.0F;

struct ChainData {
    std::uint16_t bf16[chain_rows * pitch];
    float weights[features];
    float input[chain_rows * chain_width];
    float multiplied[chain_rows * chain_width];
    float fold_sigmoid[chain_rows + 2U];
    float side_sigmoid[chain_rows + 2U];
    ch::Gfx1151FoldedGate fold;
    ch::Gfx1151FoldedGate side;
};

void test_chain(const am::Tables& tables) {
    ChainData* data = nullptr;
    check(cudaMallocManaged(&data, sizeof(ChainData)));
    std::fill_n(data->input, chain_rows * chain_width, 0.25F);
    std::fill_n(data->fold_sigmoid, chain_rows + 2U, canary);
    std::fill_n(data->side_sigmoid, chain_rows + 2U, canary);
    for (unsigned i = 0; i < features; ++i)
        data->weights[i] = static_cast<float>(static_cast<int>(i % 17U) - 8) / 47.0F;
    data->weights[0] = 0x1p24F;
    data->weights[1] = 1.0F;
    data->weights[128] = -0x1p24F;
    constexpr std::uint16_t samples[]{0x3f80U, 0xbf80U, 0x3f01U, 0x3e88U, 0x0000U, 0x8000U};
    for (unsigned row = 0; row < chain_rows; ++row) {
        for (unsigned i = 0; i < pitch; ++i)
            data->bf16[row * pitch + i] = row == 0U
                ? ((i == 0U || i == 1U || i == 128U) ? std::uint16_t{0x3f80U} : std::uint16_t{0U})
                : samples[(i / 7U + row) % 6U];
    }
    data->fold = {data->bf16, data->weights, pitch, features, tables, data->fold_sigmoid + 1U};
    data->side = data->fold;
    data->side.sigmoid_out = data->side_sigmoid + 1U;
    ch::ChainArgs args{};
    args.aux[0] = &data->fold;
    check(ch::instance<3, Folded>.enqueue({data->input, data->multiplied, chain_rows, chain_width}, args, nullptr));
    args.aux[0] = &data->side;
    check(ch::instance<1, Side>.enqueue({nullptr, nullptr, chain_rows, 0U}, args, nullptr));
    check(cudaDeviceSynchronize());

    require(data->fold_sigmoid[1] > 0.7F && data->fold_sigmoid[1] < 0.8F,
            "cancellation row did not produce sigmoid(1)");
    for (unsigned i = 1; i <= chain_rows; ++i) {
        if (std::bit_cast<std::uint32_t>(data->side_sigmoid[i]) == std::bit_cast<std::uint32_t>(data->fold_sigmoid[i])) continue;
        std::fprintf(stderr, "chain sigmoid row %u differs\n", i - 1U);
        throw std::runtime_error("chain bits differ");
    }
    require(data->side_sigmoid[0] == canary && data->side_sigmoid[chain_rows + 1U] == canary,
            "chain wrote outside valid rows");
    check(cudaFree(data));
    std::puts("F3G chain: 8 raw row sigmoids vs folded entry EXACT, side-output bounds PASS");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: strix-f3g-cuda-fixture TABLES_DIR\n");
        return 2;
    }
    try {
        auto tables = am::load_tables(argv[1]);
        test_route(tables);
        test_chain(tables);
        am::free_tables(tables);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "F3G fixture: %s\n", error.what());
        return 1;
    }
}
