#pragma once

#include "strix/moe/route_policy.hpp"

#include <array>
#include <cmath>

namespace strix::tests::b8 {

inline constexpr unsigned hidden = 288U, experts = 19U, top_k = 9U;

struct Policy {
    static constexpr moe::RoutePolicy value{
        moe::RouteScore::sigmoid, true, moe::RouteKey::float_serial, moe::RouteOrder::selection,
        experts, top_k, 1U, 1U, true, true, 1e-20F, 2.5F};
};

struct Math {
#if defined(__CUDACC__)
    __host__ __device__
#endif
    static float exp(float x) { return ::expf(x); }
};

struct RouterInputs {
    std::array<float, hidden> input;
    std::array<float, experts * hidden> weights;
    std::array<float, experts> bias;
};

inline RouterInputs router_inputs(bool tied) {
    RouterInputs args;
    for (unsigned i = 0; i < hidden; ++i)
        args.input[i] = static_cast<float>(static_cast<int>(i % 31U) - 15) / 29.0F;
    for (unsigned e = 0; e < experts; ++e) {
        for (unsigned i = 0; i < hidden; ++i) {
            const auto matrix_row = tied ? 0U : e;
            args.weights[e * hidden + i] =
                static_cast<float>(static_cast<int>((matrix_row * 17U + i * 3U) % 19U) - 9) / 23.0F;
        }
        args.bias[e] = tied ? (e % 3U == 1U ? 2.0F : 0.0F)
                           : static_cast<float>(static_cast<int>((e * 5U) % 11U) - 5) / 37.0F;
    }
    return args;
}

inline constexpr std::array<std::uint32_t, top_k> tied_ids{1U, 4U, 7U, 10U, 13U, 16U, 0U, 2U, 3U};
inline constexpr std::array<std::uint32_t, top_k> tied_groups{1U, 4U, 5U, 6U, 7U, 8U, 0U, 2U, 3U};

}  // namespace strix::tests::b8
