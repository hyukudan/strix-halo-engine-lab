#pragma once

#include "moe_core/device_grouped_plan.hpp"

namespace strix::tests::microscaled_r1 {

namespace packed = weight::microscaled;

struct RoutedCase {
    const moe_core::device_grouped::ExpertWeights* matrices;
    const unsigned* ids;
    const float* weights;
    const unsigned* failure;
    const float* input;
    unsigned experts, hidden, intermediate;

    template <class Args>
    Args bind(std::uint16_t* activation, float* output) const {
        return {matrices, ids, weights, failure, input, activation, output, experts, hidden, intermediate, 3.0F, 2.0F};
    }
};

template <packed::MatrixFormat Format, unsigned TopK>
cudaError_t launch_reference(const RoutedCase& route, std::uint16_t* activation, float* output, cudaStream_t stream);

} // namespace strix::tests::microscaled_r1
