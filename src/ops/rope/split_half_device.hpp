#pragma once

#include "core/pointwise.hpp"
#include "strix/weight/formats/scalar_float.hpp"

namespace strix::ops::rope {

// Split-half rotary on contiguous BF16-rounded heads, with positions generated
// from the absolute first row. Each thread owns both elements of one pair.
struct SplitHalfBf16 {
    float* rows;
    unsigned head_width;
    unsigned heads;
    unsigned first_position;
    float theta;

    __device__ void operator()(std::size_t i) const {
        const auto half = head_width / 2U;
        const auto head = i / half;
        const auto channel = static_cast<unsigned>(i % half);
        const auto position = first_position + static_cast<unsigned>(head / heads);
        const auto exponent = __fdiv_rn(static_cast<float>(channel), static_cast<float>(half));
        const auto angle = __fdiv_rn(static_cast<float>(position), powf(theta, exponent));
        float sine{}, cosine{};
        sincosf(angle, &sine, &cosine);
        auto* pair = rows + head * head_width + channel;
        const auto x = pair[0], y = pair[half];
        using weight::scalar_float::bf16_round;
        pair[0] = bf16_round(__fsub_rn(__fmul_rn(x, cosine), __fmul_rn(y, sine)));
        pair[half] = bf16_round(__fadd_rn(__fmul_rn(x, sine), __fmul_rn(y, cosine)));
    }
};

}  // namespace strix::ops::rope
