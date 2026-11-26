#pragma once

#include "strix/attn/group_pool.hpp"

#include <cmath>
#include <span>
#include <stdexcept>

namespace strix::attn {

// In-place FP32 reference for unweighted Q/K RMSNorm. Each head has its own reduction.
// Q and K can have different head counts or widths, and are passed independently.
inline void head_rms_reference(std::span<float> heads, unsigned head_width, float epsilon) {
    if (head_width == 0 || heads.empty() || heads.size() % head_width != 0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0F)
        throw std::invalid_argument("head RMSNorm: invalid shape or epsilon");
    for (std::size_t i = 0; i < heads.size(); i += head_width) {
        const float inv = group_pool::rms_inv(heads.data() + i, head_width, epsilon);
        for (float& value : heads.subspan(i, head_width)) value *= inv;
    }
}

}  // namespace strix::attn
