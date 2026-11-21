#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strix::cpu::detail {

// Each lane owns consecutive 16-element chunks spaced 32 chunks apart. SIMD may
// form the products in parallel. Every lane still adds them in input order, with
// zero padding in the final chunk, before the fixed 16, 8, 4, 2, 1 butterfly.
template<class FullProducts>
float chunk16_reduce(const float* weights, const float* input, std::uint32_t width,
                     FullProducts full_products) noexcept {
    float lanes[32] = {};
    const auto count = static_cast<std::size_t>(width);
    for (std::size_t offset = 0; offset < count;) {
        const auto live = std::min<std::size_t>(16, count - offset);
        float products[16] = {};
        if (live == 16) {
            full_products(weights + offset, input + offset, products);
        } else {
            for (std::size_t i = 0; i < live; ++i) products[i] = weights[offset + i] * input[offset + i];
        }
        auto& lane = lanes[(offset / 16) % 32];
        for (float product : products) lane = lane + product;
        offset += live;
    }
    for (unsigned d = 16; d != 0; d >>= 1U)
        for (unsigned l = 0; l < d; ++l) lanes[l] = lanes[l] + lanes[l + d];
    return lanes[0];
}

float chunk16_avx2(const float* weights, const float* input, std::uint32_t width) noexcept;
float chunk16_avx512(const float* weights, const float* input, std::uint32_t width) noexcept;

}  // namespace strix::cpu::detail
