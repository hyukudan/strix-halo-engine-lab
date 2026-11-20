#pragma once
// MoE expert-output combine: the shared contract (vendor-free; the HIP kernel is src/moe_core/route_wave_kernel.hpp).
//
// Inputs of one row r: the route's weights w[r][k] and expert outputs eo[r][k][width] in the route's OUTPUT order
// (include/strix/moe/route_policy.hpp: selection order or ascending id), and an optional base[r][width]. Declared order
// per element d (k ascending, every bit of every binding):
//   fma_f32 (GLM-5):      acc = +0; acc = fma(w_k, eo_k[d], acc); y[d] = acc + base[d]            (base required)
//   bf16_steps (a reference that accumulates experts with a BF16 index-add in ascending id, rounding every add):
//                         acc = +0; acc = bf(acc + bf(eo_k[d] * w_k)); y[d] = base ? bf(acc + base[d]) : acc
//                         w == nullptr: unit weights, term = eo_k[d] (no product): with top_k = 1 and BF16-valued eo
//                         this is the BF16 add y = bf(eo + base) (e.g. routed output + shared expert output).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

namespace strix::moe {

enum class CombineForm : std::uint8_t { fma_f32, bf16_steps };

// Round a float to the nearest bfloat16 (ties to even); NaN stays NaN (the bits of ds41::round_to_bf16).
inline float combine_round_bf16(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof bits);
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x7fffffU)) return x;
    bits = (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U;
    float y = 0.0F;
    std::memcpy(&y, &bits, sizeof y);
    return y;
}

// Host oracle of `rows` rows. w: [rows][top_k] or empty (bf16_steps unit weights); eo: [rows][top_k][width];
// base: [rows][width] or empty (fma_f32 requires it); y: [rows][width].
inline void combine_ref(CombineForm form, std::uint32_t rows, std::uint32_t top_k, std::uint32_t width, std::span<const float> w,
                        std::span<const float> eo, std::span<const float> base, std::span<float> y) {
    const std::size_t n = static_cast<std::size_t>(rows) * width;
    if (top_k == 0U || eo.size() < n * top_k || y.size() < n || (!w.empty() && w.size() < static_cast<std::size_t>(rows) * top_k) ||
        (!base.empty() && base.size() < n))
        throw std::invalid_argument("moe combine: spans are smaller than the shape");
    if (form == CombineForm::fma_f32 && (w.empty() || base.empty())) throw std::invalid_argument("moe combine: fma_f32 needs weights and base");
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t d = 0; d < width; ++d) {
            float acc = 0.0F;
            for (std::uint32_t k = 0; k < top_k; ++k) {
                const float e = eo[(static_cast<std::size_t>(r) * top_k + k) * width + d];
                if (form == CombineForm::fma_f32) {
                    acc = std::fma(w[static_cast<std::size_t>(r) * top_k + k], e, acc);
                } else {
                    const float term = w.empty() ? e : combine_round_bf16(e * w[static_cast<std::size_t>(r) * top_k + k]);
                    acc = combine_round_bf16(acc + term);
                }
            }
            const std::size_t at = static_cast<std::size_t>(r) * width + d;
            if (form == CombineForm::fma_f32) y[at] = acc + base[at];
            else y[at] = base.empty() ? acc : combine_round_bf16(acc + base[at]);
        }
}

}  // namespace strix::moe
