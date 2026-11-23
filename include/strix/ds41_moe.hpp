#pragma once

// deepseek41 routed-expert window: declared arithmetic of the fused gate/up SwiGLU, the down
// projection and the combine, the window routes in CSR form, and the host reference of the whole
// window. Device kernels in hip_ds41_formats.hpp reproduce it bitwise.
//
// Per (row, slot) the window keeps the unweighted down vector ("raw slot vector"); the combine
// applies the routing weights. A rank that computes a subset of the experts therefore produces the
// same raw vectors, and the combine is the same on one or two nodes:
//   y = (shared + low) + high, low = sum over the row's experts with id < split in ascending id of
//   w * raw, high likewise for ids >= split; each step is one FP32 product then one FP32 add.

#include "strix/ds41_formats.hpp"
#include "strix/ds41_msvc_builtins.hpp"
#include "strix/moe/activation_policy.hpp"
#include "strix/moe/declared_exp.hpp"
#include "strix/numeric/declared.hpp"

#include <cstdint>
#include <span>
#include <vector>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_DS41_HD __host__ __device__
#else
#define STRIX_DS41_HD
#endif

namespace strix::ds41 {

// Keep DS41's established API and declared arithmetic, now bound to its shared
// numeric home. No changes to constants, rounding or operation ordering.
namespace detail {
STRIX_DS41_HD inline float pow2(int k) { return ::strix::moe::declared_math::detail::pow2(k); }
} // namespace detail
using ::strix::numeric::declared_exp;
STRIX_DS41_HD inline float swiglu_clamped(float gate, float up, float limit) {
    return ::strix::moe::declared_math::swiglu_clamped(gate, up, limit);
}

// Natural logarithm of a positive normal float from basic IEEE operations: y = m * 2^k with m in
// [sqrt(1/2), sqrt(2)), ln m = 2 * atanh((m - 1) / (m + 1)) by its odd series to s^9, and
// k * ln 2 in two parts.
STRIX_DS41_HD inline float declared_log(float y) {
    const auto bits = __builtin_bit_cast(std::uint32_t, y);
    int k = static_cast<int>((bits >> 23U) & 0xffU) - 127;
    float m = __builtin_bit_cast(float, (bits & 0x7fffffU) | 0x3f800000U);
    if (m > 1.41421356F) { m = m * 0.5F; k = k + 1; }
    const float s = (m - 1.0F) / (m + 1.0F), s2 = s * s;
    float p = 0.111111111F;
    p = p * s2 + 0.142857143F;
    p = p * s2 + 0.2F;
    p = p * s2 + 0.333333333F;
    p = p * s2 + 1.0F;
    const float kf = static_cast<float>(k);
    return kf * 0.693359375F + (kf * -2.12194440e-4F + (2.0F * s) * p);
}
// log(1 + e) for e >= 0, accurate for small e: log(u) * e / (u - 1) with u = 1 + e.
STRIX_DS41_HD inline float declared_log1p(float e) {
    const float u = 1.0F + e;
    if (u == 1.0F) return e;
    return declared_log(u) * (e / (u - 1.0F));
}
using ::strix::numeric::declared_sqrt;
// Router score of a logit: sqrt(softplus(z)), softplus with the usual threshold (z > 20 -> z).
STRIX_DS41_HD inline float router_score(float z) {
    return declared_sqrt(z > 20.0F ? z : declared_log1p(declared_exp(z)));
}

// Router of one window. logits[r][e] = x[r] . weight[e] in the declared product order (lane l takes
// elements l, l + 32, ... in ascending order, each step acc = acc + w * x, then the butterfly);
// scores = router_score(logits); the top_k experts maximize score + bias (bias_image for rows marked
// as image tokens), larger first, the lower id first on ties; weights = score / (sum + 1e-20) * scale,
// the sum taken in selection order (no normalization when top_k == 1).
struct RouterOutput {
    std::vector<float> scores;          // [row][experts]
    std::vector<std::uint32_t> ids;     // [row][top_k]
    std::vector<float> weights;         // [row][top_k]
};
[[nodiscard]] RouterOutput reference_router(std::span<const float> x, std::uint32_t rows, std::uint32_t hidden,
                                            std::span<const float> weight, std::uint32_t experts, std::span<const float> bias,
                                            std::span<const float> bias_image, std::span<const std::uint8_t> image_rows,
                                            std::uint32_t top_k, float route_scale);

// Window routes over the union of the window's experts. Entry i of expert slot j (begins[j] <= i <
// begins[j + 1]) reads activation row inputs[i] and writes output slot outputs[i]. Experts are in
// ascending id and the entries of an expert in ascending row. Gate/up entries read the row and write
// the (row, k) slot = row * top_k + k; down entries read and write that slot.
struct WindowRoutes {
    std::vector<std::uint32_t> experts, begins, rows, slots;
    // Per row: its top_k slots in ascending expert id, and how many of them have id < split.
    std::vector<std::uint32_t> order, low_counts;
    std::uint32_t max_rows = 0;
};
// ids[row * top_k + k] is the k-th selected expert of the row (distinct within a row).
[[nodiscard]] WindowRoutes build_window_routes(std::span<const std::uint32_t> ids, std::uint32_t rows, std::uint32_t top_k,
                                               std::uint32_t split);
// The same routes with only the experts in [first, end): the work of a rank that owns them. The
// combine order and low counts are those of the whole window.
[[nodiscard]] WindowRoutes restrict_routes(const WindowRoutes& routes, std::uint32_t first, std::uint32_t end);

// Product of one weight row in any of the three formats with an A8 row (product_* above).
[[nodiscard]] float product(WeightFormat format, std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width);

struct ExpertTensors {
    WeightFormat gate_up_format = WeightFormat::iq2_xxs, down_format = WeightFormat::q2_k;
    std::span<const std::uint8_t> gate, up, down; // whole [experts] tensors
    std::uint32_t hidden = 0, inter = 0;
    float limit = 10.0F;
    [[nodiscard]] std::uint64_t gate_row_bytes() const noexcept {
        return hidden / block_elements(gate_up_format) * block_bytes(gate_up_format);
    }
    [[nodiscard]] std::uint64_t down_row_bytes() const noexcept {
        return inter / block_elements(down_format) * block_bytes(down_format);
    }
};
struct WindowReference {
    std::vector<float> gate, up;          // [slot][inter]
    std::vector<A8Activation> activation; // [slot], inter values each
    std::vector<float> raw;               // [slot][hidden], unweighted
    std::vector<float> y;                 // [row][hidden]
};
// Sum of one half of a row's weighted raw slot vectors (half 0: ids below split, half 1: the rest),
// in ascending id from 0: the low and high terms of the combine, which a rank owning that half can
// send instead of its raw vectors. canonical = (shared + low) + high.
[[nodiscard]] std::vector<float> reference_half_sum(const WindowRoutes& routes, std::span<const float> weights,
                                                    std::span<const float> raw, std::uint32_t rows, std::uint32_t top_k,
                                                    std::uint32_t hidden, unsigned half);

// Reference of one window: x[row] the A8 MoE inputs, weights[row * top_k + k] the routing weights of
// the slots, shared[row * hidden + i] the shared-expert output.
[[nodiscard]] WindowReference reference_window(const ExpertTensors& tensors, const std::vector<A8Activation>& x,
                                               const WindowRoutes& routes, std::span<const float> weights,
                                               std::span<const float> shared, std::uint32_t top_k);

} // namespace strix::ds41
