// The dense flash with the relative bias projected inside it (DenseProjectedBias) against the same flash reading the
// [row][head][distance] image written by the producer (row_transforms::ShortBf16Projection, the stage it replaces):
// outputs bit-identical, and the same failure bit, including a non-finite logit only at a distance no score reads.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "attn_core/prepared_bf16_attention.hpp"
#include "core/row_transforms.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {
namespace dense = strix::attn_core::dense;
namespace prep = strix::core::row_transforms;
using strix::attn_core::RelativeBiasProjection;

constexpr unsigned dim = 128, group = 3, kv_heads = 2, heads = kv_heads * group, width = 16, extent = 1024;
constexpr unsigned rows = 200;

using ImageBase = dense::PreparedBf16Policy<dim, group, true, true, 1U, 64U>;
struct ImagePolicy : ImageBase {
    static constexpr float kScaleLog2 = 0.08838834764831845F * 1.4426950408889634F;
};
struct ProjectedPolicy
    : dense::PreparedBf16ProjectedBias<dense::PreparedBf16Policy<dim, group, false, true, 1U, 64U>, width, extent> {
    static constexpr float kScaleLog2 = ImagePolicy::kScaleLog2;
};
static_assert(dense::DenseProjectedBias<ProjectedPolicy> && !dense::DenseScoreBias<ProjectedPolicy>);
static_assert(dense::DenseScoreBias<ImagePolicy> && !dense::DenseProjectedBias<ImagePolicy>);

void require_hip(hipError_t result) {
    if (result != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(result) << '\n';
        std::exit(2);
    }
}

std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16U;
    x *= 0x7feb352dU;
    x ^= x >> 15U;
    x *= 0x846ca68bU;
    return x ^ (x >> 16U);
}
float sample(std::uint32_t seed) { return static_cast<float>(mix(seed) & 0xffffU) / 32768.0F - 1.0F; }
std::uint16_t bf16(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof bits);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

template <class T> T* managed(std::size_t count) {
    void* p = nullptr;
    require_hip(hipMallocManaged(&p, count * sizeof(T)));
    std::memset(p, 0, count * sizeof(T));
    return static_cast<T*>(p);
}

struct Case {
    const char* name;
    unsigned first;
    unsigned window;
    bool unused_infinite_weight;  // W[3][extent - 1] = inf: read by no score of these rows
    float big;                    // all features of one (row, head) set to this when nonzero
    bool flagged;                 // the producer's failure bit expected
};

bool run(const Case& c) {
    const unsigned slots = c.first + rows;
    auto* query = managed<float>(std::size_t{rows} * heads * dim);
    auto* out_image = managed<float>(std::size_t{rows} * heads * dim);
    auto* out_projected = managed<float>(std::size_t{rows} * heads * dim);
    auto* keys = managed<std::uint16_t>(std::size_t{slots} * kv_heads * dim);
    auto* values = managed<std::uint16_t>(std::size_t{slots} * kv_heads * dim);
    auto* zeros = managed<std::uint16_t>(dim);
    auto* features = managed<float>(std::size_t{rows} * heads * width);
    auto* weights = managed<std::uint16_t>(std::size_t{width} * extent);
    auto* logits = managed<float>(std::size_t{rows} * heads * extent);
    auto* sinks = managed<float>(std::size_t{rows} * heads);
    auto* failure = managed<unsigned>(3);

    for (std::size_t i = 0; i < std::size_t{rows} * heads * dim; ++i) query[i] = 2.0F * sample(static_cast<std::uint32_t>(i) ^ 0x51U);
    for (std::size_t i = 0; i < std::size_t{slots} * kv_heads * dim; ++i) {
        keys[i] = bf16(sample(static_cast<std::uint32_t>(i) ^ 0x1234U));
        values[i] = bf16(sample(static_cast<std::uint32_t>(i) ^ 0x8765U));
    }
    for (std::size_t i = 0; i < std::size_t{rows} * heads * width; ++i) features[i] = 3.0F * sample(static_cast<std::uint32_t>(i) ^ 0xabcU);
    for (std::size_t i = 0; i < std::size_t{width} * extent; ++i) weights[i] = bf16(0.5F * sample(static_cast<std::uint32_t>(i) ^ 0x77U));
    for (std::size_t i = 0; i < std::size_t{rows} * heads; ++i) sinks[i] = sample(static_cast<std::uint32_t>(i) ^ 0x99U);
    if (c.unused_infinite_weight) weights[3U * extent + extent - 1U] = 0x7f80U;
    if (c.big != 0.0F) {
        for (unsigned k = 0; k < width; ++k) features[(std::size_t{17} * heads + 4U) * width + k] = c.big;
    }

    // The stage the projection replaces: the image, its finite check into failure[0].
    using Wide = prep::ShortBf16Projection<width, true, 4U>;
    if (!Wide::fits(features, weights, logits, extent)) return false;
    require_hip(prep::elements(Wide::launch_size(std::size_t{rows} * heads, extent),
                               Wide{features, weights, logits, extent, failure}, nullptr));
    require_hip(hipDeviceSynchronize());
    const unsigned producer_failure = failure[0];

    const dense::PreparedBf16Args image{query, out_image, keys, values, zeros, failure + 1, rows, kv_heads, c.first,
                                        c.window, slots, false, {logits, heads, extent}, {sinks, heads}, {}};
    dense::PreparedBf16Args projected = image;
    projected.output = out_projected;
    projected.failure = failure + 2;
    projected.relative_projection = RelativeBiasProjection{features, weights, heads, extent};
    require_hip(dense::launch_prepared_bf16<ImagePolicy>(image, nullptr, {}, 1U));
    require_hip(dense::launch_prepared_bf16<ProjectedPolicy>(projected, nullptr, {}, 1U));
    require_hip(hipDeviceSynchronize());

    const bool equal = std::memcmp(out_image, out_projected, std::size_t{rows} * heads * dim * sizeof(float)) == 0;
    // Image path: the producer's bit plus the flash's own bits; projected path: the flash computes both.
    const unsigned image_bits = producer_failure | failure[1];
    const unsigned projected_bits = failure[2];
    const bool flag_expected = producer_failure == (c.flagged ? 1U : 0U);
    const bool ok = equal && image_bits == projected_bits && flag_expected;
    std::cout << c.name << ": outputs " << (equal ? "EQUAL" : "DIFF") << ", failure image " << image_bits
              << " projected " << projected_bits << (ok ? " ok" : " FAIL") << '\n';
    for (void* p : {static_cast<void*>(query), static_cast<void*>(out_image), static_cast<void*>(out_projected),
                    static_cast<void*>(keys), static_cast<void*>(values), static_cast<void*>(zeros),
                    static_cast<void*>(features), static_cast<void*>(weights), static_cast<void*>(logits),
                    static_cast<void*>(sinks), static_cast<void*>(failure)})
        require_hip(hipFree(p));
    return ok;
}

}  // namespace

int main() {
    const Case cases[] = {
        {"causal from 0", 0U, 0U, false, 0.0F, false},
        {"sliding 1024 from 1500", 1500U, 1024U, false, 0.0F, false},
        {"causal from 700 (past the extent)", 700U, 0U, false, 0.0F, false},
        {"unused infinite weight", 0U, 0U, true, 0.0F, true},
        // Bound 16 x 1.2e37 x max|w| (~0.5) > 2^126: every distance evaluated, all finite (|logit| < 1e38).
        {"large feature, exhaustive, finite", 0U, 0U, false, 1.2e37F, false},
        // 16 x 3e38 x ~0.5 overflows for the distances whose weights share a sign.
        {"huge feature, overflow", 0U, 0U, false, 3.0e38F, true},
    };
    bool ok = true;
    for (const auto& c : cases) ok &= run(c);
    if (!ok) {
        std::cerr << "projected relative bias differs from the image\n";
        return 1;
    }
    return 0;
}
