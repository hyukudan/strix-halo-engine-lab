#include "core/hc_core.hpp"
#include "glm/common/glm_math.hpp"
#include "strix/ds41_dspark.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
namespace hc = strix::core::hc;
namespace ds = strix::ds41::dspark;
namespace glm = strix::glm;

struct RawMath {
    static float rstd(float x, unsigned n, float eps) { return ds::rstd_of(x, n, eps); }
    static float sigmoid(float x) { return ds::sigmoid(x); }
    static float exp(float x) { return strix::ds41::declared_exp(x); }
};
struct ProjectedMath {
    static float sigmoid(float x) { return glm::sigmoid(x); }
    static float exp(float x) { return glm::declared_exp(x); }
};
using Raw = hc::PackedForm<RawMath, hc::PackedInput::raw_with_squares, ds::sinkhorn_iterations>;
using Projected = hc::PackedForm<ProjectedMath, hc::PackedInput::projected, glm::kHcIters>;

void same(float actual, float expected, const char* stage, unsigned index) {
    if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected)) {
        std::fprintf(stderr, "%s[%u]: %a != %a\n", stage, index, static_cast<double>(actual), static_cast<double>(expected));
        std::exit(1);
    }
}

// The original serial R6 computes all denominators before applying each
// normalization pass. The same denominators are gathered by the wave version.
void projected_reference(const float* z, const float* scale, const float* base, float* m) {
    for (unsigned j = 0; j < 4; ++j) m[j] = glm::sigmoid(z[j] * scale[0] + base[j]) + glm::kHcEps;
    for (unsigned j = 4; j < 8; ++j) m[j] = 2.0F * glm::sigmoid(z[j] * scale[1] + base[j]);
    float* c = m + 8;
    for (unsigned k = 0; k < 16; ++k) c[k] = z[8 + k] * scale[2] + base[8 + k];
    for (unsigned j = 0; j < 4; ++j) {
        float* row = c + j * 4;
        float mx = row[0];
        for (unsigned k = 1; k < 4; ++k) mx = row[k] > mx ? row[k] : mx;
        float e[4];
        for (unsigned k = 0; k < 4; ++k) e[k] = glm::declared_exp(row[k] - mx);
        const float sum = ((e[0] + e[1]) + e[2]) + e[3];
        for (unsigned k = 0; k < 4; ++k) row[k] = e[k] / sum + glm::kHcEps;
    }
    for (unsigned it = 0; it < glm::kHcIters; ++it) {
        if (it > 0) {
            float sr[4];
            for (unsigned j = 0; j < 4; ++j) {
                const float* row = c + j * 4;
                sr[j] = (((row[0] + row[1]) + row[2]) + row[3]) + glm::kHcEps;
            }
            for (unsigned j = 0; j < 4; ++j)
                for (unsigned k = 0; k < 4; ++k) c[j * 4 + k] = c[j * 4 + k] / sr[j];
        }
        float sc[4];
        for (unsigned k = 0; k < 4; ++k)
            sc[k] = (((c[k] + c[4 + k]) + c[8 + k]) + c[12 + k]) + glm::kHcEps;
        for (unsigned j = 0; j < 4; ++j)
            for (unsigned k = 0; k < 4; ++k) c[j * 4 + k] = c[j * 4 + k] / sc[k];
    }
}

template <bool Squares>
void check_projection() {
    constexpr auto waves = ds::stream_width / ds::sum_chunk;
    std::array<float, ds::stream_width> x{};
    std::array<std::uint16_t, ds::stream_width> fn{};
    for (unsigned i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>((13 * i) % 257) - 128) / 13.0F;
        fn[i] = static_cast<std::uint16_t>(0x2800U + (509 * i) % 0x1900U);
    }
    const auto decode = strix::ds41::f16_to_float;
    std::array<float, waves> chunks{};
    for (unsigned w = 0; w < waves; ++w) {
        std::array<float, 32> lanes{};
        const unsigned first = w * ds::sum_chunk;
        for (unsigned lane = 0; lane < lanes.size(); ++lane)
            lanes[lane] = hc::packed_lane_dot<Squares>(x.data(), fn.data(), first, first + ds::sum_chunk, lane, decode);
        for (unsigned offset = 16; offset > 0; offset /= 2) {
            const auto previous = lanes;
            for (unsigned lane = 0; lane < lanes.size(); ++lane) lanes[lane] = previous[lane] + previous[lane ^ offset];
        }
        chunks[w] = lanes[0];
    }
    const float expected = ds::chunk_sum(ds::stream_width, [&](unsigned i) {
        if constexpr (Squares) return x[i] * x[i];
        else return decode(fn[i]) * x[i];
    });
    same(hc::packed_chunk_total(chunks.data(), waves), expected, Squares ? "squares" : "projection", 0);
}
}  // namespace

int main() {
    check_projection<false>();
    check_projection<true>();
    constexpr std::array scale{0.25F, -0.5F, 1.75F};
    std::array<float, 25> raw{};
    std::array<float, 24> base{}, normalized{}, actual{}, expected{};
    for (unsigned t = 0; t < 64; ++t) {
        for (unsigned j = 0; j < 24; ++j) {
            raw[j] = static_cast<float>(static_cast<int>((17 * j + 37 * t) % 257) - 128) * 0.75F;
            base[j] = static_cast<float>(static_cast<int>((31 * j + 11 * t) % 103) - 51) * 0.125F;
        }
        raw[24] = t == 0 ? 0.0F : static_cast<float>(134 * t);
        const float rstd = ds::rstd_of(raw[24], ds::stream_width, ds::norm_eps);
        for (unsigned j = 0; j < 24; ++j) normalized[j] = raw[j] * rstd;
        ds::sinkhorn_core(normalized.data(), scale.data(), base.data(), expected.data());
        hc::mix_final_packed<Raw>(raw.data(), scale.data(), base.data(), ds::hc_eps, actual.data(), ds::stream_width, ds::norm_eps);
        for (unsigned j = 0; j < 24; ++j) same(actual[j], expected[j], "raw", j);

        projected_reference(raw.data(), scale.data(), base.data(), expected.data());
        hc::mix_final_packed<Projected>(raw.data(), scale.data(), base.data(), glm::kHcEps, actual.data());
        for (unsigned j = 0; j < 24; ++j) same(actual[j], expected[j], "projected", j);
    }

    constexpr float ones[4] = {1, 1, 1, 1};
    constexpr float x[8] = {0x1p25F, -0.0F, 1.0F, -0.0F, -0x1p25F, -0.0F, 1.0F, -0.0F};
    same(hc::packed_dot(ones, x, 0, 2), 1.0F, "collapse association", 0);
    same(hc::packed_dot(ones, x, 1, 2), -0.0F, "collapse leading term", 1);
    for (unsigned k = 0; k < 4; ++k) {
        const float residual = ((actual[8 + k] * x[0] + actual[12 + k] * x[2]) + actual[16 + k] * x[4]) + actual[20 + k] * x[6];
        same(hc::packed_expand(actual.data(), x, -2.0F, k, 0, 2), actual[4 + k] * -2.0F + residual, "expand", k);
    }
    unsigned waits = 0;
    const auto wait = [&] { ++waits; };
    if (hc::consume_peer(x, ones, wait) != x || waits != 0) return 1;
    if (hc::consume_peer(nullptr, ones, wait) != ones || waits != 1) return 1;
    same(hc::consume_rank_sum(nullptr, x, 1, 1), -0.0F, "folded exchange", 1);
    same(hc::consume_rank_sum(ones, x, 0, 3), ones[3] + x[3], "rank zero", 0);
    same(hc::consume_rank_sum(ones, x, 1, 3), x[3] + ones[3], "rank one", 1);
    std::puts("Packed HC scalar bit comparisons passed (projection, squares, 64 raw rows, 64 R6 rows, collapse, expand, exchange)");
}
