// Shared F3 route projection (include/strix/moe/route_project.hpp): the unfused lane chain RouteLaneDot<RouteAccMulAdd>
// with the per-row bias class reproduces the DS41 host router (ds41::reference_router: scores, ids and weight bits) on
// synthetic rows with image-marked rows, and the scalar form of the DS41 HIP score kernels (lane l: elements l, l + 32,
// ... ascending, acc = acc + w * x, butterfly 16 .. 1). RouteAccFma is a different class (one rounding per step).
#include "strix/ds41_moe.hpp"
#include "strix/moe/route_project.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

using namespace strix::moe;

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}
std::uint32_t bits(float v) {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, sizeof u);
    return u;
}

struct Ds41Math {
    static float exp(float x) { return strix::ds41::declared_exp(x); }
    static float log1p(float x) { return strix::ds41::declared_log1p(x); }
    static float sqrt(float x) { return strix::ds41::declared_sqrt(x); }
};

struct Lcg {
    std::uint64_t s;
    std::uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(s >> 33U);
    }
    float uniform(float scale) { return (static_cast<float>(next() % 65536U) / 32768.0F - 1.0F) * scale; }
};

// Scalar form of the DS41 HIP router score kernels (one wave, written lane by lane).
float wave_logit(const float* x, const float* w, std::size_t hidden) {
    std::array<float, 32> acc{};
    for (unsigned lane = 0; lane < 32U; ++lane)
        for (std::size_t i = lane; i < hidden; i += 32U) {
            const float p = w[i] * x[i];
            acc[lane] = acc[lane] + p;
        }
    for (unsigned d = 16U; d; d /= 2U) {
        std::array<float, 32> next{};
        for (unsigned l = 0; l < 32U; ++l) next[l] = acc[l] + acc[l ^ d];
        acc = next;
    }
    return acc[0];
}

void ds41_case(std::uint32_t rows, std::uint32_t hidden, std::uint32_t experts, std::uint32_t top_k, float scale,
               bool image, std::uint64_t seed) {
    Lcg g{seed};
    std::vector<float> x(static_cast<std::size_t>(rows) * hidden), w(static_cast<std::size_t>(experts) * hidden);
    std::vector<float> bias(experts), bias_image(experts);
    for (auto& v : x) v = g.uniform(2.0F);
    for (auto& v : w) v = g.uniform(0.25F);
    for (std::uint32_t e = 0; e < experts; ++e) {
        bias[e] = (e % 5U) == 0U ? 0.0F : g.uniform(0.5F);
        bias_image[e] = g.uniform(0.5F);
    }
    std::vector<std::uint8_t> image_rows;
    if (image)
        for (std::uint32_t r = 0; r < rows; ++r) image_rows.push_back(static_cast<std::uint8_t>(r % 3U == 1U));
    const auto ref = strix::ds41::reference_router(x, rows, hidden, w, experts, bias, bias_image, image_rows, top_k, scale);

    const RoutePolicy policy = preset::ds41(experts, top_k, scale);
    std::vector<float> scores(static_cast<std::size_t>(rows) * experts), weights(static_cast<std::size_t>(rows) * top_k);
    std::vector<std::uint32_t> ids(static_cast<std::size_t>(rows) * top_k);
    route_project_rows_ref<RouteLaneDot<RouteAccMulAdd>, Ds41Math>(policy, x, rows, hidden, w, bias, bias_image, image_rows,
                                                                   scores, ids, weights);
    bool same_scores = true, same_ids = ids == ref.ids, same_weights = true, same_wave = true;
    for (std::size_t i = 0; i < scores.size(); ++i) same_scores = same_scores && bits(scores[i]) == bits(ref.scores[i]);
    for (std::size_t i = 0; i < weights.size(); ++i) same_weights = same_weights && bits(weights[i]) == bits(ref.weights[i]);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t e = 0; e < experts; ++e) {
            const float a = wave_logit(x.data() + std::size_t{r} * hidden, w.data() + std::size_t{e} * hidden, hidden);
            const float b = RouteLaneDot<RouteAccMulAdd>::dot_ref(std::span<const float>(x).subspan(std::size_t{r} * hidden, hidden),
                                                                  std::span<const float>(w).subspan(std::size_t{e} * hidden, hidden));
            same_wave = same_wave && bits(a) == bits(b);
        }
    check(same_scores, "ds41 scores bits");
    check(same_ids, "ds41 ids");
    check(same_weights, "ds41 weight bits");
    check(same_wave, "ds41 HIP lane chain bits");
}

void row_bias_class() {
    const float b[2] = {1.0F, 2.0F}, bi[2] = {3.0F, 4.0F};
    const std::uint8_t marks[3] = {0U, 1U, 0U};
    const RouteRowBias plain{b, bi, nullptr}, marked{b, bi, marks};
    check(plain.row(1) == b, "null image rows select bias");
    check(marked.row(0) == b && marked.row(1) == bi && marked.row(2) == b, "image rows select bias_image");
}

void fma_is_a_different_class() {
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24: the unfused product rounds the 2^-24 tie away (to even), the fused step keeps it.
    const float a = 1.0F + 0x1p-12F;
    check(bits(RouteAccMulAdd::step(-1.0F, a, a)) != bits(RouteAccFma::step(-1.0F, a, a)), "mul+add differs from fma");
    check(RouteAccFma::step(-1.0F, a, a) == 0x1p-11F + 0x1p-24F, "fma keeps the low product bits");
    check(RouteAccMulAdd::step(-1.0F, a, a) == 0x1p-11F, "mul+add rounds the product");
}

}  // namespace

int main() {
    ds41_case(1U, 256U, 64U, 6U, 1.5F, false, 1U);
    ds41_case(4U, 4096U, 256U, 6U, 1.5F, true, 2U);
    ds41_case(8U, 1000U, 96U, 1U, 1.5F, true, 3U);   // single pick keeps its raw score; hidden not a multiple of 32
    ds41_case(3U, 512U, 384U, 8U, 2.0F, true, 4U);
    row_bias_class();
    fma_is_a_different_class();
    if (g_failures != 0) return 1;
    std::printf("route lane dot: ok\n");
    return 0;
}
