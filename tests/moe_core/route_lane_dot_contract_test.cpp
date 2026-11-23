// RouteAccMulAdd is two roundings whatever the host contraction flags are: this translation unit is compiled with the
// compiler's default contraction (GCC gnu++20: fast; Clang: on) and FMA enabled where the compiler supports it (CMake),
// the setting under which a literal acc + w * x contracts. It also pins route_project_rows_ref to validated spans: no bias span is formed over an operand
// the call did not validate (selection bias off, or an image-only row class).
#include "strix/moe/route_project.hpp"

#include <array>
#include <cmath>
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

struct StdMath {
    static float exp(float x) { return std::exp(x); }
    static float log1p(float x) { return std::log1p(x); }
    static float sqrt(float x) { return std::sqrt(x); }
};

volatile float g_a = 1.0F + 0x1p-12F;  // opaque to the optimiser: the step is evaluated at run time

void step_is_unfused() {
    const float a = g_a, minus_one = -1.0F;
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24: the product rounds to 1 + 2^-11 (tie to even); a fused step keeps 2^-24.
    check(RouteAccMulAdd::step(minus_one, a, a) == 0x1p-11F, "mul+add rounds the product under contraction");
    check(RouteAccFma::step(minus_one, a, a) == 0x1p-11F + 0x1p-24F, "fma keeps the low product bits");
    // The second case of the review: a lane whose first product is -1 then (1 + 2^-23) * (1 - 2^-23).
    const float u = 1.0F + 0x1p-23F, v = 1.0F - 0x1p-23F;
    volatile float vu = u, vv = v;
    check(RouteAccMulAdd::step(minus_one, vu, vv) == 0.0F, "mul+add: (1 + 2^-23)(1 - 2^-23) rounds to 1");
    check(RouteAccFma::step(minus_one, vu, vv) == -0x1p-46F, "fma: same operands keep -2^-46");
}

void lane_chain_is_unfused() {
    // Lane 0 folds element 0 (w = 1, x = -1) then element 32 (w = x = a): -1, then step(-1, a, a).
    std::vector<float> x(33U, 0.0F), w(33U, 0.0F);
    x[0] = -1.0F;
    w[0] = 1.0F;
    x[32] = w[32] = g_a;
    const float logit = RouteLaneDot<RouteAccMulAdd>::dot_ref(x, w);
    check(logit == 0x1p-11F, "dot_ref lane chain is unfused under contraction");
}

void bias_spans_are_validated_only() {
    // Two experts. Selection bias off with a one-element ordinary bias span and no image marks: valid, bias unused.
    RoutePolicy policy;
    policy.score = RouteScore::sigmoid;
    policy.experts = 2U;
    policy.top_k = 1U;
    policy.renormalize = false;
    const std::array<float, 2> x = {1.0F, 1.0F}, weight = {0.5F, -0.5F};  // hidden = 1; logits +0.5, -0.5
    std::array<float, 2> scores{};
    std::array<std::uint32_t, 1> ids{};
    std::array<float, 1> weights{};
    const float short_bias[1] = {100.0F};
    route_project_rows_ref<RouteLaneDot<RouteAccMulAdd, 1U>, StdMath>(
        policy, std::span<const float>(x.data(), 1U), 1U, 1U, weight, std::span<const float>(short_bias, 1U), {}, {},
        scores, ids, weights);
    check(ids[0] == 0U, "no selection bias: pick ignores the short bias span");

    // Selection bias on with image marks: row 0 selects with bias, row 1 (marked) with bias_image.
    policy.selection_bias = true;
    const std::array<float, 2> bias = {0.0F, 0.0F}, bias_image = {0.0F, 10.0F};
    const std::array<std::uint8_t, 2> marks = {0U, 1U};
    const std::array<float, 2> xs = {1.0F, 1.0F};
    std::array<float, 4> scores2{};
    std::array<std::uint32_t, 2> ids2{};
    std::array<float, 2> weights2{};
    route_project_rows_ref<RouteLaneDot<RouteAccMulAdd, 1U>, StdMath>(policy, xs, 2U, 1U, weight, bias, bias_image, marks,
                                                                     scores2, ids2, weights2);
    check(ids2[0] == 0U && ids2[1] == 1U, "image-marked row selects with bias_image");
}

}  // namespace

int main() {
    step_is_unfused();
    lane_chain_is_unfused();
    bias_spans_are_validated_only();
    if (g_failures != 0) return 1;
    std::printf("route lane dot contract: ok\n");
    return 0;
}
