#include "strix/moe/route_policy.hpp"
#include "strix/numeric/math_gfx1151.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>

namespace {

namespace moe = strix::moe;
namespace gfx = strix::numeric::gfx1151;
constexpr auto experts = moe::kGfx1151RouteExperts;
constexpr float neg_inf = -std::numeric_limits<float>::infinity();
constexpr float nan = std::numeric_limits<float>::quiet_NaN();

void require(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

struct PowersOfTwo {
    // Exact exponent values to isolate division and the addition tree from the math tables.
    static float exp(float z) { return std::exp2(z); }
};

constexpr moe::RoutePolicy make_policy(unsigned k, bool normalize) {
    return {.score = moe::RouteScore::softmax, .experts = experts, .top_k = k,
            .renormalize = normalize, .renorm_eps = 0.0F, .math = moe::RouteMath::gfx1151_v1};
}

void test_policy() {
    static_assert(moe::RoutePolicy{}.math == moe::RouteMath::native);
    // Model route presets live in the model TOMLs now; any policy that does not ask for gfx1151 math is native.
    static_assert(moe::RoutePolicy{.score = moe::RouteScore::sigmoid, .experts = 256, .top_k = 8}.math ==
                  moe::RouteMath::native);
    static_assert(!moe::validate(make_policy(1, true)) && !moe::validate(make_policy(16, false)));
    static_assert(moe::validate(make_policy(17, true)) != nullptr);
    auto p = make_policy(4, true);
    p.selection_bias = true;
    require(moe::validate(p), "gfx1151 accepted a biased route");
    p = make_policy(4, true);
    p.experts = 256;
    require(moe::validate(p), "gfx1151 accepted a different expert count");
    p = make_policy(1, true);
    p.renorm_single = false;
    require(moe::validate(p), "gfx1151 accepted skipping normalization for top-1");
}

void test_fold_and_order() {
    std::array<float, experts> logits;
    logits.fill(neg_inf);
    logits[0] = 0.0F;
    logits[256] = logits[128] = logits[384] = -24.0F;
    std::array<std::uint32_t, 5> ids{};
    std::array<float, 5> weights{};
    moe::route_rows_ref<PowersOfTwo>(make_policy(5, false), logits, {}, ids, weights);
    constexpr std::array<std::uint32_t, 5> want{0U, 128U, 256U, 384U, 1U};
    require(ids == want, "ties, zero-probability -inf logits, or selection order changed");
    require(std::bit_cast<std::uint32_t>(weights[0]) == 0x3f7ffffeU, "per-lane/halving fold or full division changed");
    require(std::bit_cast<std::uint32_t>(weights[1]) == 0x337ffffeU, "small full-softmax weight changed");
    require(std::bit_cast<std::uint32_t>(weights[4]) == 0U, "zero-probability weight changed");
    require(moe::route_status_from_last_id(ids.back(), experts) == moe::RouteStatus::ok, "finite row marked invalid");

    std::array<float, experts> scores;
    moe::route_gfx1151_scores_ref(logits, scores, PowersOfTwo::exp);
    float selected_sum = 0.0F;
    for (const auto id : want) selected_sum = selected_sum + scores[id];
    moe::route_rows_ref<PowersOfTwo>(make_policy(5, true), logits, {}, ids, weights);
    require(ids == want, "normalization changed expert slots");
    for (unsigned slot = 0; slot < ids.size(); ++slot)
        require(weights[slot] == scores[want[slot]] / selected_sum, "slot-order normalization changed");

    moe::route_rows_ref<PowersOfTwo>(make_policy(1, true), logits, {}, std::span{ids}.first(1), std::span{weights}.first(1));
    require(ids[0] == 0U && weights[0] == 1.0F, "top-1 normalization changed");

    // Adding three half-ULPs after the largest score rounds each away.
    // Summing them first (for example by ascending expert ID) changes the result.
    scores.fill(0.0F);
    scores[7] = 1.0F;
    scores[1] = scores[2] = scores[3] = 0x1p-24F;
    moe::route_select_ref(make_policy(4, true), scores, {}, std::span{ids}.first(4), std::span{weights}.first(4));
    require(ids[0] == 7U && ids[1] == 1U && weights[0] == 1.0F && weights[1] == 0x1p-24F,
            "slot-order renormalization changed the half-ULP fold");
}

void test_nonfinite() {
    std::array<float, 3U * experts> logits;
    logits.fill(neg_inf);
    logits[experts] = nan;
    logits[2U * experts] = 0.0F;
    logits[3U * experts - 1U] = 0.0F;

    const auto exp = [](float z) { return gfx::amd_expf({}, z); };  // only exp(0), exp(-inf), and NaN
    for (const bool normalize : {false, true}) {
        const auto p = make_policy(2, normalize);
        for (unsigned row = 0; row < 3U; ++row) {
            std::array<float, experts> scores;
            std::array<float, 2> weights;
            std::array<std::uint32_t, 2> ids;
            moe::route_gfx1151_scores_ref(std::span{logits}.subspan(row * experts).first<experts>(), scores, exp);
            moe::route_select_ref(p, scores, {}, ids, weights);
            if (row < 2U) {
                require(ids[0] == experts && ids[1] == experts, "all-masked row lost sentinel IDs");
                require(moe::route_status_from_last_id(ids.back(), experts) == moe::RouteStatus::invalid_route,
                        "all-masked row lost invalid_route");
                for (const float weight : weights)
                    require(normalize ? std::isnan(weight) : weight == neg_inf, "empty-slot weight changed");
            } else {
                require(ids[0] == 0U && ids[1] == experts - 1U && weights[0] == 0.5F && weights[1] == 0.5F,
                        "gfx1151 exp binding or tie order changed");
            }
        }
    }

    std::array<float, experts> scores;
    scores.fill(nan);
    scores[9] = 0.5F;
    std::array<float, 2> weights;
    std::array<std::uint32_t, 2> ids;
    moe::route_select_ref(make_policy(2, false), scores, {}, ids, weights);
    require(ids[0] == 9U && ids[1] == experts && weights[0] == 0.5F && weights[1] == neg_inf,
            "selector changed the partially masked probability contract");
}

}  // namespace

int main() {
    test_policy();
    test_fold_and_order();
    test_nonfinite();
    std::puts("F3G host: gfx1151 route policy, fold, slots and invalid_route PASS");
}
