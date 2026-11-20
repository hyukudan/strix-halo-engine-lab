// Shared MoE router policy (include/strix/moe/route_policy.hpp): MODEL_B top-16/512 with bias, ties, NaN / -0 keys, and the
// family byte-compares: the shared oracle reproduces the ids and weight bits of each family's host router on synthetic
// logits (DS41 reference_router, GLM-5 route, MiniMax-M3 route_row), and each family's output digest is pinned so a
// re-bind of the family call site to the shared oracle is checked to keep its bits.
#include "strix/ds41_moe.hpp"
#include "strix/moe/activation_policy.hpp"
#include "strix/moe/route_policy.hpp"

#if defined(STRIX_RP_GLM5)
#include "glm/host/glm_host.hpp"
#endif
#if defined(STRIX_RP_MINIMAX)
#include "strix/minimax_moe.hpp"
#endif
#if defined(STRIX_RP_HY4)
#include "hy4/ref/hy4_host_ref.hpp"
#include "strix/hy4/presets.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using namespace strix::moe;
// Sigmoid router with ordered keys and ascending-id output, top-16 of 512, x 3.998 (a model binding keeps its own
// RoutePolicy in its model package).
constexpr RoutePolicy kSigmoid512{RouteScore::sigmoid, true, RouteKey::ordered_bits, RouteOrder::ascending_id, 512U, 16U, 1U, 1U,
                                  true, true, 1e-20F, 3.998F};

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
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

// Deterministic synthetic logits on a 1/64 grid (exact ties) with a few > 20 (softplus threshold), bias on a 1/256 grid
// with zeros (ties survive the bias).
struct Lcg {
    std::uint64_t s;
    std::uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(s >> 33U);
    }
};
std::vector<float> logits(Lcg& g, std::size_t n) {
    std::vector<float> z(n);
    for (auto& v : z) {
        const std::uint32_t r = g.next();
        v = (r % 97U) == 0U ? 20.0F + static_cast<float>(r % 7U) * 0.5F : static_cast<float>(static_cast<int>(r % 1025U) - 512) / 64.0F;
    }
    return z;
}
std::vector<float> biases(Lcg& g, std::size_t n) {
    std::vector<float> b(n);
    for (auto& v : b) {
        const std::uint32_t r = g.next();
        v = (r % 3U) == 0U ? 0.0F : static_cast<float>(static_cast<int>(r % 257U) - 128) / 256.0F;
    }
    return b;
}

struct Digest {
    std::uint64_t h = 1469598103934665603ULL;
    void add(std::uint32_t v) {
        for (unsigned i = 0; i < 4; ++i) {
            h ^= (v >> (8U * i)) & 0xffU;
            h *= 1099511628211ULL;
        }
    }
};

struct Row {
    std::vector<std::uint32_t> ids;
    std::vector<float> w;
};
Row oracle(const RoutePolicy& p, const std::vector<float>& s, const std::vector<float>& b) {
    Row r{std::vector<std::uint32_t>(p.top_k), std::vector<float>(p.top_k)};
    route_select_ref(p, s, b, r.ids, r.w);
    return r;
}
bool same(const Row& a, const std::uint32_t* ids, const float* w) {
    for (std::size_t k = 0; k < a.ids.size(); ++k)
        if (a.ids[k] != ids[k] || bits(a.w[k]) != bits(w[k])) return false;
    return true;
}

// Independent expectation for finite choices: stable sort by (choice desc, id asc).
std::vector<std::uint32_t> sorted_pick(const std::vector<float>& s, const std::vector<float>& b, std::uint32_t k) {
    std::vector<std::uint32_t> idx(s.size());
    std::iota(idx.begin(), idx.end(), 0U);
    std::stable_sort(idx.begin(), idx.end(), [&](std::uint32_t x, std::uint32_t y) { return s[x] + b[x] > s[y] + b[y]; });
    idx.resize(k);
    return idx;
}

void modelb_top16() {
    const RoutePolicy p = kSigmoid512;
    check(validate(p) == nullptr, "modelb preset validates");
    Lcg g{11};
    for (int row = 0; row < 64; ++row) {
        const auto z = logits(g, p.experts);
        const auto b = biases(g, p.experts);
        std::vector<float> s(p.experts);
        route_scores_ref<Ds41Math>(p.score, z, s);
        const Row r = oracle(p, s, b);
        auto want = sorted_pick(s, b, p.top_k);
        std::sort(want.begin(), want.end());
        check(r.ids == want, "modelb top-16 = (score + bias desc, id asc), output ascending id");
        float sum = 0.0F;
        for (const auto id : r.ids) sum = sum + s[id];
        bool w_ok = true, unbiased = true;
        for (std::uint32_t k = 0; k < p.top_k; ++k) {
            w_ok = w_ok && bits(r.w[k]) == bits((s[r.ids[k]] / (sum + 1e-20F)) * 3.998F);
            unbiased = unbiased && s[r.ids[k]] > 0.0F;
        }
        check(w_ok, "modelb weights = sigmoid / (sum + 1e-20) x 3.998 (bias for selection only)");
        check(unbiased, "modelb weights come from the unbiased sigmoid scores");
    }
}

void ties_and_keys() {
    // All choices equal: the lowest ids win, in ascending order.
    RoutePolicy p = kSigmoid512;
    std::vector<float> s(p.experts, 0.25F), b(p.experts, 0.0F);
    Row r = oracle(p, s, b);
    bool lowest = true;
    for (std::uint32_t k = 0; k < p.top_k; ++k) lowest = lowest && r.ids[k] == k;
    check(lowest, "ties -> lower id first");
    // The bias decides between equal scores; weights ignore it.
    b[300] = 0.5F;
    r = oracle(p, s, b);
    check(r.ids[0] == 0U && r.ids[p.top_k - 2U] == 14U && r.ids[p.top_k - 1U] == 300U, "bias picks 300 over id 15 (output ascending)");
    check(bits(r.w[0]) == bits(r.w[p.top_k - 1U]), "bias does not enter the weights");
    // -0 and +0 are one key (both keys); a NaN taken first stays under float_serial (DS41 / MiniMax serial scans).
    RoutePolicy q{RouteScore::sigmoid, false, RouteKey::ordered_bits, RouteOrder::selection, 4U, 2U, 1U, 1U, true, true, 1e-20F, 1.0F};
    const std::vector<float> zeros{-0.0F, 0.0F, -1.0F, -2.0F}, none;
    r = oracle(q, zeros, none);
    check(r.ids[0] == 0U && r.ids[1] == 1U, "ordered_bits: -0 == +0, lower id");
    q.key = RouteKey::float_serial;
    r = oracle(q, zeros, none);
    check(r.ids[0] == 0U && r.ids[1] == 1U, "float_serial: -0 == +0, lower id");
    const std::vector<float> nan_first{std::numeric_limits<float>::quiet_NaN(), 3.0F, 2.0F, 1.0F};
    r = oracle(q, nan_first, none);
    check(r.ids[0] == 0U && r.ids[1] == 1U, "float_serial: a first-candidate NaN is kept");
    // Ascending output order changes the sum order, not the picks.
    q.order = RouteOrder::ascending_id;
    const std::vector<float> desc{1.0F, 3.0F, 2.0F, 0.5F};
    r = oracle(q, desc, none);
    check(r.ids[0] == 1U && r.ids[1] == 2U, "ascending_id output");
    // Single pick without renormalisation (DS41) and the group guard.
    const RoutePolicy one = preset::ds41(4U, 1U, 2.0F);
    r = oracle(one, desc, std::vector<float>(4, 0.0F));
    check(r.ids[0] == 1U && bits(r.w[0]) == bits(6.0F), "ds41 top-1: raw score x scale");
    RoutePolicy grouped = kSigmoid512;
    grouped.n_group = 8U;
    check(validate(grouped) != nullptr, "group-limited routing is rejected");
    bool threw = false;
    try {
        (void)oracle(grouped, s, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "oracle throws on an unbindable policy");
    // Softmax transform: declared order, sums to ~1.
    std::vector<float> sm(4);
    route_scores_ref<Ds41Math>(RouteScore::softmax, desc, sm);
    check(sm[1] > sm[2] && sm[2] > sm[0] && std::fabs(sm[0] + sm[1] + sm[2] + sm[3] - 1.0F) < 1e-6F, "softmax transform");
}

void clamp_trait() {
    const float v[] = {-30.0F, -10.0F, -9.5F, -0.0F, 0.0F, 3.25F, 10.0F, 10.5F, 1e30F, std::numeric_limits<float>::quiet_NaN()};
    bool ds41_ok = true, none_ok = true;
    for (const float g : v)
        for (const float u : v) {
            const SwigluClamped c = swiglu_clamp(g, u, 10.0F);
            const float shared = (c.gate / (1.0F + strix::ds41::declared_exp(-c.gate))) * c.up;
            ds41_ok = ds41_ok && bits(shared) == bits(strix::ds41::swiglu_clamped(g, u, 10.0F));
            const SwigluClamped n = swiglu_clamp(g, u, kSwigluNoClamp);
            none_ok = none_ok && bits(n.gate) == bits(g) && bits(n.up) == bits(u);
#if defined(STRIX_RP_GLM5)
            const float glm = strix::glm::silu(c.gate) * c.up;
            ds41_ok = ds41_ok && bits(glm) == bits(strix::glm::host::swiglu(g, u));
#endif
        }
    check(ds41_ok, "clamp trait reproduces the family SwiGLU bits");
    check(none_ok, "clamp NONE (+inf) passes every input through");
}

void ds41_compare(Digest& d) {
    struct Case {
        std::uint32_t experts, top_k;
        float scale;
    };
    const Case cases[] = {{256U, 8U, 2.5F}, {384U, 6U, 1.5F}, {64U, 1U, 2.0F}, {512U, 16U, 3.998F}};
    Lcg g{41};
    for (const Case& c : cases) {
        const std::uint32_t rows = 16;
        std::vector<float> x(rows, 1.0F), w = logits(g, c.experts);
        // hidden = 1: logits[r][e] = 0 + w[e] * 1 exactly; per-row variety through the image bias rows.
        const auto b = biases(g, c.experts), bi = biases(g, c.experts);
        std::vector<std::uint8_t> image(rows);
        for (std::uint32_t r = 0; r < rows; ++r) image[r] = static_cast<std::uint8_t>(r % 2U);
        const auto out = strix::ds41::reference_router(x, rows, 1U, w, c.experts, b, bi, image, c.top_k, c.scale);
        const RoutePolicy p = preset::ds41(c.experts, c.top_k, c.scale);
        std::vector<float> s(c.experts);
        route_scores_ref<Ds41Math>(p.score, w, s);
        bool score_ok = true, sel_ok = true;
        for (std::uint32_t e = 0; e < c.experts; ++e) score_ok = score_ok && bits(s[e]) == bits(out.scores[e]);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const Row o = oracle(p, s, image[r] ? bi : b);
            sel_ok = sel_ok && same(o, out.ids.data() + std::size_t{r} * c.top_k, out.weights.data() + std::size_t{r} * c.top_k);
            for (std::uint32_t k = 0; k < c.top_k; ++k) {
                d.add(out.ids[std::size_t{r} * c.top_k + k]);
                d.add(bits(out.weights[std::size_t{r} * c.top_k + k]));
            }
        }
        check(score_ok, "ds41: sqrt_softplus<declared math> = router_score bits");
        check(sel_ok, "ds41: shared oracle = reference_router ids and weight bits");
    }
}

#if defined(STRIX_RP_MINIMAX)
void minimax_compare(Digest& d) {
    namespace mm = strix::minimax::moe;
    struct MmMath {
        static float exp(float x) { return strix::minimax::msa::exp_f(x); }
    };
    const RoutePolicy p = preset::minimax_m3;
    Lcg g{3};
    std::vector<float> y(mm::kHidden, 0.0F), w(std::size_t{mm::kExperts} * mm::kHidden, 0.0F);
    y[0] = 1.0F;
    bool ok = true;
    for (int row = 0; row < 32; ++row) {
        const auto z = logits(g, mm::kExperts);
        const auto b = biases(g, mm::kExperts);
        for (std::uint32_t e = 0; e < mm::kExperts; ++e) w[std::size_t{e} * mm::kHidden] = z[e];
        const mm::RouteRow ref = mm::route_row(y.data(), w.data(), b.data());
        std::vector<float> s(mm::kExperts);
        route_scores_ref<MmMath>(p.score, z, s);
        ok = ok && same(oracle(p, s, b), ref.ids, ref.weights);
        for (unsigned k = 0; k < mm::kTopE; ++k) {
            d.add(ref.ids[k]);
            d.add(bits(ref.weights[k]));
        }
    }
    check(ok, "minimax: shared oracle = route_row ids and weight bits");
}
#endif

#if defined(STRIX_RP_GLM5)
void glm_compare(Digest& d) {
    namespace gh = strix::glm::host;
    struct GlmMath {
        static float exp(float x) { return strix::glm::declared_exp(x); }
    };
    const RoutePolicy p = preset::glm5;
    Lcg g{5};
    const std::vector<float> x{1.0F};
    bool ok = true;
    for (int row = 0; row < 32; ++row) {
        const auto z = logits(g, p.experts);
        const auto b = biases(g, p.experts);
        const gh::Mat router{z.data(), gh::WType::F32, p.experts, 1U};
        const gh::Route ref = gh::route(x, router, b.data());
        std::vector<float> s(p.experts);
        route_scores_ref<GlmMath>(p.score, z, s);
        ok = ok && same(oracle(p, s, b), ref.id.data(), ref.w.data());
        for (unsigned k = 0; k < p.top_k; ++k) {
            d.add(ref.id[k]);
            d.add(bits(ref.w[k]));
        }
    }
    check(ok, "glm5: shared oracle = route ids and weight bits");
}
#endif


// Family output digests on the synthetic set, recorded on 015a043ab + 347072d22 BEFORE the call-site re-binds (WSL GCC 13,
// -ffp-contract=off): the re-bound family routers must keep them.
constexpr std::uint64_t kDs41Digest = 0x168f06b494ae3b13ULL, kMiniMaxDigest = 0xefdea130689467e7ULL, kGlmDigest = 0x8e7e4246cef34df4ULL;
void pin(const char* name, const Digest& d, std::uint64_t expected) {
    std::printf("%s digest %016llx\n", name, static_cast<unsigned long long>(d.h));
    if (expected != 0) check(d.h == expected, name);
}

}  // namespace

#if defined(STRIX_RP_HY4)
// strix::hy4::preset::route through the shared oracle == the Hy4-flash CPU oracle's router (src/hy4/ref route: stable sort on
// score + bias, ids ascending, ascending sum): ids and weight bits, including rows with many tied keys.
void hy4_oracle_compare() {
    const RoutePolicy p = strix::hy4::preset::route;
    Lcg g{17};
    bool ok = true;
    for (int row = 0; row < 64; ++row) {
        auto z = logits(g, p.experts);
        auto b = biases(g, p.experts);
        if (row % 4 == 0)
            for (std::size_t e = 0; e < z.size(); ++e) {  // coarse logits, no bias: ties at the cut
                z[e] = static_cast<float>(static_cast<int>(z[e] * 2.0F)) * 0.5F;
                b[e] = 0.0F;
            }
        std::vector<float> s(p.experts);
        route_scores_ref<Ds41Math>(p.score, z, s);
        const Row r = oracle(p, s, b);
        const auto ref = strix::hy4::route(z, b, p.top_k, p.scale);
        ok = ok && ref.ids.size() == p.top_k && same(r, ref.ids.data(), ref.weights.data());
    }
    check(ok, "hy4: shared oracle (strix::hy4::preset::route) = Hy4 CPU oracle route ids and weight bits");
}
#endif

int main() {
    modelb_top16();
#if defined(STRIX_RP_HY4)
    hy4_oracle_compare();
#endif
    ties_and_keys();
    clamp_trait();
    Digest ds41;
    ds41_compare(ds41);
    pin("ds41", ds41, kDs41Digest);
#if defined(STRIX_RP_MINIMAX)
    Digest mm;
    minimax_compare(mm);
    pin("minimax", mm, kMiniMaxDigest);
#endif
#if defined(STRIX_RP_GLM5)
    Digest glm;
    glm_compare(glm);
    pin("glm5", glm, kGlmDigest);
#endif
    std::printf("%s\n", g_failures == 0 ? "route_policy_tests PASS" : "route_policy_tests FAIL");
    return g_failures == 0 ? 0 : 1;
}
