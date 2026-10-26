// GLM Tier A host emulation: semantic check against the float64 oracle on real-weight cases exported by
// tools/glm/export_host_case.py (mHC L3, KDA L4 rank 0 x 12 steps, MLA + indexer L3 rank 0 x 9 tokens, router L3),
// plus unit checks of the declared helpers (ordered keys, top-k ties, rollback truncation).
// Not a bitwise gate: fp32 host vs float64 oracle, max relative error per block must stay <= 2e-4.
// usage: glm_host_cpu_tests CASES_DIR
#include "glm/host/glm_host.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::glm;
using namespace strix::glm::host;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct CaseDir {
    std::string dir;
    std::map<std::string, std::vector<char>> raw;
    explicit CaseDir(std::string d) : dir(std::move(d)) {
        std::ifstream m(dir + "/manifest.txt");
        if (!m) throw std::runtime_error("no manifest in " + dir);
        std::string line;
        while (std::getline(m, line)) {
            std::istringstream ss(line);
            std::string name, dt;
            std::size_t count = 0;
            ss >> name >> dt >> count;
            const std::size_t es = dt == "u16" ? 2 : (dt == "f64" ? 8 : 4);
            std::vector<char> b(count * es);
            std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
            f.read(b.data(), static_cast<std::streamsize>(b.size()));
            if (static_cast<std::size_t>(f.gcount()) != b.size()) throw std::runtime_error("short " + name);
            raw.emplace(name, std::move(b));
        }
    }
    template <class T> const T* p(const std::string& n) const { return reinterpret_cast<const T*>(raw.at(n).data()); }
    template <class T> std::size_t n(const std::string& k) const { return raw.at(k).size() / sizeof(T); }
};

double rel(std::span<const float> a, const double* ref, std::size_t n) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) {
        num = std::max(num, std::fabs(static_cast<double>(a[i]) - ref[i]));
        den = std::max(den, std::fabs(ref[i]));
    }
    return den > 0 ? num / den : num;
}
std::string fmt(const char* what, double r) {
    char b[160];
    std::snprintf(b, sizeof b, "%s: max rel %.3e", what, r);
    return b;
}

void units() {
    check(ordered_key(1.0F) > ordered_key(0.5F) && ordered_key(0.0F) == ordered_key(-0.0F) &&
              ordered_key(-1.0F) < ordered_key(-0.5F) && ordered_key(-0.5F) < ordered_key(0.0F),
          "ordered_key monotone, -0 == +0");
    const std::vector<float> s{1.0F, 3.0F, 3.0F, -0.0F, 0.0F, 3.0F};
    const auto t = topk_ordered(s, 3);
    check(t == std::vector<std::uint32_t>({1, 2, 5}), "top-k ties -> lower index, output ascending");
    const auto t2 = topk_ordered(s, 5);
    check(t2 == std::vector<std::uint32_t>({0, 1, 2, 3, 5}), "top-k -0/+0 tie -> lower index");
    check(declared_exp(89.0F) == __builtin_huge_valf() && declared_exp(-104.0F) == 0.0F && sigmoid(-200.0F) == 0.0F &&
              sigmoid(200.0F) == 1.0F,
          "declared_exp / sigmoid saturation");
    DsaCache c;
    c.latent.assign(10 * kKvLora, 1);
    c.ikey.assign(10 * kIdxDim, 1);
    c.igate.assign(10 * kIdxDim, 1);
    c.pooled.assign(2 * kIdxDim, 0.0F);
    c.truncate(7);
    check(c.tokens() == 7 && c.pools() == 1 && c.latent.size() == 7 * kKvLora, "rollback truncation across a pool boundary");
}

void hc_case(const std::string& root) {
    CaseDir c(root + "/hc_l3");
    HcWeights w{Mat{c.p<float>("fn"), WType::F32, kHcMix, kHc * kHidden}, c.p<float>("base"), c.p<float>("scale")};
    std::span<const float> X(c.p<float>("X"), kHc * kHidden), y(c.p<float>("y"), kHidden);
    const HcMix m = hc_mix(X, w);
    check(rel(m.pre, c.p<double>("pre"), 4) <= 2e-4, fmt("mHC pre", rel(m.pre, c.p<double>("pre"), 4)));
    check(rel(m.post, c.p<double>("post"), 4) <= 2e-4, fmt("mHC post", rel(m.post, c.p<double>("post"), 4)));
    check(rel(m.comb, c.p<double>("comb"), 16) <= 2e-4, fmt("mHC comb (sinkhorn)", rel(m.comb, c.p<double>("comb"), 16)));
    std::vector<float> x(kHidden), Xo(kHc * kHidden);
    hc_collapse(X, m, x);
    hc_expand(X, m, y, Xo);
    check(rel(x, c.p<double>("collapse"), kHidden) <= 2e-4, fmt("mHC collapse", rel(x, c.p<double>("collapse"), kHidden)));
    check(rel(Xo, c.p<double>("expand"), Xo.size()) <= 2e-4, fmt("mHC expand", rel(Xo, c.p<double>("expand"), Xo.size())));
}

void kda_case(const std::string& root) {
    CaseDir c(root + "/kda_l4_r0");
    constexpr unsigned H = 32, C = H * kKdaDim, steps = 12;
    KdaState st{H, std::vector<float>(static_cast<std::size_t>(H) * kKdaDim * kKdaDim, 0.0F),
                std::vector<float>(static_cast<std::size_t>(3) * C * 3, 0.0F)};
    KdaWeights w{c.p<std::uint16_t>("conv"), c.p<float>("A_log"), c.p<float>("dt_bias"), Vec{c.p<std::uint16_t>("o_norm"), WType::BF16, kKdaDim}};
    double worst = 0;
    std::vector<float> y(C);
    for (unsigned t = 0; t < steps; ++t) {
        KdaInputs in{{c.p<float>("qkv") + static_cast<std::size_t>(t) * 3 * C, 3 * C}, {c.p<float>("f") + static_cast<std::size_t>(t) * C, C},
                     {c.p<float>("gate") + static_cast<std::size_t>(t) * C, C}, {c.p<float>("b") + static_cast<std::size_t>(t) * H, H}};
        kda_step(st, in, w, y);
        worst = std::max(worst, rel(y, c.p<double>("y") + static_cast<std::size_t>(t) * C, C));
    }
    check(worst <= 2e-4, fmt("KDA L4 rank 0, 12 steps: output", worst));
    const double rs = rel(st.S, c.p<double>("S_final"), st.S.size());
    check(rs <= 2e-4, fmt("KDA L4 rank 0: final recurrent state", rs));
}

void mla_case(const std::string& root) {
    CaseDir c(root + "/mla_l3_r0");
    constexpr unsigned H = 32, T = 9;
    IndexerWeights w{Mat{c.p<std::uint16_t>("indexer.wk"), WType::F16, kIdxDim, kHidden},
                     Mat{c.p<std::uint16_t>("indexer.index_kpool_compress_gate"), WType::F16, kIdxDim, kHidden},
                     Mat{c.p<std::uint16_t>("indexer.weights_proj"), WType::F16, kIdxHeads, kHidden},
                     Vec{c.p<std::uint16_t>("indexer.k_norm"), WType::F16, kIdxDim}, Vec{c.p<std::uint16_t>("indexer.k_norm.bias"), WType::F16, kIdxDim},
                     c.p<float>("indexer.index_kpool_compress_ape")};
    const Mat kvb{c.p<std::uint16_t>("kv_b_proj"), WType::F16, 2 * kMlaDim * H, kKvLora};  // rank 0 = first 32 heads' rows
    DsaCache cache;
    double worst = 0, worst_f32 = 0;
    std::vector<float> o(H * kMlaDim);
    for (unsigned t = 0; t < T; ++t) {
        std::span<const float> x(c.p<float>("x") + static_cast<std::size_t>(t) * kHidden, kHidden);
        dsa_append(cache, {c.p<float>("latent") + static_cast<std::size_t>(t) * kKvLora, kKvLora}, x, w);
        const auto tok = indexer_select(cache, t, {c.p<float>("qI") + static_cast<std::size_t>(t) * kIdxHeads * kIdxDim, kIdxHeads * kIdxDim}, x, w);
        check(tok.size() == t + 1, "MLA L3 token list at t=" + std::to_string(t) + " is dense causal (" + std::to_string(tok.size()) + ")");
        mla_attention(cache, tok, {c.p<float>("q") + static_cast<std::size_t>(t) * H * kMlaDim, H * kMlaDim}, kvb, H, o);
        worst = std::max(worst, rel(o, c.p<double>("o_bf16lat") + static_cast<std::size_t>(t) * H * kMlaDim, o.size()));
        worst_f32 = std::max(worst_f32, rel(o, c.p<double>("o") + static_cast<std::size_t>(t) * H * kMlaDim, o.size()));
    }
    check(worst <= 2e-4, fmt("MLA L3 rank 0 (absorbed) vs oracle on the same bf16-rounded latents, 9 tokens", worst));
    std::printf("INFO MLA L3: effect of the declared bf16 latent cache vs fp32 latents: max rel %.3e\n", worst_f32);
    check(cache.pools() == 2, "pooled cache holds 2 complete pools after 9 tokens");
}

void moe_case(const std::string& root) {
    CaseDir c(root + "/moe_l3");
    const Route r = route({c.p<float>("x"), kHidden}, Mat{c.p<std::uint16_t>("router"), WType::F16, kExperts, kHidden}, c.p<float>("bias"));
    bool same = true;
    for (unsigned k = 0; k < kTopE; ++k) same &= r.id[k] == c.p<std::uint32_t>("ids")[k];
    check(same, "router L3 top-8 ids == oracle");
    check(rel(r.w, c.p<double>("w"), kTopE) <= 2e-4, fmt("router L3 weights", rel(r.w, c.p<double>("w"), kTopE)));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s CASES_DIR\n", argv[0]);
        return 2;
    }
    units();
    hc_case(argv[1]);
    kda_case(argv[1]);
    mla_case(argv[1]);
    moe_case(argv[1]);
    std::printf("failures: %d\n", failures);
    return failures ? 1 : 0;
}
