// Optimized GLM attention decode kernels (src/glm/attn) == Tier A host emulation bit for bit, and ==
// the batch-1/2 device kernels; then (bench mode) per-layer timing vs a bytes-read ceiling at 1K / 8K / 32K context.
// usage: hip_glm_attn_fast_tests CASES_DIR            gate (bitwise checks, exit 1 on any failure)
//        hip_glm_attn_fast_tests CASES_DIR --bench    gate, then the bench (BENCH lines)
#include "glm/attn/glm_attn_fast.hpp"
#include "glm/host/glm_host.hpp"
#include "glm/kernels/glm_dsa.hpp"
#include "glm/kernels/glm_tier_a.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::glm;
namespace H = strix::glm::host;
namespace D = strix::glm::dev;
namespace F = strix::glm::fast;

namespace {

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
template <class T> T* dalloc(std::size_t n) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    return p;
}
template <class T> T* zalloc(std::size_t n) {
    T* p = dalloc<T>(n);
    hok(hipMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "memset");
    return p;
}
template <class T> T* up(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    hok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
template <class T> void put(T* d, const std::vector<T>& h) { hok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d"); }
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    hok(hipDeviceSynchronize(), "sync");
    hok(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
template <class T> void fr(T* p) { hok(hipFree(p), "free"); }
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
bool same_n(const float* a, const float* b, std::size_t n) { return std::memcmp(a, b, n * 4) == 0; }

std::vector<float> rnd(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }
std::vector<std::uint16_t> rnd_f16(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = to_f16(d(g));
    return v;
}
std::vector<std::uint16_t> rnd_bf16(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = float_to_bf16(d(g));
    return v;
}
unsigned* new_flag() { return zalloc<unsigned>(1); }
unsigned read_flag(const unsigned* f) { return down(f, 1)[0]; }

struct CaseDir {
    std::map<std::string, std::vector<char>> raw;
    explicit CaseDir(const std::string& dir) {
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
    template <class T> std::vector<T> v(const std::string& n) const {
        const auto& b = raw.at(n);
        std::vector<T> out(b.size() / sizeof(T));
        std::memcpy(out.data(), b.data(), b.size());
        return out;
    }
};

// ================================================================= gate
void test_f16_decode() {
    float* d = zalloc<float>(65536);
    F::f16_table(d, nullptr);
    const auto got = down(d, 65536);
    bool all = true;
    for (std::uint32_t i = 0; i < 65536; ++i) {
        const float ref = f16_to_float(static_cast<std::uint16_t>(i));
        const float host = F::f16_bits_to_float(i);
        all &= std::memcmp(&ref, &got[i], 4) == 0 && std::memcmp(&ref, &host, 4) == 0;
    }
    check(all, "branch-free F16 decode == f16_to_float on all 65,536 bit patterns (device and host)");
    fr(d);
}

void test_dense(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/mla_l3_r0");
    const auto wk = c.v<std::uint16_t>("indexer.wk"), gate = c.v<std::uint16_t>("indexer.index_kpool_compress_gate");
    const auto wp = c.v<std::uint16_t>("indexer.weights_proj");
    const auto fb = rnd_f16(static_cast<std::size_t>(4096) * 128, g, 0.08F), gb = rnd_f16(static_cast<std::size_t>(4096) * 128, g, 0.08F);
    std::uint16_t *dwk = up(wk), *dgate = up(gate), *dwp = up(wp), *dfb = up(fb), *dgb = up(gb);
    bool all = true, old = true;
    for (unsigned R = 1; R <= 8; ++R) {
        const auto x = rnd(static_cast<std::size_t>(R) * kHidden, g, 0.5F);
        const auto a = rnd(static_cast<std::size_t>(R) * 128, g, 1.0F), bvec = rnd(static_cast<std::size_t>(R) * 128, g, 1.0F);
        float *dx = up(x), *da = up(a), *db = up(bvec);
        // outputs poisoned with NaN, one extra row that must stay NaN
        const std::vector<float> nan_k(static_cast<std::size_t>(R + 1) * 128, std::nanf("")), nan_w(static_cast<std::size_t>(R + 1) * 32, std::nanf(""));
        const std::vector<float> nan_b(static_cast<std::size_t>(R + 1) * 4096, std::nanf(""));
        float *yk = up(nan_k), *yg = up(nan_k), *yw = up(nan_w), *yfb = up(nan_b), *ygb = up(nan_b);
        const F::DenseJob j3[3] = {{dwk, dx, yk, 128, kHidden, kHidden, 128}, {dgate, dx, yg, 128, kHidden, kHidden, 128},
                                   {dwp, dx, yw, 32, kHidden, kHidden, 32}};
        F::dense_f16(j3, 3, R, nullptr);
        const F::DenseJob j2[2] = {{dfb, da, yfb, 4096, 128, 128, 4096}, {dgb, db, ygb, 4096, 128, 128, 4096}};
        F::dense_f16(j2, 2, R, nullptr);
        const auto hk = down(yk, nan_k.size()), hg = down(yg, nan_k.size()), hw = down(yw, nan_w.size());
        const auto hfb = down(yfb, nan_b.size()), hgb = down(ygb, nan_b.size());
        for (unsigned r = 0; r < R; ++r) {
            const std::span<const float> xr{x.data() + static_cast<std::size_t>(r) * kHidden, kHidden};
            std::vector<float> rk(128), rg(128), rw(32), rfb(4096), rgb(4096);
            H::dense(xr, H::Mat{wk.data(), H::WType::F16, 128, kHidden}, rk);
            H::dense(xr, H::Mat{gate.data(), H::WType::F16, 128, kHidden}, rg);
            H::dense(xr, H::Mat{wp.data(), H::WType::F16, 32, kHidden}, rw);
            H::dense({a.data() + r * 128, 128}, H::Mat{fb.data(), H::WType::F16, 4096, 128}, rfb);
            H::dense({bvec.data() + r * 128, 128}, H::Mat{gb.data(), H::WType::F16, 4096, 128}, rgb);
            all &= same_n(rk.data(), hk.data() + r * 128, 128) && same_n(rg.data(), hg.data() + r * 128, 128) &&
                   same_n(rw.data(), hw.data() + r * 32, 32) && same_n(rfb.data(), hfb.data() + r * 4096, 4096) &&
                   same_n(rgb.data(), hgb.data() + r * 4096, 4096);
        }
        for (std::size_t i = static_cast<std::size_t>(R) * 128; i < nan_k.size(); ++i) all &= std::isnan(hk[i]) && std::isnan(hg[i]);
        for (std::size_t i = static_cast<std::size_t>(R) * 4096; i < nan_b.size(); ++i) all &= std::isnan(hfb[i]) && std::isnan(hgb[i]);
        for (std::size_t i = static_cast<std::size_t>(R) * 32; i < nan_w.size(); ++i) all &= std::isnan(hw[i]);
        // the batch-1 dense kernel on the same inputs
        float* yo = zalloc<float>(static_cast<std::size_t>(R) * 128);
        D::dense(dx, dwk, D::WType::F16, kHidden, 0, 128, yo, R, nullptr);
        const auto ho = down(yo, static_cast<std::size_t>(R) * 128);
        old &= same_n(ho.data(), hk.data(), static_cast<std::size_t>(R) * 128);
        for (float* p : {dx, da, db, yk, yg, yw, yfb, ygb, yo}) fr(p);
    }
    check(all, "R1 dense_f16 fused jobs (real L3 wk + gate + weights_proj K=4096; f_b/g_b-shaped K=128 x 4096) == host, R=1..8, extra row untouched");
    check(old, "R1 dense_f16 == batch-1 dense kernel (wk, R=1..8)");
    // both schedules (staged = default for in >= 512, lane) on the K=4096 jobs and on the grouped mla.out shape
    // (32 groups = heads, n 256, in 512, real L3 kv_b V rows), R = 1..8, vs host
    {
        CaseDir c2(cases + "/mla_l3_r0");
        const auto kvb = c2.v<std::uint16_t>("kv_b_proj");
        std::uint16_t* dkvb = up(kvb);
        bool sched = true;
        for (unsigned R = 1; R <= 8; ++R) {
            const auto x = rnd(static_cast<std::size_t>(R) * kHidden, g, 0.5F);
            const auto ob = rnd(static_cast<std::size_t>(R) * 32 * kKvLora, g, 0.3F);
            float *dx = up(x), *dob = up(ob);
            std::vector<std::vector<float>> got(2);
            std::vector<std::vector<float>> got_o(2);
            for (int sc = 0; sc < 2; ++sc) {
                F::set_dense_schedule(sc == 0 ? F::DenseSchedule::kStaged : F::DenseSchedule::kLane);
                float* yk = up(std::vector<float>(static_cast<std::size_t>(R) * 288, std::nanf("")));
                const F::DenseJob j3[3] = {{dwk, dx, yk, 128, kHidden, kHidden, 288}, {dgate, dx, yk + 128, 128, kHidden, kHidden, 288},
                                           {dwp, dx, yk + 256, 32, kHidden, kHidden, 288}};
                F::dense_f16(j3, 3, R, nullptr);
                got[sc] = down(yk, static_cast<std::size_t>(R) * 288);
                float* yo = up(std::vector<float>(static_cast<std::size_t>(R) * 32 * kMlaDim, std::nanf("")));
                F::mla_out(dob, dkvb, 32, yo, R, nullptr);
                got_o[sc] = down(yo, static_cast<std::size_t>(R) * 32 * kMlaDim);
                fr(yk);
                fr(yo);
            }
            F::set_dense_schedule(F::DenseSchedule::kStaged);
            sched &= same(got[0], got[1]) && same(got_o[0], got_o[1]);
            for (unsigned r = 0; r < R; ++r) {
                const std::span<const float> xr{x.data() + static_cast<std::size_t>(r) * kHidden, kHidden};
                std::vector<float> rk(128), rg(128), rw(32);
                H::dense(xr, H::Mat{wk.data(), H::WType::F16, 128, kHidden}, rk);
                H::dense(xr, H::Mat{gate.data(), H::WType::F16, 128, kHidden}, rg);
                H::dense(xr, H::Mat{wp.data(), H::WType::F16, 32, kHidden}, rw);
                sched &= same_n(rk.data(), got[0].data() + r * 288, 128) && same_n(rg.data(), got[0].data() + r * 288 + 128, 128) &&
                         same_n(rw.data(), got[0].data() + r * 288 + 256, 32);
                for (unsigned h = 0; h < 32; ++h)
                    for (unsigned d = 0; d < kMlaDim; ++d) {
                        const float v = chain_dot([&](unsigned l) { return ob[(static_cast<std::size_t>(r) * 32 + h) * kKvLora + l]; },
                                                  [&](unsigned l) { return f16_to_float(kvb[static_cast<std::size_t>(h * 2 * kMlaDim + kMlaDim + d) * kKvLora + l]); }, kKvLora);
                        sched &= std::memcmp(&v, &got_o[0][(static_cast<std::size_t>(r) * 32 + h) * kMlaDim + d], 4) == 0;
                    }
            }
            fr(dx);
            fr(dob);
        }
        check(sched, "R1 dense schedules: staged (LDS tiles) == lane == host on wk|gate|wproj (K=4096, one 288-wide output) and on the grouped mla.out (32 heads x 256 x 512, real L3 kv_b), R=1..8");
        fr(dkvb);
    }
    for (std::uint16_t* p : {dwk, dgate, dwp, dfb, dgb}) fr(p);
}

struct KdaCase {
    std::vector<std::uint16_t> conv, onorm;
    std::vector<float> A_log, dt, qkv, f, gate, b;
};
void kda_inputs(const KdaCase& kc, unsigned t, unsigned R, std::mt19937& g, std::vector<float>& iq, std::vector<float>& ifv,
                std::vector<float>& ig, std::vector<float>& ib) {
    constexpr unsigned Hh = 32, C = Hh * kKdaDim;
    iq.assign(static_cast<std::size_t>(R) * 3 * C, 0.0F);
    ifv.assign(static_cast<std::size_t>(R) * C, 0.0F);
    ig.assign(ifv.size(), 0.0F);
    ib.assign(R * Hh, 0.0F);
    for (unsigned r = 0; r < R; ++r) {
        if (r == 0 && t < 12) {
            std::copy_n(kc.qkv.data() + static_cast<std::size_t>(t) * 3 * C, 3 * C, iq.begin());
            std::copy_n(kc.f.data() + static_cast<std::size_t>(t) * C, C, ifv.begin());
            std::copy_n(kc.gate.data() + static_cast<std::size_t>(t) * C, C, ig.begin());
            std::copy_n(kc.b.data() + static_cast<std::size_t>(t) * Hh, Hh, ib.begin());
        } else {
            const auto a = rnd(3 * C, g, 1.0F), bb = rnd(C, g, 1.0F), cc = rnd(C, g, 1.0F), dd = rnd(Hh, g, 1.0F);
            std::copy(a.begin(), a.end(), iq.begin() + static_cast<std::size_t>(r) * 3 * C);
            std::copy(bb.begin(), bb.end(), ifv.begin() + static_cast<std::size_t>(r) * C);
            std::copy(cc.begin(), cc.end(), ig.begin() + static_cast<std::size_t>(r) * C);
            std::copy(dd.begin(), dd.end(), ib.begin() + r * Hh);
        }
    }
}

void test_kda(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/kda_l4_r0");
    constexpr unsigned Hh = 32, C = Hh * kKdaDim;
    KdaCase kc{c.v<std::uint16_t>("conv"), c.v<std::uint16_t>("o_norm"), c.v<float>("A_log"), c.v<float>("dt_bias"),
               c.v<float>("qkv"),          c.v<float>("f"),               c.v<float>("gate"),  c.v<float>("b")};
    if (kc.conv.size() != static_cast<std::size_t>(3) * C * kConvK || kc.onorm.size() != kKdaDim || kc.A_log.size() != Hh ||
        kc.dt.size() != C || kc.qkv.size() != static_cast<std::size_t>(12) * 3 * C || kc.f.size() != static_cast<std::size_t>(12) * C ||
        kc.gate.size() != static_cast<std::size_t>(12) * C || kc.b.size() != static_cast<std::size_t>(12) * Hh) {
        check(false, "kda_l4_r0 fixture shapes");
        return;
    }
    const D::KdaDev kd{Hh, up(kc.conv), up(kc.A_log), up(kc.dt), up(kc.onorm)};
    const H::KdaWeights hw{kc.conv.data(), kc.A_log.data(), kc.dt.data(), H::Vec{kc.onorm.data(), H::WType::BF16, kKdaDim}};
    const std::size_t SZ = static_cast<std::size_t>(Hh) * kKdaDim * kKdaDim, CZ = static_cast<std::size_t>(3) * C * 3;
    // (a) rows = independent slots, R = 1..8, 64 steps. Row 0 starts at zero and runs the 12 real steps first; rows
    //     1..R-1 start from random non-zero states. One extra slot R (state, conv, output) and one extra input row are
    //     NaN-poisoned and must stay bit-identical. vs host and vs the batch-1 kernel (own copy of the state).
    for (unsigned R = 1; R <= 8; ++R) {
        std::vector<H::KdaState> hs;
        std::vector<float> S0(static_cast<std::size_t>(R + 1) * SZ, std::nanf("")), c0(static_cast<std::size_t>(R + 1) * CZ, std::nanf(""));
        for (unsigned r = 0; r < R; ++r) {
            H::KdaState st{Hh, r == 0 ? std::vector<float>(SZ, 0.0F) : rnd(SZ, g, 0.05F), r == 0 ? std::vector<float>(CZ, 0.0F) : rnd(CZ, g, 0.5F)};
            std::copy(st.S.begin(), st.S.end(), S0.begin() + r * SZ);
            std::copy(st.conv.begin(), st.conv.end(), c0.begin() + r * CZ);
            hs.push_back(std::move(st));
        }
        float *dS = up(S0), *dconv = up(c0), *oS = up(S0), *oconv = up(c0);
        float *dscr = zalloc<float>(static_cast<std::size_t>(R) * Hh * 4 * kKdaDim), *dbeta = zalloc<float>(R * Hh);
        const std::vector<float> ynan(static_cast<std::size_t>(R + 1) * C, std::nanf(""));
        float *dO = zalloc<float>(static_cast<std::size_t>(R) * C), *dY = up(ynan), *oY = zalloc<float>(static_cast<std::size_t>(R) * C);
        unsigned *flag = new_flag(), *oflag = new_flag();
        bool host_ok = true, old_ok = true, pad_ok = true;
        for (unsigned t = 0; t < 64; ++t) {
            std::vector<float> iq, ifv, ig, ib;
            kda_inputs(kc, t, R + 1, g, iq, ifv, ig, ib);
            std::fill(iq.begin() + static_cast<std::size_t>(R) * 3 * C, iq.end(), std::nanf(""));  // poisoned padding row
            std::fill(ifv.begin() + static_cast<std::size_t>(R) * C, ifv.end(), std::nanf(""));
            std::fill(ig.begin() + static_cast<std::size_t>(R) * C, ig.end(), std::nanf(""));
            std::fill(ib.begin() + R * Hh, ib.end(), std::nanf(""));
            float *dq = up(iq), *df = up(ifv), *dg = up(ig), *db = up(ib);
            F::kda_step(kd, dq, df, dg, db, dS, dconv, dY, flag, R, nullptr);
            D::kda_step(kd, dq, df, dg, db, oS, oconv, dscr, dbeta, dO, oY, oflag, R, nullptr);
            const auto yd = down(dY, static_cast<std::size_t>(R + 1) * C), yo = down(oY, static_cast<std::size_t>(R) * C);
            old_ok &= same_n(yd.data(), yo.data(), static_cast<std::size_t>(R) * C);
            pad_ok &= same_n(yd.data() + static_cast<std::size_t>(R) * C, ynan.data(), C);
            for (unsigned r = 0; r < R; ++r) {
                std::vector<float> yh(C);
                H::kda_step(hs[r], H::KdaInputs{{iq.data() + static_cast<std::size_t>(r) * 3 * C, 3 * C}, {ifv.data() + static_cast<std::size_t>(r) * C, C},
                                                {ig.data() + static_cast<std::size_t>(r) * C, C}, {ib.data() + r * Hh, Hh}},
                            hw, yh);
                host_ok &= same_n(yh.data(), yd.data() + static_cast<std::size_t>(r) * C, C);
            }
            for (float* p : {dq, df, dg, db}) fr(p);
        }
        const auto Sd = down(dS, (R + 1) * SZ), cd = down(dconv, (R + 1) * CZ);
        for (unsigned r = 0; r < R; ++r)
            host_ok &= same_n(hs[r].S.data(), Sd.data() + r * SZ, SZ) && same_n(hs[r].conv.data(), cd.data() + r * CZ, CZ);
        pad_ok &= same_n(Sd.data() + R * SZ, S0.data() + R * SZ, SZ) && same_n(cd.data() + R * CZ, c0.data() + R * CZ, CZ);
        old_ok &= same_n(Sd.data(), down(oS, R * SZ).data(), R * SZ) && same_n(cd.data(), down(oconv, R * CZ).data(), R * CZ);
        host_ok &= read_flag(flag) == 0;
        check(host_ok && pad_ok, "R8-R10 fused KDA step == host, R=" + std::to_string(R) +
                                     ": real L4 rank-0 weights, row 0 = 12 real steps from zero, other rows random non-zero start, 64 steps; outputs every step, S, conv; poisoned padding slot/row untouched; no flag");
        check(old_ok, "R8-R10 fused KDA step == batch-1 kda_step (3 kernels), R=" + std::to_string(R) + ", 64 steps");
        for (float* p : {dS, dconv, oS, oconv, dscr, dbeta, dO, dY, oY}) fr(p);
        fr(flag);
        fr(oflag);
    }
    // (b) sequential positions of one slot (speculative verify): npos = 1..8 from a non-zero state, with snapshots.
    //     Then rollback: restore the snapshot after position a (S + conv), re-advance one position with the rows kernel
    //     and compare with a host run that never speculated past a.
    bool seq_ok = true, rb_ok = true;
    for (unsigned npos = 1; npos <= 8; ++npos) {
        H::KdaState hs{Hh, rnd(SZ, g, 0.05F), rnd(CZ, g, 0.5F)};
        float *dS = up(hs.S), *dconv = up(hs.conv), *snap = zalloc<float>(npos * SZ), *csnap = zalloc<float>(npos * CZ);
        float* dY = zalloc<float>(static_cast<std::size_t>(npos) * C);
        unsigned* flag = new_flag();
        std::vector<float> iq, ifv, ig, ib;
        kda_inputs(kc, 99, npos, g, iq, ifv, ig, ib);  // npos random positions (rows = positions)
        float *dq = up(iq), *df = up(ifv), *dg = up(ig), *db = up(ib);
        F::kda_steps_seq(kd, dq, df, dg, db, dS, dconv, dY, snap, csnap, flag, npos, nullptr);
        const auto yd = down(dY, static_cast<std::size_t>(npos) * C), sn = down(snap, npos * SZ), cn = down(csnap, npos * CZ);
        std::vector<H::KdaState> after;
        for (unsigned p = 0; p < npos; ++p) {
            std::vector<float> yh(C);
            H::kda_step(hs, H::KdaInputs{{iq.data() + static_cast<std::size_t>(p) * 3 * C, 3 * C}, {ifv.data() + static_cast<std::size_t>(p) * C, C},
                                         {ig.data() + static_cast<std::size_t>(p) * C, C}, {ib.data() + p * Hh, Hh}},
                        hw, yh);
            after.push_back(hs);
            seq_ok &= same_n(yh.data(), yd.data() + static_cast<std::size_t>(p) * C, C);
            seq_ok &= same_n(hs.S.data(), sn.data() + p * SZ, SZ) && same_n(hs.conv.data(), cn.data() + p * CZ, CZ);
        }
        seq_ok &= same(hs.S, down(dS, SZ)) && same(hs.conv, down(dconv, CZ)) && read_flag(flag) == 0;
        // rollback to a = npos / 2 (accept positions 0..a), re-advance with a fresh input
        const unsigned a = npos / 2;
        hok(hipMemcpy(dS, snap + a * SZ, SZ * 4, hipMemcpyDeviceToDevice), "rb");
        hok(hipMemcpy(dconv, csnap + a * CZ, CZ * 4, hipMemcpyDeviceToDevice), "rb");
        std::vector<float> jq, jf, jg, jb;
        kda_inputs(kc, 99, 1, g, jq, jf, jg, jb);
        float *eq = up(jq), *ef = up(jf), *eg = up(jg), *eb = up(jb);
        F::kda_step(kd, eq, ef, eg, eb, dS, dconv, dY, flag, 1, nullptr);
        H::KdaState ref = after[a];
        std::vector<float> yh(C);
        H::kda_step(ref, H::KdaInputs{{jq.data(), 3 * C}, {jf.data(), C}, {jg.data(), C}, {jb.data(), Hh}}, hw, yh);
        rb_ok &= same_n(yh.data(), down(dY, C).data(), C) && same(ref.S, down(dS, SZ)) && same(ref.conv, down(dconv, CZ));
        for (float* p : {dS, dconv, snap, csnap, dY, dq, df, dg, db, eq, ef, eg, eb}) fr(p);
        fr(flag);
    }
    check(seq_ok, "R8-R10 KDA sequential positions of one slot (npos 1..8, non-zero state) == host step by step: outputs, per-position S + conv snapshots, final state");
    check(rb_ok, "KDA rollback: restore the snapshot after position npos/2, re-advance one step == host run that never speculated (npos 1..8)");
    // (c) non-finite state sets the flag (C2)
    {
        float *dS = up(std::vector<float>(SZ, 3.0e38F)), *dconv = zalloc<float>(CZ), *dY = zalloc<float>(C);
        unsigned* flag = new_flag();
        std::vector<float> iq, ifv, ig, ib;
        kda_inputs(kc, 0, 1, g, iq, ifv, ig, ib);
        for (auto& v : iq) v = 50.0F;  // large k, v: the update overflows
        for (auto& v : ifv) v = -50.0F;
        float *dq = up(iq), *df = up(ifv), *dg = up(ig), *db = up(ib);
        F::kda_step(kd, dq, df, dg, db, dS, dconv, dY, flag, 1, nullptr);
        H::KdaState hs{Hh, std::vector<float>(SZ, 3.0e38F), std::vector<float>(CZ, 0.0F)};
        bool host_threw = false;
        std::vector<float> yh(C);
        try {
            H::kda_step(hs, H::KdaInputs{{iq.data(), 3 * C}, {ifv.data(), C}, {ig.data(), C}, {ib.data(), Hh}}, hw, yh);
        } catch (const H::NonFinite&) {
            host_threw = true;
        }
        check(read_flag(flag) == 1 && host_threw, "C2 non-finite KDA state: device flag set, host rejects");
        for (float* p : {dS, dconv, dY, dq, df, dg, db}) fr(p);
        fr(flag);
    }
}

// ---- DSA: one host cache per slot, device slots, R rows (slot, t) per call
struct DevSlot {
    D::DsaSlot s{};
    void alloc(unsigned cap) {
        s = D::DsaSlot{zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kKvLora), zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim),
                       zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim), zalloc<float>(static_cast<std::size_t>(cap / kKpool) * kIdxDim), cap};
    }
    void release() {
        fr(s.latent);
        fr(s.ikey);
        fr(s.igate);
        fr(s.pooled);
    }
};
struct IdxW {
    std::vector<std::uint16_t> wk, gate, wp, knw, knb, kvb;
    std::vector<float> ape;
    H::IndexerWeights hw() const {
        return {H::Mat{wk.data(), H::WType::F16, kIdxDim, kHidden}, H::Mat{gate.data(), H::WType::F16, kIdxDim, kHidden},
                H::Mat{wp.data(), H::WType::F16, kIdxHeads, kHidden}, H::Vec{knw.data(), H::WType::F16, kIdxDim},
                H::Vec{knb.data(), H::WType::F16, kIdxDim}, ape.data()};
    }
};
IdxW load_idx(const std::string& cases) {
    CaseDir c(cases + "/mla_l3_r0");
    IdxW w{c.v<std::uint16_t>("indexer.wk"), c.v<std::uint16_t>("indexer.index_kpool_compress_gate"), c.v<std::uint16_t>("indexer.weights_proj"),
           c.v<std::uint16_t>("indexer.k_norm"), c.v<std::uint16_t>("indexer.k_norm.bias"), {}, c.v<float>("indexer.index_kpool_compress_ape")};
    const auto all = c.v<std::uint16_t>("kv_b_proj");
    w.kvb.assign(all.begin(), all.begin() + static_cast<std::size_t>(32) * 2 * kMlaDim * kKvLora);  // rank 0: heads 0..31
    return w;
}

// Runs the fast DSA query path for R rows: dense (wk|gate|wp) is NOT run here (append already done); wp output given.
struct DsaBufs {
    unsigned R, key_stride;
    D::DsaSlot* dslots;
    unsigned* dt;
    float *qI, *wraw, *q, *o;
    std::uint32_t *keys, *hist0, *tok, *n;
    F::MlaWork mw;
    void alloc(unsigned rows, unsigned kstride) {
        R = rows;
        key_stride = kstride;
        dslots = dalloc<D::DsaSlot>(rows);
        dt = zalloc<unsigned>(rows);
        qI = zalloc<float>(static_cast<std::size_t>(rows) * kIdxHeads * kIdxDim);
        wraw = zalloc<float>(static_cast<std::size_t>(rows) * kIdxHeads);
        q = zalloc<float>(static_cast<std::size_t>(rows) * 64 * kMlaDim);
        o = zalloc<float>(static_cast<std::size_t>(rows) * 64 * kMlaDim);
        keys = zalloc<std::uint32_t>(static_cast<std::size_t>(rows) * kstride);
        hist0 = zalloc<std::uint32_t>(static_cast<std::size_t>(rows) * 256);
        tok = zalloc<std::uint32_t>(static_cast<std::size_t>(rows) * F::kTokStride);
        n = zalloc<std::uint32_t>(rows);
        mw = F::MlaWork{zalloc<float>(static_cast<std::size_t>(rows) * 64 * kKvLora), zalloc<std::uint32_t>(static_cast<std::size_t>(rows) * 64),
                        zalloc<float>(static_cast<std::size_t>(rows) * 64 * F::kTokStride), zalloc<float>(static_cast<std::size_t>(rows) * 64 * F::kTokStride),
                        zalloc<float>(static_cast<std::size_t>(rows) * 64 * kKvLora)};
    }
};

// Host R11 score of pool p (the host indexer_select loop, kept here to check the device keys directly).
float host_score(const H::DsaCache& c, unsigned p, const float* qI, const float* wh) {
    const float* K = &c.pooled[static_cast<std::size_t>(p) * kIdxDim];
    float acc = 0.0F;
    for (unsigned h = 0; h < kIdxHeads; ++h) {
        const float dot = chain_dot([&](unsigned i) { return qI[h * kIdxDim + i]; }, [&](unsigned i) { return K[i]; }, kIdxDim);
        const float sc = dot * kInvSqrt128;
        acc = __builtin_fmaf(wh[h], sc > 0.0F ? sc : 0.0F, acc);
    }
    return acc;
}

// Runs the query path on rows already in b (slots, t, qI, q, wraw uploaded) and checks everything against the host:
// keys + top-byte histogram, token lists, absorbed q (row 0), MLA outputs. Returns false on any mismatch.
struct QueryCheck {
    bool keys = true, lists = true, qt = true, outs = true, pad = true;
};
// The padding row R (one past the call's rows) of every input, output and workspace buffer is filled with 0xFF bytes
// (NaN / 0xffffffff) before the call; afterwards every byte must still be 0xFF.
struct PadRegion {
    void* p;
    std::size_t bytes;
};
template <class T> PadRegion pad_at(T* base, std::size_t elems_per_row, unsigned r) {
    return PadRegion{base + elems_per_row * r, elems_per_row * sizeof(T)};
}
std::vector<PadRegion> pad_regions(DsaBufs& b, unsigned R) {
    constexpr std::size_t Hh = 32;
    return {pad_at(b.qI, kIdxHeads * kIdxDim, R), pad_at(b.wraw, kIdxHeads, R),           pad_at(b.q, Hh * kMlaDim, R),
            pad_at(b.o, Hh * kMlaDim, R),         pad_at(b.keys, b.key_stride, R),        pad_at(b.hist0, 256, R),
            pad_at(b.tok, F::kTokStride, R),      pad_at(b.n, 1, R),                      pad_at(b.mw.qt, Hh * kKvLora, R),
            pad_at(b.mw.mkey, Hh, R),             pad_at(b.mw.sc, Hh * F::kTokStride, R), pad_at(b.mw.p, Hh * F::kTokStride, R),
            pad_at(b.mw.ob, Hh * kKvLora, R)};
}
void run_and_check(DsaBufs& b, unsigned R, const std::vector<const H::DsaCache*>& hcs, const std::vector<unsigned>& tt,
                   const std::vector<float>& qI, const std::vector<float>& q, const std::vector<float>& wraw,
                   const IdxW& w, std::uint16_t* dkvb, unsigned p_bound, QueryCheck& qc) {
    constexpr unsigned Hh = 32;
    const F::DsaRows rows{b.dslots, b.dt, R};
    if (R + 1 > b.R) throw std::logic_error("run_and_check: buffers need a padding row");
    const auto pads = pad_regions(b, R);
    for (const auto& pr : pads) hok(hipMemset(pr.p, 0xff, pr.bytes), "pad");
    std::vector<std::uint32_t> ref_keys, ref_hist;
    for (unsigned rg : {1U, 2U, 3U, 4U, 5U}) {
        F::set_indexer_score_rows(rg);
        hok(hipMemset(b.keys, 0xa5, static_cast<std::size_t>(R) * b.key_stride * sizeof(std::uint32_t)), "score keys");
        hok(hipMemset(b.hist0, 0, static_cast<std::size_t>(R) * 256 * sizeof(std::uint32_t)), "score hist");
        F::indexer_score(rows, b.qI, b.wraw, b.keys, b.key_stride, b.hist0, p_bound, nullptr);
        const auto keys = down(b.keys, static_cast<std::size_t>(R) * b.key_stride);
        const auto hist = down(b.hist0, static_cast<std::size_t>(R) * 256);
        if (rg == 1) {
            ref_keys = keys;
            ref_hist = hist;
        } else {
            qc.keys &= same(keys, ref_keys) && same(hist, ref_hist);
        }
    }
    F::set_indexer_score_rows(1);
    F::indexer_select(rows, b.keys, b.key_stride, b.hist0, b.tok, b.n, nullptr);
    F::mla_attention(rows, b.tok, b.n, b.q, dkvb, Hh, b.mw, b.o, nullptr);
    const auto nd = down(b.n, R), tokd = down(b.tok, static_cast<std::size_t>(R) * F::kTokStride);
    const auto od = down(b.o, static_cast<std::size_t>(R) * Hh * kMlaDim);
    const auto kd = down(b.keys, static_cast<std::size_t>(R) * b.key_stride), hd = down(b.hist0, static_cast<std::size_t>(R) * 256);
    const auto qtd = down(b.mw.qt, static_cast<std::size_t>(Hh) * kKvLora);
    for (const auto& pr : pads) {
        const auto bytes = down(static_cast<const std::uint8_t*>(pr.p), pr.bytes);
        qc.pad &= std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t v) { return v == 0xff; });
    }
    for (unsigned r = 0; r < R; ++r) {
        const H::DsaCache& hc = *hcs[r];
        const float* qIr = qI.data() + static_cast<std::size_t>(r) * kIdxHeads * kIdxDim;
        const unsigned P = (tt[r] + 1) / kKpool;
        float wh[kIdxHeads];
        for (unsigned h = 0; h < kIdxHeads; ++h) wh[h] = wraw[r * kIdxHeads + h] * kInvSqrt32;
        std::vector<std::uint32_t> hist(256, 0);
        for (unsigned p = 0; p < P; ++p) {
            const std::uint32_t key = ordered_key(host_score(hc, p, qIr, wh));
            qc.keys &= kd[static_cast<std::size_t>(r) * b.key_stride + p] == key;
            ++hist[key >> 24U];
        }
        qc.keys &= std::equal(hist.begin(), hist.end(), hd.begin() + static_cast<std::size_t>(r) * 256);
        // the host list from the host function itself (its own dense for w_h): equal to the device list
        std::vector<std::uint32_t> ref;
        {
            std::vector<float> I(P);
            for (unsigned p = 0; p < P; ++p) I[p] = host_score(hc, p, qIr, wh);
            if (P > 0)
                for (std::uint32_t p : H::topk_ordered(I, std::min(kIdxPools, P)))
                    for (unsigned j = 0; j < kKpool; ++j) ref.push_back(p * kKpool + j);
            for (unsigned pos = P * kKpool; pos <= tt[r]; ++pos) ref.push_back(pos);
        }
        if (nd[r] > F::kTokStride) {
            qc.lists = false;
            continue;
        }
        const std::vector<std::uint32_t> got(tokd.begin() + static_cast<std::size_t>(r) * F::kTokStride,
                                             tokd.begin() + static_cast<std::size_t>(r) * F::kTokStride + nd[r]);
        qc.lists &= got == ref;
        const std::span<const float> qr{q.data() + static_cast<std::size_t>(r) * Hh * kMlaDim, Hh * kMlaDim};
        if (r == 0)
            for (unsigned h = 0; h < Hh; ++h)
                for (unsigned l = 0; l < kKvLora; ++l) {
                    const float v = chain_dot([&](unsigned d) { return qr[h * kMlaDim + d]; },
                                              [&](unsigned d) { return f16_to_float(w.kvb[static_cast<std::size_t>(h * 2 * kMlaDim + d) * kKvLora + l]); }, kMlaDim);
                    qc.qt &= std::memcmp(&v, &qtd[h * kKvLora + l], 4) == 0;
                }
        std::vector<float> oh(Hh * kMlaDim);
        H::mla_attention(hc, ref, qr, H::Mat{w.kvb.data(), H::WType::F16, Hh * 2 * kMlaDim, kKvLora}, Hh, oh);
        qc.outs &= same_n(oh.data(), od.data() + static_cast<std::size_t>(r) * Hh * kMlaDim, Hh * kMlaDim);
    }
}

// Direct score gate: no cache conversions hide non-finite or subnormal inputs.
void test_idx_score_rows(std::mt19937& g) {
    constexpr unsigned max_rows = 8, pools = 131, stride = pools + 3;
    constexpr std::uint32_t poison = 0xa5a5a5a5U;
    const std::uint32_t edge_bits[] = {0x7fc12345U, 0xffc54321U, 0x7f800000U, 0xff800000U,
                                     0x80000000U, 0x00000001U, 0x80000001U, 0x00000000U};
    float* pooled[2] = {dalloc<float>(pools * kIdxDim), dalloc<float>(pools * kIdxDim)};
    float *qI = dalloc<float>(max_rows * kIdxHeads * kIdxDim), *wraw = dalloc<float>(max_rows * kIdxHeads);
    auto* slots = dalloc<D::DsaSlot>(max_rows);
    auto* tpos = dalloc<unsigned>(max_rows);
    auto* keys = dalloc<std::uint32_t>((max_rows + 1) * stride);
    auto* hist = dalloc<std::uint32_t>((max_rows + 1) * 256);
    bool equal[2] = {true, true}, padding = true;
    // Separate q/key edge cases also exercise finite scores alongside NaN-clamped heads.
    for (unsigned mode = 0; mode < 4; ++mode) {
        for (unsigned s = 0; s < 2; ++s) {
            auto k = rnd(pools * kIdxDim, g, 1.0F);
            if ((mode & 2U) != 0)
                for (unsigned p = 0; p < pools; ++p)
                    k[p * kIdxDim + (p % kIdxDim)] = __builtin_bit_cast(float, edge_bits[(p + s) % 8]);
            put(pooled[s], k);
        }
        for (unsigned R = 1; R <= max_rows; ++R) {
            auto q = rnd(R * kIdxHeads * kIdxDim, g, 1.0F);
            if ((mode & 1U) != 0)
                for (unsigned r = 0; r < R; ++r)
                    for (unsigned h = 0; h < kIdxHeads; ++h)
                        q[(r * kIdxHeads + h) * kIdxDim + (h % kIdxDim)] =
                            __builtin_bit_cast(float, edge_bits[(r + h) % 8]);
            put(qI, q);
            put(wraw, rnd(R * kIdxHeads, g, 1.0F));
            for (unsigned mixed = 0; mixed < 2; ++mixed) {
                std::vector<D::DsaSlot> sl(R);
                for (unsigned r = 0; r < R; ++r) sl[r].pooled = pooled[mixed != 0 ? r % 2 : 0];
                put(slots, sl);
                for (unsigned start : {0U, 247U, 251U, 503U, 515U}) {
                    std::vector<unsigned> tt(R);
                    for (unsigned r = 0; r < R; ++r) tt[r] = start + r;
                    put(tpos, tt);
                    std::vector<std::uint32_t> ref_keys, ref_hist;
                    for (unsigned rg : {1U, 2U, 3U, 4U, 5U}) {
                        const std::vector<std::uint32_t> key_init(static_cast<std::size_t>(R + 1) * stride, poison);
                        std::vector<std::uint32_t> hist_init(static_cast<std::size_t>(R + 1) * 256, 0);
                        std::fill(hist_init.begin() + static_cast<std::size_t>(R) * 256, hist_init.end(), poison);
                        put(keys, key_init);
                        put(hist, hist_init);
                        F::set_indexer_score_rows(rg);
                        F::indexer_score(F::DsaRows{slots, tpos, R}, qI, wraw, keys, stride, hist, pools, nullptr);
                        const auto got_keys = down(keys, key_init.size()), got_hist = down(hist, hist_init.size());
                        if (rg == 1) {
                            ref_keys = got_keys;
                            ref_hist = got_hist;
                        } else {
                            equal[mixed] &= same(got_keys, ref_keys) && same(got_hist, ref_hist);
                        }
                        for (unsigned r = 0; r < R; ++r) {
                            const unsigned P = (tt[r] + 1) / kKpool;
                            for (unsigned p = P; p < stride; ++p) padding &= got_keys[r * stride + p] == poison;
                            std::uint32_t count = 0;
                            for (unsigned e = 0; e < 256; ++e) count += got_hist[r * 256 + e];
                            padding &= count == P;
                        }
                        for (unsigned p = 0; p < stride; ++p) padding &= got_keys[R * stride + p] == poison;
                        for (unsigned e = 0; e < 256; ++e) padding &= got_hist[R * 256 + e] == poison;
                    }
                }
            }
        }
    }
    F::set_indexer_score_rows(1);
    check(equal[0], "idx_score rg2/rg3/rg4/rg5 == rg1: keys/hist memcmp, R1..8 same slot, consecutive positions, finite and q/key NaN/Inf/-0/subnormals");
    check(equal[1], "idx_score rg2/rg3/rg4/rg5 == rg1: keys/hist memcmp, R1..8 alternating slots, finite and q/key NaN/Inf/-0/subnormals");
    check(padding, "idx_score rg1/rg2/rg3/rg4/rg5: zero pools, partial/crossed tiles, histogram counts, untouched key tails and padding row");
    for (float* p : {pooled[0], pooled[1], qI, wraw}) fr(p);
    fr(slots); fr(tpos); fr(keys); fr(hist);
}

void test_dsa(const std::string& cases, std::mt19937& g) {
    const IdxW w = load_idx(cases);
    const auto hw = w.hw();
    const D::IndexerDev dw{up(w.wk), up(w.gate), up(w.wp), up(w.knw), up(w.knb), up(w.ape)};
    std::uint16_t* dkvb = up(w.kvb);
    constexpr unsigned Hh = 32, T = 3000, cap = 3072;
    // (a) append 3000 tokens to 3 slots through the fused dense + dsa_store: slots 0 and 1 one position per call
    //     (rows = 2 slots), slot 2 several consecutive positions of the same slot per call (1..8 rows, crossing pool
    //     boundaries; the store/pool split makes this legal).
    H::DsaCache hc[3];
    DevSlot ds[3];
    for (auto& d : ds) d.alloc(cap);
    DsaBufs b;
    b.alloc(9, cap / kKpool);  // 8 rows + the padding row
    unsigned* flag = new_flag();
    float *dx = zalloc<float>(8 * kHidden), *dlat = zalloc<float>(8 * kKvLora), *dkey = zalloc<float>(8 * kIdxDim), *dgs = zalloc<float>(8 * kIdxDim);
    std::vector<std::vector<float>> xs(T * 3);
    auto append_rows = [&](const std::vector<unsigned>& slot_of, const std::vector<unsigned>& pos) {
        const unsigned R = static_cast<unsigned>(slot_of.size());
        std::vector<float> xr, lr;
        std::vector<D::DsaSlot> sl;
        for (unsigned r = 0; r < R; ++r) {
            auto& x = xs[pos[r] * 3 + slot_of[r]];
            x = rnd(kHidden, g, 0.5F);
            const auto lat = rnd(kKvLora, g, 1.0F);
            H::dsa_append(hc[slot_of[r]], lat, x, hw);
            xr.insert(xr.end(), x.begin(), x.end());
            lr.insert(lr.end(), lat.begin(), lat.end());
            sl.push_back(ds[slot_of[r]].s);
        }
        put(dx, xr);
        put(dlat, lr);
        put(b.dslots, sl);
        put(b.dt, pos);
        const F::DenseJob j2[2] = {{dw.wk, dx, dkey, kIdxDim, kHidden, kHidden, kIdxDim}, {dw.gate, dx, dgs, kIdxDim, kHidden, kHidden, kIdxDim}};
        F::dense_f16(j2, 2, R, nullptr);
        F::dsa_store(F::DsaRows{b.dslots, b.dt, R}, dlat, dkey, dgs, dw, b.hist0, flag, nullptr);
    };
    for (unsigned t = 0; t < T; ++t) append_rows({0, 1}, {t, t});
    const unsigned sizes[] = {3, 1, 4, 2, 5, 8, 7, 6};
    for (unsigned t = 0, k = 0; t < T; ++k) {
        const unsigned n = std::min(sizes[k % 8], T - t);
        std::vector<unsigned> so(n, 2), po(n);
        for (unsigned r = 0; r < n; ++r) po[r] = t + r;
        append_rows(so, po);
        t += n;
    }
    bool app = read_flag(flag) == 0;
    for (unsigned r = 0; r < 3; ++r)
        app &= same(down(ds[r].s.latent, static_cast<std::size_t>(T) * kKvLora), hc[r].latent) &&
               same(down(ds[r].s.ikey, static_cast<std::size_t>(T) * kIdxDim), hc[r].ikey) &&
               same(down(ds[r].s.igate, static_cast<std::size_t>(T) * kIdxDim), hc[r].igate) &&
               same(down(ds[r].s.pooled, static_cast<std::size_t>(T / kKpool) * kIdxDim), hc[r].pooled);
    check(app, "R4/R12 fused dense + dsa_store (real L3 indexer weights), 3 slots x 3000 tokens incl. 1..8 consecutive positions of one slot per call: latent/key/gate/pooled caches == host, no flag");
    // (b) queries: R = 1..8 rows per call mixing the 3 slots and positions (0..4, 2047..2052, 2999), real-scale and
    //     all-tie qI; keys + histogram, token lists, absorbed q, MLA outputs == host; == batch-2 kernels at R = 1.
    const std::vector<unsigned> ts = {0, 1, 2, 3, 4, 2047, 2048, 2049, 2050, 2051, 2052, 2999};
    QueryCheck qc;
    bool ties = true, olds = true;
    float *oq = zalloc<float>(Hh * kMlaDim), *oqt = zalloc<float>(Hh * kKvLora), *oo = zalloc<float>(Hh * kMlaDim), *owh = zalloc<float>(kIdxHeads);
    float* osc = zalloc<float>(cap / kKpool);
    std::uint32_t *owork = zalloc<std::uint32_t>(cap), *otok = zalloc<std::uint32_t>(F::kTokStride), *on = zalloc<std::uint32_t>(1);
    for (int mode = 0; mode < 2; ++mode)
        for (unsigned R = 1; R <= 8; ++R)
            for (unsigned start = 0; start < ts.size(); start += R) {
                std::vector<D::DsaSlot> sl(R);
                std::vector<unsigned> tt(R), slot_of(R);
                std::vector<const H::DsaCache*> hcs(R);
                std::vector<float> qI, q, wraw;
                for (unsigned r = 0; r < R; ++r) {
                    tt[r] = ts[(start + r) % ts.size()];
                    slot_of[r] = (start + r + R) % 3;
                    sl[r] = ds[slot_of[r]].s;
                    hcs[r] = &hc[slot_of[r]];
                    const auto qi = mode == 0 ? rnd(kIdxHeads * kIdxDim, g, 1.0F) : std::vector<float>(kIdxHeads * kIdxDim, 0.0F);
                    qI.insert(qI.end(), qi.begin(), qi.end());
                    const auto qq = rnd(Hh * kMlaDim, g, 1.0F);
                    q.insert(q.end(), qq.begin(), qq.end());
                    std::vector<float> wr(kIdxHeads);
                    H::dense(xs[tt[r] * 3 + slot_of[r]], hw.weights_proj, wr);  // == the device dense (checked in test_dense)
                    wraw.insert(wraw.end(), wr.begin(), wr.end());
                }
                put(b.dslots, sl);
                put(b.dt, tt);
                put(b.qI, qI);
                put(b.q, q);
                put(b.wraw, wraw);
                run_and_check(b, R, hcs, tt, qI, q, wraw, w, dkvb, cap / kKpool, qc);
                const auto nd = down(b.n, R), tokd = down(b.tok, static_cast<std::size_t>(R) * F::kTokStride);
                const auto od = down(b.o, static_cast<std::size_t>(R) * Hh * kMlaDim);
                for (unsigned r = 0; r < R; ++r) {
                    // the host indexer_select itself (its own dense for w_h) gives the same list
                    const auto ref = H::indexer_select(*hcs[r], tt[r], {qI.data() + static_cast<std::size_t>(r) * kIdxHeads * kIdxDim, kIdxHeads * kIdxDim},
                                                       xs[tt[r] * 3 + slot_of[r]], hw);
                    if (nd[r] > F::kTokStride) {
                        qc.lists = false;
                        continue;
                    }
                    const std::vector<std::uint32_t> got(tokd.begin() + static_cast<std::size_t>(r) * F::kTokStride,
                                                         tokd.begin() + static_cast<std::size_t>(r) * F::kTokStride + nd[r]);
                    qc.lists &= got == ref;
                    if (mode == 1 && tt[r] >= 2051) {  // all tie -> the lowest 512 pools
                        bool low = ref.size() >= kIdxTopk;
                        for (unsigned i = 0; low && i < kIdxTopk; ++i) low = ref[i] == i;
                        ties &= low;
                    }
                    if (R == 1) {  // batch-2 kernels on the same inputs
                        float* dxr = up(xs[tt[r] * 3 + slot_of[r]]);
                        D::indexer_select(ds[slot_of[r]].s, tt[r], b.qI, dxr, dw, owh, osc, owork, otok, on, nullptr);
                        hok(hipMemcpy(oq, b.q, Hh * kMlaDim * 4, hipMemcpyDeviceToDevice), "c");
                        D::mla_attention(ds[slot_of[r]].s, otok, on, oq, dkvb, Hh, oqt, oo, nullptr);
                        const auto n0 = down(on, 1)[0];
                        olds &= n0 == nd[r] && n0 <= F::kTokStride && same(down(otok, n0), got) &&
                                same(down(oo, Hh * kMlaDim), std::vector<float>(od.begin(), od.begin() + Hh * kMlaDim));
                        fr(dxr);
                    }
                }
            }
    for (int variant = 0; variant < 2; ++variant) {  // 0: lane everything; 1: staged scores on; R = 8 rows, same checks
        if (variant == 0)
            F::set_dense_schedule(F::DenseSchedule::kLane);
        else
            F::set_scores_schedule(F::DenseSchedule::kStaged);
        const unsigned R = 8;
        std::vector<D::DsaSlot> sl(R);
        std::vector<unsigned> tt(R);
        std::vector<const H::DsaCache*> hcs(R);
        std::vector<float> qI, q, wraw;
        for (unsigned r = 0; r < R; ++r) {
            tt[r] = ts[(r * 5) % ts.size()];
            sl[r] = ds[r % 3].s;
            hcs[r] = &hc[r % 3];
            const auto qi = rnd(kIdxHeads * kIdxDim, g, 1.0F);
            qI.insert(qI.end(), qi.begin(), qi.end());
            const auto qq = rnd(Hh * kMlaDim, g, 1.0F);
            q.insert(q.end(), qq.begin(), qq.end());
            std::vector<float> wr(kIdxHeads);
            H::dense(xs[tt[r] * 3 + r % 3], hw.weights_proj, wr);
            wraw.insert(wraw.end(), wr.begin(), wr.end());
        }
        put(b.dslots, sl);
        put(b.dt, tt);
        put(b.qI, qI);
        put(b.q, q);
        put(b.wraw, wraw);
        QueryCheck ql;
        run_and_check(b, R, hcs, tt, qI, q, wraw, w, dkvb, cap / kKpool, ql);
        F::set_dense_schedule(F::DenseSchedule::kStaged);
        F::set_scores_schedule(F::DenseSchedule::kLane);
        check(ql.keys && ql.lists && ql.qt && ql.outs && ql.pad,
              variant == 0 ? "lane schedule (STRIX_GLM_ATTN_SCHED=lane): same query-path checks, R=8"
                           : "staged scores (STRIX_GLM_ATTN_SCORES=staged): same query-path checks, R=8");
    }
    check(qc.keys, "R11 indexer keys (ordered) + top-byte histogram == host, R=1..8, 3 slots");
    check(qc.lists, "R5 + section 1.4 token lists, R=1..8 rows mixing 3 slots, t = 0..4, 2047..2052, 2999, real-scale and all-tie qI == host indexer_select");
    check(ties, "all-tie queries (qI = 0) select the lowest pools (lower-index rule)");
    check(qc.qt, "R13 absorbed q (row 0 of every call) == host chain");
    check(qc.outs, "R13 fused MLA (absorb/scores/softmax/pv/out), 32 rank heads, real L3 kv_b, R=1..8 == host (n up to 2051)");
    check(qc.pad, "DSA query path: the 0xFF-poisoned padding row of every input/output/workspace buffer untouched, R=1..8");
    check(olds, "fast indexer + MLA == batch-2 kernels (indexer_select + mla_attention) bitwise, R=1");
    // (c) C3 / C4: a latent that rounds to bf16 inf, and a pooled key that becomes NaN (ape = +inf), flag + host rejects
    {
        DevSlot d;
        d.alloc(8);
        H::DsaCache h3;
        std::vector<float> lat = rnd(kKvLora, g, 1.0F);
        lat[7] = 3.4e38F;  // finite fp32, bf16 RNE -> +inf
        const auto x = rnd(kHidden, g, 0.5F);
        bool host_c3 = false;
        try {
            H::dsa_append(h3, lat, x, hw);
        } catch (const H::NonFinite&) {
            host_c3 = true;
        }
        put(dx, x);
        put(dlat, lat);
        put(b.dslots, std::vector<D::DsaSlot>{d.s});
        put(b.dt, std::vector<unsigned>{0});
        hok(hipMemset(flag, 0, 4), "m");
        const F::DenseJob j2[2] = {{dw.wk, dx, dkey, kIdxDim, kHidden, kHidden, kIdxDim}, {dw.gate, dx, dgs, kIdxDim, kHidden, kHidden, kIdxDim}};
        F::dense_f16(j2, 2, 1, nullptr);
        F::dsa_store(F::DsaRows{b.dslots, b.dt, 1}, dlat, dkey, dgs, dw, b.hist0, flag, nullptr);
        check(host_c3 && read_flag(flag) == 1, "C3 latent finite -> bf16 inf: device flag set, host rejects");
        // C4
        std::vector<float> ape_inf = w.ape;
        for (unsigned i = 0; i < kIdxDim; ++i) ape_inf[i] = __builtin_huge_valf();
        IdxW w4 = w;
        w4.ape = ape_inf;
        const auto hw4 = w4.hw();
        H::DsaCache h4;
        bool host_c4 = false;
        float* dape = up(ape_inf);
        D::IndexerDev dw4 = dw;
        dw4.ape = dape;
        hok(hipMemset(flag, 0, 4), "m");
        for (unsigned t = 0; t < kKpool; ++t) {
            const auto lt = rnd(kKvLora, g, 1.0F), xt = rnd(kHidden, g, 0.5F);
            try {
                H::dsa_append(h4, lt, xt, hw4);
            } catch (const H::NonFinite&) {
                host_c4 = true;
            }
            put(dx, xt);
            put(dlat, lt);
            put(b.dt, std::vector<unsigned>{t});
            F::dense_f16(j2, 2, 1, nullptr);
            F::dsa_store(F::DsaRows{b.dslots, b.dt, 1}, dlat, dkey, dgs, dw4, b.hist0, flag, nullptr);
        }
        check(host_c4 && read_flag(flag) == 1, "C4 pooled key NaN (ape = +inf): device flag set, host rejects");
        fr(dape);
        d.release();
    }
    for (float* p : {dx, dlat, dkey, dgs, oq, oqt, oo, owh, osc}) fr(p);
    for (std::uint32_t* p : {owork, otok, on}) fr(p);
    fr(flag);
    for (auto& d : ds) d.release();
    fr(dkvb);
}

// (d) large P through the whole query path: synthetic caches up to t = 131071 (P = 32768). Pooled keys either quantized
//     or REPEATED vectors (37 distinct rows reused: exact score ties that straddle the k-th key), 2 rows.
void test_large_p(const std::string& cases, std::mt19937& g) {
    const IdxW w = load_idx(cases);
    const auto hw = w.hw();
    std::uint16_t* dkvb = up(w.kvb);
    constexpr unsigned Hh = 32;
    QueryCheck qc;
    for (unsigned T : {8192U, 32768U, 131072U})
        for (int repeated = 0; repeated < 2; ++repeated) {
            H::DsaCache hc;
            hc.latent = rnd_bf16(static_cast<std::size_t>(T) * kKvLora, g, 1.0F);
            hc.ikey.assign(static_cast<std::size_t>(T) * kIdxDim, 0);
            hc.igate.assign(static_cast<std::size_t>(T) * kIdxDim, 0);
            hc.pooled.resize(static_cast<std::size_t>(T / kKpool) * kIdxDim);
            if (repeated) {
                const auto base = rnd(37 * kIdxDim, g, 1.0F);
                for (unsigned p = 0; p < T / kKpool; ++p)
                    std::copy_n(base.begin() + ((p * 2654435761U) >> 7U) % 37 * kIdxDim, kIdxDim, hc.pooled.begin() + static_cast<std::size_t>(p) * kIdxDim);
            } else {
                std::uniform_int_distribution<int> lv(-8, 8);
                for (auto& v : hc.pooled) v = static_cast<float>(lv(g)) * 0.25F;
            }
            DevSlot ds;
            ds.alloc(T);
            put(ds.s.latent, hc.latent);
            put(ds.s.pooled, hc.pooled);
            DsaBufs b;
            b.alloc(3, T / kKpool);  // 2 rows + the padding row
            for (unsigned tq : {T - 1, T - 3, T / 2 + 1}) {
                const std::vector<unsigned> tt = {tq, T - 2};
                std::vector<float> qI, q, wraw;
                for (unsigned r = 0; r < 2; ++r) {
                    const auto xr = rnd(kHidden, g, 0.5F);
                    const auto qi = rnd(kIdxHeads * kIdxDim, g, 1.0F);
                    qI.insert(qI.end(), qi.begin(), qi.end());
                    const auto qq = rnd(Hh * kMlaDim, g, 1.0F);
                    q.insert(q.end(), qq.begin(), qq.end());
                    std::vector<float> wr(kIdxHeads);
                    H::dense(xr, hw.weights_proj, wr);
                    wraw.insert(wraw.end(), wr.begin(), wr.end());
                }
                put(b.dslots, std::vector<D::DsaSlot>{ds.s, ds.s});
                put(b.dt, tt);
                put(b.qI, qI);
                put(b.q, q);
                put(b.wraw, wraw);
                run_and_check(b, 2, {&hc, &hc}, tt, qI, q, wraw, w, dkvb, T / kKpool, qc);
            }
            ds.release();
        }
    check(qc.keys && qc.lists, "R11/R5 large P (2048, 8192, 32768 pools; quantized and repeated pooled vectors = exact ties across the k-th key): keys, histogram, token lists == host, 2 rows");
    check(qc.qt && qc.outs && qc.pad, "R13 fused MLA on the large-P selections == host; padding row untouched");
    fr(dkvb);
}

// (e) the selection alone on adversarial keys: few distinct values straddling the k-th key, -0 / +0 mixed, subnormals,
//     negatives, all equal; P = 513 .. 32768 including non-multiples of 1024. Host = topk_ordered on the same scores.
void test_select_keys(std::mt19937& g) {
    bool all = true;
    for (unsigned P : {513U, 1000U, 1024U, 1025U, 3000U, 8192U, 32767U, 32768U})
        for (int kind = 0; kind < 4; ++kind) {
            std::vector<float> sc(P);
            std::uniform_int_distribution<int> u(0, 1 << 20);
            for (unsigned p = 0; p < P; ++p) {
                const int v = u(g);
                switch (kind) {
                    case 0: sc[p] = static_cast<float>(v % 3) - 1.0F; break;                                  // 3 values
                    case 1: sc[p] = (v % 4 == 0) ? -0.0F : (v % 4 == 1 ? 0.0F : (v % 4 == 2 ? 1e-45F : -1e-45F)); break;  // +-0, subnormals
                    case 2: sc[p] = static_cast<float>(v % 997) * -1.0e-40F + (v % 2 ? 0.0F : -0.0F); break;  // negative subnormals
                    default: sc[p] = 0.5F; break;                                                            // all equal
                }
            }
            std::vector<std::uint32_t> keys(P), hist(256, 0);
            for (unsigned p = 0; p < P; ++p) {
                keys[p] = ordered_key(sc[p]);
                ++hist[keys[p] >> 24U];
            }
            const unsigned t = P * kKpool - 1 + (P % 4);  // tail of 0..3 positions, P visible pools
            std::uint32_t *dk = up(keys), *dh = up(hist), *dtok = zalloc<std::uint32_t>(F::kTokStride), *dn = zalloc<std::uint32_t>(1);
            unsigned* dt = up(std::vector<unsigned>{t});
            D::DsaSlot* dsl = zalloc<D::DsaSlot>(1);
            F::indexer_select(F::DsaRows{dsl, dt, 1}, dk, P, dh, dtok, dn, nullptr);
            const auto n = down(dn, 1)[0];
            std::vector<std::uint32_t> ref;
            for (std::uint32_t p : H::topk_ordered(sc, std::min(kIdxPools, P)))
                for (unsigned j = 0; j < kKpool; ++j) ref.push_back(p * kKpool + j);
            for (unsigned pos = P * kKpool; pos <= t; ++pos) ref.push_back(pos);
            all &= n <= F::kTokStride && same(down(dtok, n), ref);
            for (std::uint32_t* p : {dk, dh, dtok, dn}) fr(p);
            fr(dt);
            fr(dsl);
        }
    check(all, "R5 select alone on adversarial keys (3-value ties at the k-th key, -0/+0, subnormals, all equal; P = 513..32768, tails 0..3) == host stable order");
}

// (f) review #1: the FastAttn score skip (p_bound <= kIdxPools -> indexer_score not run). Selection alone on STALE
//     keys / histogram (random garbage, as left by an earlier call) for rows with P = 0 / 1 / 511 / 512 visible pools
//     (tails 0..3, speculative-style consecutive positions in one call) must equal the host list; then P = 513 rows on
//     the same (reused) key / hist / token buffers with real keys must equal the host stable top-k.
void test_select_skip(std::mt19937& g) {
    bool skip_ok = true, reuse_ok = true;
    constexpr unsigned R = 4, KS = 600;
    std::uint32_t *dk = zalloc<std::uint32_t>(static_cast<std::size_t>(R) * KS), *dh = zalloc<std::uint32_t>(static_cast<std::size_t>(R) * 256);
    std::uint32_t *dtok = zalloc<std::uint32_t>(static_cast<std::size_t>(R) * F::kTokStride), *dn = zalloc<std::uint32_t>(R);
    unsigned* dt = zalloc<unsigned>(R);
    D::DsaSlot* dsl = zalloc<D::DsaSlot>(R);
    std::uniform_int_distribution<std::uint32_t> u32;
    const auto host_list = [](unsigned t, const std::vector<float>* sc) {
        const unsigned P = (t + 1) / kKpool;
        std::vector<std::uint32_t> ref;
        if (sc == nullptr) {
            for (unsigned i = 0; i < P * kKpool; ++i) ref.push_back(i);
        } else {
            for (std::uint32_t p : H::topk_ordered(*sc, std::min(kIdxPools, P)))
                for (unsigned j = 0; j < kKpool; ++j) ref.push_back(p * kKpool + j);
        }
        for (unsigned pos = P * kKpool; pos <= t; ++pos) ref.push_back(pos);
        return ref;
    };
    const auto run = [&](const std::vector<unsigned>& tt, const std::vector<std::vector<float>>* scs) {
        put(dt, tt);
        hok(hipMemset(dtok, 0xff, static_cast<std::size_t>(R) * F::kTokStride * 4), "poison tok");  // no stale outputs
        hok(hipMemset(dn, 0xff, static_cast<std::size_t>(R) * 4), "poison n");
        F::indexer_select(F::DsaRows{dsl, dt, static_cast<unsigned>(tt.size())}, dk, KS, dh, dtok, dn, nullptr);
        const auto n = down(dn, tt.size());
        const auto tok = down(dtok, tt.size() * F::kTokStride);
        bool ok = true;
        for (std::size_t r = 0; r < tt.size(); ++r) {
            const auto ref = host_list(tt[r], scs ? &(*scs)[r] : nullptr);
            ok &= n[r] == ref.size() &&
                  std::equal(ref.begin(), ref.end(), tok.begin() + static_cast<std::ptrdiff_t>(r * F::kTokStride));
        }
        return ok;
    };
    const std::vector<std::vector<unsigned>> skip_sets = {
        {0, 1, 2, 3}, {3, 4, 5, 6}, {2043, 2044, 2045, 2046}, {2047, 2048, 2049, 2050}, {2050, 2046, 0, 2047}};
    for (int rep = 0; rep < 2; ++rep)
        for (const auto& tt : skip_sets) {
            std::vector<std::uint32_t> junk(static_cast<std::size_t>(R) * KS), hj(static_cast<std::size_t>(R) * 256);
            for (auto& v : junk) v = u32(g);
            for (auto& v : hj) v = u32(g) % 4096U;
            put(dk, junk);
            put(dh, hj);
            skip_ok &= run(tt, nullptr);
        }
    // reuse: P = 513 (t = 2052..2055) on the same buffers, real keys + histogram of random scores with ties
    for (int rep = 0; rep < 3; ++rep) {
        const std::vector<unsigned> tt = {2052, 2053, 2054, 2055};
        std::vector<std::vector<float>> scs(R);
        std::vector<std::uint32_t> keys(static_cast<std::size_t>(R) * KS, 0xDEADBEEFU), hist(static_cast<std::size_t>(R) * 256, 0);
        std::uniform_int_distribution<int> ui(0, 40);
        for (unsigned r = 0; r < R; ++r) {
            const unsigned P = (tt[r] + 1) / kKpool;
            scs[r].resize(P);
            for (unsigned p = 0; p < P; ++p) {
                scs[r][p] = static_cast<float>(ui(g) - 20) * 0.25F;
                keys[r * KS + p] = ordered_key(scs[r][p]);
                ++hist[r * 256 + (keys[r * KS + p] >> 24U)];
            }
        }
        put(dk, keys);
        put(dh, hist);
        reuse_ok &= run(tt, &scs);
        // and straight back to a skip call on the reused buffers
        skip_ok &= run({2047, 2050, 7, 0}, nullptr);
    }
    for (std::uint32_t* p : {dk, dh, dtok, dn}) fr(p);
    fr(dt);
    fr(dsl);
    check(skip_ok, "R5 skip path (regression case): select alone on stale keys/hist, P = 0/1/511/512 rows (tails 0..3, mixed rows) == host take-all");
    check(reuse_ok, "R5 skip -> P = 513 on reused key/hist/token buffers == host stable top-k, then skip again");
}

// Dense edge geometry: partial tiles (n = 1, 15, 17), a grouped job mixed with ungrouped jobs in one launch, and an
// x pointer misaligned by one float (staged needs 16 B: the launch falls back to the lane schedule). Every schedule
// (staged tile 16, staged tile 8, lane) == host, R = 1..8, outputs past n untouched.
void test_dense_edges(std::mt19937& g) {
    bool all = true;
    const auto w1 = rnd_f16(static_cast<std::size_t>(17) * 4096, g, 0.05F), w2 = rnd_f16(static_cast<std::size_t>(3) * 2 * 17 * 512, g, 0.05F);
    std::uint16_t *d1 = up(w1), *d2 = up(w2);
    for (unsigned R = 1; R <= 8; ++R) {
        const auto x = rnd(static_cast<std::size_t>(R) * 4096 + 1, g, 0.5F), xg = rnd(static_cast<std::size_t>(R) * 3 * 512, g, 0.5F);
        float *dx = up(x), *dxg = up(xg);
        for (int mode = 0; mode < 3; ++mode) {
            F::set_dense_schedule(mode == 2 ? F::DenseSchedule::kLane : F::DenseSchedule::kStaged);
            F::set_dense_tile(mode == 1 ? 8 : 16);
            for (int mis = 0; mis < 2; ++mis) {
                const float* xin = dx + mis;  // mis = 1: misaligned x -> lane fallback
                const unsigned ys = 64;
                float* y = up(std::vector<float>(static_cast<std::size_t>(R) * ys * 4, std::nanf("")));
                F::DenseJob j[4];
                j[0] = {d1, xin, y, 1, 4096, 4096, ys};
                j[1] = {d1 + 4096, xin, y + 1, 15, 4096, 4096, ys};
                j[2] = {d1, xin, y + 16, 17, 4096, 4096, ys};
                j[3] = {d2, dxg, y + 33, 17, 512, 3 * 512, ys};
                j[3].groups = 3;
                j[3].w_gstride = static_cast<std::size_t>(2) * 17 * 512;
                j[3].x_gstride = 512;
                j[3].y_gstride = 17;  // grouped outputs go to the second half of y: [R][3 groups x 17]
                j[3].y = y + static_cast<std::size_t>(R) * ys * 2;
                j[3].y_stride = 3 * 17;
                F::dense_f16(j, 4, R, nullptr);
                const auto yd = down(y, static_cast<std::size_t>(R) * ys * 4);
                for (unsigned r = 0; r < R; ++r) {
                    const std::span<const float> xr{x.data() + mis + static_cast<std::size_t>(r) * 4096, 4096};
                    std::vector<float> a(1), b(15), c(17);
                    H::dense(xr, H::Mat{w1.data(), H::WType::F16, 17, 4096}, a);
                    H::dense(xr, H::Mat{w1.data() + 4096, H::WType::F16, 16, 4096}, b);
                    H::dense(xr, H::Mat{w1.data(), H::WType::F16, 17, 4096}, c);
                    all &= same_n(a.data(), yd.data() + r * ys, 1) && same_n(b.data(), yd.data() + r * ys + 1, 15) &&
                           same_n(c.data(), yd.data() + r * ys + 16, 17);
                    for (unsigned k = 33; k < ys; ++k) all &= std::isnan(yd[r * ys + k]);  // past every ungrouped job
                    for (unsigned gg = 0; gg < 3; ++gg) {
                        std::vector<float> e(17);
                        H::dense({xg.data() + static_cast<std::size_t>(r) * 3 * 512 + gg * 512, 512},
                                 H::Mat{w2.data() + static_cast<std::size_t>(gg) * 2 * 17 * 512, H::WType::F16, 17, 512}, e);
                        all &= same_n(e.data(), yd.data() + static_cast<std::size_t>(R) * ys * 2 + r * 51 + gg * 17, 17);
                    }
                }
                fr(y);
            }
        }
        F::set_dense_schedule(F::DenseSchedule::kStaged);
        F::set_dense_tile(16);
        fr(dx);
        fr(dxg);
    }
    check(all, "R1 dense edges: n = 1/15/17 partial tiles + a grouped job (3 x 17 x 512) mixed in one launch, misaligned-x fallback; staged tile 16 == staged tile 8 == lane == host, R=1..8, untouched past n");
    fr(d1);
    fr(d2);
}

// MLA scores directly: raw scores and the per-head max keys, staged == lane == host, token counts at the 32/64/256
// boundaries (1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 2051), 3 rows with different n; unused score entries poisoned.
void test_scores_direct(std::mt19937& g) {
    constexpr unsigned Hh = 32, cap = 4096;
    H::DsaCache hc;
    hc.latent = rnd_bf16(static_cast<std::size_t>(cap) * kKvLora, g, 1.0F);
    DevSlot ds;
    ds.alloc(cap);
    put(ds.s.latent, hc.latent);
    const std::vector<unsigned> ns = {1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 2051};
    bool all = true;
    D::DsaSlot* dsl = up(std::vector<D::DsaSlot>(3, ds.s));
    unsigned* dt = zalloc<unsigned>(3);
    for (unsigned start = 0; start < ns.size(); start += 3) {
        std::vector<std::uint32_t> nn(3), tok(3 * F::kTokStride, 0);
        std::uniform_int_distribution<unsigned> pos(0, cap - 1);
        for (unsigned r = 0; r < 3; ++r) {
            nn[r] = ns[(start + r) % ns.size()];
            for (unsigned t = 0; t < nn[r]; ++t) tok[r * F::kTokStride + t] = pos(g);
        }
        const auto qt = rnd(static_cast<std::size_t>(3) * Hh * kKvLora, g, 0.2F);
        std::uint32_t *dtok = up(tok), *dn = up(nn);
        float* dqt = up(qt);
        std::vector<std::vector<float>> sc(2);
        std::vector<std::vector<std::uint32_t>> mk(2);
        for (int mode = 0; mode < 2; ++mode) {
            F::set_scores_schedule(mode == 0 ? F::DenseSchedule::kStaged : F::DenseSchedule::kLane);
            float* dsc = dalloc<float>(static_cast<std::size_t>(3) * Hh * F::kTokStride);
            hok(hipMemset(dsc, 0xff, static_cast<std::size_t>(3) * Hh * F::kTokStride * 4), "m");
            std::uint32_t* dmk = zalloc<std::uint32_t>(3 * Hh);
            F::mla_scores(F::DsaRows{dsl, dt, 3}, dtok, dn, dqt, Hh, dsc, dmk, nullptr);
            sc[mode] = down(dsc, static_cast<std::size_t>(3) * Hh * F::kTokStride);
            mk[mode] = down(dmk, 3 * Hh);
            fr(dsc);
            fr(dmk);
        }
        F::set_scores_schedule(F::DenseSchedule::kLane);
        all &= same(sc[0], sc[1]) && mk[0] == mk[1];
        for (unsigned r = 0; r < 3; ++r)
            for (unsigned h = 0; h < Hh; ++h) {
                float m = 0.0F;
                const float* q = &qt[(static_cast<std::size_t>(r) * Hh + h) * kKvLora];
                const float* got = &sc[0][(static_cast<std::size_t>(r) * Hh + h) * F::kTokStride];
                for (unsigned t = 0; t < nn[r]; ++t) {
                    const std::uint16_t* lat = &hc.latent[static_cast<std::size_t>(tok[r * F::kTokStride + t]) * kKvLora];
                    const float s = chain_dot([&](unsigned l) { return q[l]; }, [&](unsigned l) { return bf16_to_float(lat[l]); }, kKvLora) * kMlaScale;
                    all &= std::memcmp(&s, &got[t], 4) == 0;
                    m = (t == 0 || s > m) ? s : m;
                }
                all &= mk[0][r * Hh + h] == ordered_key(m);
                for (unsigned t = nn[r]; t < F::kTokStride; ++t) all &= __builtin_bit_cast(std::uint32_t, got[t]) == 0xffffffffU;
            }
        fr(dtok);
        fr(dn);
        fr(dqt);
    }
    check(all, "R13 MLA raw scores + per-head max keys: staged == lane == host, n = 1/31/32/33/63/64/65/255/256/257/2051, 3 rows; unused score entries untouched");
    fr(dsl);
    fr(dt);
    ds.release();
}

// PV ownership variants retain the same 256 chains and reduction tree.
void test_pv_direct(std::mt19937& g) {
    constexpr unsigned Hh = 32;
    const std::uint32_t pbits[] = {0x7fc12345U, 0xffc54321U, 0x7f800000U, 0xff800000U,
                                  0x80000000U, 0x00000001U, 0x80000001U, 0x007fffffU};
    const std::uint16_t cbits[] = {0x7fc1U, 0xffc5U, 0x7f80U, 0xff80U, 0x8000U, 0x0001U, 0x8001U, 0x007fU};
    for (unsigned ctx : {1024U, 8192U}) {
        DevSlot ds;
        ds.alloc(ctx);
        const auto finite_lat = rnd_bf16(static_cast<std::size_t>(ctx) * kKvLora, g, 1.0F);
        for (unsigned mode = 0; mode < 2; ++mode) {
            auto lat = finite_lat;
            if (mode != 0)
                for (unsigned t = 0; t < ctx; ++t)
                    for (unsigned l = 0; l < kKvLora; ++l)
                        if (l % 32 < 8) lat[static_cast<std::size_t>(t) * kKvLora + l] = cbits[(t + l) % 8];
            put(ds.s.latent, lat);
            for (unsigned R = 1; R <= 8; ++R) {
                auto* dsl = up(std::vector<D::DsaSlot>(R, ds.s));
                auto* dt = zalloc<unsigned>(R);
                const std::size_t out_count = static_cast<std::size_t>(R) * Hh * kKvLora;
                const std::size_t pad_count = static_cast<std::size_t>(Hh) * kKvLora;
                auto* ob = dalloc<float>(out_count + pad_count);
                auto prob = rnd(static_cast<std::size_t>(R) * Hh * F::kTokStride, g, 0.25F);
                if (mode != 0)
                    for (unsigned r = 0; r < R; ++r)
                        for (unsigned h = 0; h < Hh; ++h)
                            for (unsigned t = 0; t < F::kTokStride; t += 251)
                                prob[(static_cast<std::size_t>(r) * Hh + h) * F::kTokStride + t] =
                                    __builtin_bit_cast(float, pbits[(h + t + r) % 8]);
                auto* dp = up(prob);
                const std::vector<unsigned> counts = {0, 1, 255, 256, 257, 1023, 1024, ctx == 1024 ? 1021U : F::kMaxTok};
                bool all = true;
                for (unsigned start = 0; start < counts.size(); start += R) {
                    std::vector<std::uint32_t> nn(R), tok(static_cast<std::size_t>(R) * F::kTokStride, 0xffffffffU);
                    for (unsigned r = 0; r < R; ++r) {
                        nn[r] = counts[(start + r) % counts.size()];
                        for (unsigned t = 0; t < nn[r]; ++t)
                            tok[static_cast<std::size_t>(r) * F::kTokStride + t] = (37U * t + 113U * r) % ctx;
                    }
                    auto* dn = up(nn);
                    auto* dtok = up(tok);
                    std::vector<std::uint32_t> baseline;
                    for (unsigned v : {0U, 1U, 2U}) {
                        F::set_mla_pv_variant(v);
                        hok(hipMemset(ob, 0xa5, (out_count + pad_count) * sizeof(float)), "PV poison");
                        F::mla_pv(F::DsaRows{dsl, dt, R}, dtok, dn, dp, Hh, ob, nullptr);
                        const auto bits = down(reinterpret_cast<const std::uint32_t*>(ob), out_count + pad_count);
                        if (v == 0) baseline = bits;
                        all &= std::memcmp(bits.data(), baseline.data(), out_count * sizeof(std::uint32_t)) == 0;
                        for (std::size_t i = out_count; i < bits.size(); ++i) all &= bits[i] == 0xa5a5a5a5U;
                        for (unsigned r = 0; r < R; ++r)
                            if (nn[r] == 0)
                                for (unsigned i = 0; i < Hh * kKvLora; ++i)
                                    all &= bits[static_cast<std::size_t>(r) * Hh * kKvLora + i] == 0U;
                    }
                    fr(dn);
                    fr(dtok);
                }
                check(all, "R13 MLA PV variants 0/1/2 bitwise, tails + empty rows + padding, R=" + std::to_string(R) +
                           " ctx=" + std::to_string(ctx) + (mode == 0 ? " finite" : " NaN/Inf/-0/subnormal"));
                fr(dp);
                fr(ob);
                fr(dsl);
                fr(dt);
            }
        }
        ds.release();
    }
    F::set_mla_pv_variant(0);
}

// ================================================================= bench
// Per-layer Tier A attention time per rank (TP2: 32 KDA heads, 32 MLA heads; indexer replicated), rows R, context ctx.
// Cold weights / state: L distinct layer copies rotated (working set >> the 32 MB MALL).
constexpr double kCeilGBs = 246.7;
struct Timer {
    hipEvent_t a, b;
    Timer() {
        hok(hipEventCreate(&a), "e");
        hok(hipEventCreate(&b), "e");
    }
    template <class Fn> double us(Fn fn, unsigned iters) {  // mean per call, after one warm-up call
        fn();
        hok(hipDeviceSynchronize(), "s");
        hok(hipEventRecord(a, nullptr), "r");
        for (unsigned i = 0; i < iters; ++i) fn();
        hok(hipEventRecord(b, nullptr), "r");
        hok(hipEventSynchronize(b), "s");
        float ms = 0;
        hok(hipEventElapsedTime(&ms, a, b), "t");
        return ms * 1000.0 / iters;
    }
};
void bench_line(const char* stage, unsigned R, unsigned ctx, double us, double bytes) {
    const double gbs = bytes / (us * 1e3);
    std::printf("BENCH %-26s R=%u ctx=%6u  %9.2f us  bytes=%10.0f  %7.1f GB/s  %5.1f %% of ceiling (floor %.2f us)\n", stage, R, ctx, us, bytes,
                gbs, 100.0 * gbs / kCeilGBs, bytes / (kCeilGBs * 1e3));
    std::fflush(stdout);
}

__global__ void fill_bf16_kernel(std::uint16_t* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * 256) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U ^ seed;
        h ^= h >> 15U;
        h *= 2246822519U;
        h ^= h >> 13U;
        const float v = (static_cast<float>(h & 0xffffU) - 32768.0F) * (1.0F / 16384.0F);
        p[i] = float_to_bf16(v);
    }
}
__global__ void fill_f32_kernel(float* p, std::size_t n, std::uint32_t seed, float scale) {
    for (std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * 256) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U ^ seed;
        h ^= h >> 15U;
        h *= 2246822519U;
        h ^= h >> 13U;
        p[i] = (static_cast<float>(h & 0xffffU) - 32768.0F) * (scale / 32768.0F);
    }
}
__global__ void fill_f16_kernel(std::uint16_t* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * 256) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U ^ seed;
        h ^= h >> 15U;
        h *= 2246822519U;
        h ^= h >> 13U;
        p[i] = static_cast<std::uint16_t>(0x2000U + (h & 0x0fffU) + ((h & 0x10000U) ? 0x8000U : 0U));  // |w| ~ 2^-7 .. 2^-6
    }
}
void fill_bf16(std::uint16_t* p, std::size_t n, std::uint32_t seed) { fill_bf16_kernel<<<1024, 256>>>(p, n, seed); }
void fill_f16(std::uint16_t* p, std::size_t n, std::uint32_t seed) { fill_f16_kernel<<<1024, 256>>>(p, n, seed); }
void fill_f32(float* p, std::size_t n, std::uint32_t seed, float scale) { fill_f32_kernel<<<1024, 256>>>(p, n, seed, scale); }

void bench_kda(unsigned R) {
    constexpr unsigned Hh = 32, C = Hh * kKdaDim, L = 34;
    const std::size_t SZ = static_cast<std::size_t>(Hh) * kKdaDim * kKdaDim, CZ = static_cast<std::size_t>(3) * C * 3;
    struct Lw {
        std::uint16_t *fa, *ga, *bp, *fb, *gb, *conv, *onorm;
        float *A, *dt, *S, *cs;
    };
    std::vector<Lw> lw(L);
    for (unsigned l = 0; l < L; ++l) {
        auto& x = lw[l];
        x.fa = dalloc<std::uint16_t>(128 * kHidden);
        x.ga = dalloc<std::uint16_t>(128 * kHidden);
        x.bp = dalloc<std::uint16_t>(Hh * kHidden);
        x.fb = dalloc<std::uint16_t>(static_cast<std::size_t>(C) * 128);
        x.gb = dalloc<std::uint16_t>(static_cast<std::size_t>(C) * 128);
        x.conv = dalloc<std::uint16_t>(static_cast<std::size_t>(3) * C * 4);
        x.onorm = dalloc<std::uint16_t>(128);
        x.A = dalloc<float>(Hh);
        x.dt = dalloc<float>(C);
        x.S = dalloc<float>(R * SZ);
        x.cs = dalloc<float>(R * CZ);
        fill_f16(x.fa, 128 * kHidden, l * 11 + 1);
        fill_f16(x.ga, 128 * kHidden, l * 11 + 2);
        fill_f16(x.bp, Hh * kHidden, l * 11 + 3);
        fill_f16(x.fb, static_cast<std::size_t>(C) * 128, l * 11 + 4);
        fill_f16(x.gb, static_cast<std::size_t>(C) * 128, l * 11 + 5);
        fill_bf16(x.conv, static_cast<std::size_t>(3) * C * 4, l * 11 + 6);
        fill_bf16(x.onorm, 128, l * 11 + 7);
        fill_f32(x.A, Hh, l * 11 + 8, 1.0F);
        fill_f32(x.dt, C, l * 11 + 9, 1.0F);
        fill_f32(x.S, R * SZ, l * 11 + 10, 0.01F);
        fill_f32(x.cs, R * CZ, l * 11 + 11, 0.5F);
    }
    float *x = dalloc<float>(R * kHidden), *qkv = dalloc<float>(static_cast<std::size_t>(R) * 3 * C), *fa = dalloc<float>(R * 128), *ga = dalloc<float>(R * 128);
    float *bb = dalloc<float>(R * Hh), *f = dalloc<float>(static_cast<std::size_t>(R) * C), *gt = dalloc<float>(static_cast<std::size_t>(R) * C), *y = dalloc<float>(static_cast<std::size_t>(R) * C);
    float *scr = dalloc<float>(static_cast<std::size_t>(R) * Hh * 4 * kKdaDim), *beta = dalloc<float>(R * Hh), *o = dalloc<float>(static_cast<std::size_t>(R) * C);
    fill_f32(x, R * kHidden, 7, 0.5F);
    fill_f32(qkv, static_cast<std::size_t>(R) * 3 * C, 8, 1.0F);
    unsigned* flag = new_flag();
    hok(hipDeviceSynchronize(), "s");
    Timer tm;
    unsigned cur = 0;
    auto next = [&]() -> Lw& { cur = (cur + 1) % L; return lw[cur]; };
    const double w_small = (2.0 * 128 * kHidden + Hh * kHidden) * 2, w_lr = 2.0 * C * 128 * 2, st = 2.0 * R * (SZ + CZ) * 4;
    auto fast_small = [&](Lw& w) {
        const F::DenseJob j3[3] = {{w.fa, x, fa, 128, kHidden, kHidden, 128}, {w.ga, x, ga, 128, kHidden, kHidden, 128}, {w.bp, x, bb, Hh, kHidden, kHidden, Hh}};
        F::dense_f16(j3, 3, R, nullptr);
    };
    auto fast_lr = [&](Lw& w) {
        const F::DenseJob j2[2] = {{w.fb, fa, f, C, 128, 128, C}, {w.gb, ga, gt, C, 128, 128, C}};
        F::dense_f16(j2, 2, R, nullptr);
    };
    auto fast_step = [&](Lw& w) { F::kda_step(D::KdaDev{Hh, w.conv, w.A, w.dt, w.onorm}, qkv, f, gt, bb, w.S, w.cs, y, flag, R, nullptr); };
    auto old_small = [&](Lw& w) {
        D::dense(x, w.fa, D::WType::F16, kHidden, 0, 128, fa, R, nullptr);
        D::dense(x, w.ga, D::WType::F16, kHidden, 0, 128, ga, R, nullptr);
        D::dense(x, w.bp, D::WType::F16, kHidden, 0, Hh, bb, R, nullptr);
    };
    auto old_lr = [&](Lw& w) {
        D::dense(fa, w.fb, D::WType::F16, 128, 0, C, f, R, nullptr);
        D::dense(ga, w.gb, D::WType::F16, 128, 0, C, gt, R, nullptr);
    };
    auto old_step = [&](Lw& w) { D::kda_step(D::KdaDev{Hh, w.conv, w.A, w.dt, w.onorm}, qkv, f, gt, bb, w.S, w.cs, scr, beta, o, y, flag, R, nullptr); };
    const unsigned it = 200;
    bench_line("kda.small_dense3 fast", R, 0, tm.us([&] { fast_small(next()); }, it), w_small);
    F::set_dense_schedule(F::DenseSchedule::kLane);
    bench_line("kda.small_dense3 lane", R, 0, tm.us([&] { fast_small(next()); }, it), w_small);
    F::set_dense_schedule(F::DenseSchedule::kStaged);
    F::set_dense_tile(8);
    bench_line("kda.small_dense3 tile8", R, 0, tm.us([&] { fast_small(next()); }, it), w_small);
    bench_line("kda.small_dense3 tile8", R, 0, tm.us([&] { fast_small(next()); }, it), w_small);
    F::set_dense_tile(16);
    bench_line("kda.small_dense3 fast", R, 0, tm.us([&] { fast_small(next()); }, it), w_small);
    bench_line("kda.small_dense3 old", R, 0, tm.us([&] { old_small(next()); }, it), w_small);
    bench_line("kda.lowrank_dense2 fast", R, 0, tm.us([&] { fast_lr(next()); }, it), w_lr);
    bench_line("kda.lowrank_dense2 old", R, 0, tm.us([&] { old_lr(next()); }, it), w_lr);
    {  // ABA: baseline, column-quad, shared-prep split and column-quad shared-prep split schedules.
        static constexpr unsigned seq[] = {2U, 128U, 160U, 200U, 232U, 2U, 128U, 160U, 200U, 232U, 2U};
        for (unsigned q : seq) {
            F::set_kda_lanes(q);
            const std::string name = std::string("kda.step fast q") + (q == 2U ? "2" : q == 128U ? "2c" : q == 196U ? "4sp" : q == 200U ? "cqs" : q == 160U ? "2ct" : q == 232U ? "cqst" : "unknown");
            bench_line(name.c_str(), R, 0, tm.us([&] { fast_step(next()); }, it), st);
        }
    }
    F::set_kda_lanes(2);
    bench_line("kda.step fast", R, 0, tm.us([&] { fast_step(next()); }, it), st);
    bench_line("kda.step old", R, 0, tm.us([&] { old_step(next()); }, it), st);
    bench_line("kda.state copy floor", R, 0,
               tm.us([&] { auto& w = next(); F::kda_state_copy_floor(w.S, Hh, R, nullptr); }, it), st);
    for (unsigned ab = 0; ab < 7; ++ab)  // whole layer: fast old fast old fast old fast, every old bracketed (primary numbers)
        if (ab % 2 == 0)
            bench_line("kda.layer_tierA fast", R, 0, tm.us([&] { auto& w = next(); fast_small(w); fast_lr(w); fast_step(w); }, it), w_small + w_lr + st);
        else
            bench_line("kda.layer_tierA old", R, 0, tm.us([&] { auto& w = next(); old_small(w); old_lr(w); old_step(w); }, it), w_small + w_lr + st);
    if (R > 1) {  // the verify form: R positions of one slot (state read/written once)
        float *snap = dalloc<float>(R * SZ);
        bench_line("kda.steps_seq (R pos)", R, 0,
                   tm.us([&] { auto& w = next(); F::kda_steps_seq(D::KdaDev{Hh, w.conv, w.A, w.dt, w.onorm}, qkv, f, gt, bb, w.S, w.cs, y, nullptr, nullptr, flag, R, nullptr); }, it),
                   2.0 * (SZ + CZ) * 4);
        bench_line("kda.steps_seq +snapshots", R, 0,
                   tm.us([&] { auto& w = next(); F::kda_steps_seq(D::KdaDev{Hh, w.conv, w.A, w.dt, w.onorm}, qkv, f, gt, bb, w.S, w.cs, y, snap, nullptr, flag, R, nullptr); }, it),
                   2.0 * (SZ + CZ) * 4 + R * SZ * 4.0);
        bench_line("kda.step x R sequential", R, 0, tm.us([&] {
                       auto& w = next();
                       for (unsigned p = 0; p < R; ++p)
                           F::kda_step(D::KdaDev{Hh, w.conv, w.A, w.dt, w.onorm}, qkv + static_cast<std::size_t>(p) * 3 * C, f + static_cast<std::size_t>(p) * C,
                                       gt + static_cast<std::size_t>(p) * C, bb + p * Hh, w.S, w.cs, y + static_cast<std::size_t>(p) * C, flag, 1, nullptr);
                   }, it), R * 2.0 * (SZ + CZ) * 4);
        fr(snap);
    }
    for (auto& w : lw) {
        for (std::uint16_t* p : {w.fa, w.ga, w.bp, w.fb, w.gb, w.conv, w.onorm}) fr(p);
        for (float* p : {w.A, w.dt, w.S, w.cs}) fr(p);
    }
    for (float* p : {x, qkv, fa, ga, bb, f, gt, y, scr, beta, o}) fr(p);
    fr(flag);
}

void bench_dsa(unsigned R, unsigned ctx) {
    constexpr unsigned Hh = 32, LW = 11;  // 11 kv_b / indexer weight copies
    const unsigned cap = ctx, P = ctx / kKpool;
    // cache copies: enough that the bytes the query path touches (selected latents + pooled keys, all rows) rotate
    // through >= 96 MB (3 x the 32 MB MALL), at least 3
    const double touched = static_cast<double>(R) * (std::min(P, kIdxPools) * kKpool * kKvLora * 2.0 + P * kIdxDim * 4.0);
    const unsigned LC = std::min(96U, std::max(3U, static_cast<unsigned>(96.0 * 1048576.0 / touched) + 1));
    struct Lw {
        std::uint16_t *wk, *gate, *wp, *knw, *knb, *kvb;
        float* ape;
    };
    std::vector<Lw> lw(LW);
    for (unsigned l = 0; l < LW; ++l) {
        auto& w = lw[l];
        w.wk = dalloc<std::uint16_t>(kIdxDim * kHidden);
        w.gate = dalloc<std::uint16_t>(kIdxDim * kHidden);
        w.wp = dalloc<std::uint16_t>(kIdxHeads * kHidden);
        w.knw = dalloc<std::uint16_t>(kIdxDim);
        w.knb = dalloc<std::uint16_t>(kIdxDim);
        w.kvb = dalloc<std::uint16_t>(static_cast<std::size_t>(Hh) * 2 * kMlaDim * kKvLora);
        w.ape = dalloc<float>(kKpool * kIdxDim);
        fill_f16(w.wk, kIdxDim * kHidden, l * 7 + 1);
        fill_f16(w.gate, kIdxDim * kHidden, l * 7 + 2);
        fill_f16(w.wp, kIdxHeads * kHidden, l * 7 + 3);
        fill_f16(w.knw, kIdxDim, l * 7 + 4);
        fill_f16(w.knb, kIdxDim, l * 7 + 5);
        fill_f16(w.kvb, static_cast<std::size_t>(Hh) * 2 * kMlaDim * kKvLora, l * 7 + 6);
        fill_f32(w.ape, kKpool * kIdxDim, l * 7 + 7, 0.1F);
    }
    // caches: LC copies x R slots
    std::vector<std::vector<D::DsaSlot>> slots(LC, std::vector<D::DsaSlot>(R));
    std::vector<D::DsaSlot*> dslots(LC), same_slots(LC);
    for (unsigned c = 0; c < LC; ++c) {
        for (unsigned r = 0; r < R; ++r) {
            DevSlot s;
            s.alloc(cap);
            fill_bf16(s.s.latent, static_cast<std::size_t>(cap) * kKvLora, c * 101 + r * 7 + 1);
            fill_bf16(s.s.ikey, static_cast<std::size_t>(cap) * kIdxDim, c * 101 + r * 7 + 2);
            fill_bf16(s.s.igate, static_cast<std::size_t>(cap) * kIdxDim, c * 101 + r * 7 + 3);
            fill_f32(s.s.pooled, static_cast<std::size_t>(P) * kIdxDim, c * 101 + r * 7 + 4, 1.0F);
            slots[c][r] = s.s;
        }
        dslots[c] = up(slots[c]);
        same_slots[c] = up(std::vector<D::DsaSlot>(R, slots[c][0]));
    }
    unsigned* dt = up(std::vector<unsigned>(R, ctx - 1));
    std::vector<unsigned> same_tt(R);
    for (unsigned r = 0; r < R; ++r) same_tt[r] = ctx - R + r;  // token positions ctx-R+1 .. ctx
    unsigned* same_dt = up(same_tt);
    float *x = dalloc<float>(R * kHidden), *lat = dalloc<float>(R * kKvLora), *key = dalloc<float>(R * kIdxDim), *gs = dalloc<float>(R * kIdxDim);
    float *wraw = dalloc<float>(R * kIdxHeads), *qI = dalloc<float>(R * kIdxHeads * kIdxDim), *q = dalloc<float>(static_cast<std::size_t>(R) * Hh * kMlaDim);
    float* o = dalloc<float>(static_cast<std::size_t>(R) * Hh * kMlaDim);
    fill_f32(x, R * kHidden, 1, 0.5F);
    fill_f32(lat, R * kKvLora, 2, 1.0F);
    fill_f32(qI, R * kIdxHeads * kIdxDim, 3, 1.0F);
    fill_f32(q, static_cast<std::size_t>(R) * Hh * kMlaDim, 4, 1.0F);
    std::uint32_t *keys = dalloc<std::uint32_t>(static_cast<std::size_t>(R) * P), *hist0 = zalloc<std::uint32_t>(R * 256);
    std::uint32_t *tok = dalloc<std::uint32_t>(static_cast<std::size_t>(R) * F::kTokStride), *n = dalloc<std::uint32_t>(R);
    F::MlaWork mw{dalloc<float>(static_cast<std::size_t>(R) * Hh * kKvLora), dalloc<std::uint32_t>(R * Hh),
                  dalloc<float>(static_cast<std::size_t>(R) * Hh * F::kTokStride), dalloc<float>(static_cast<std::size_t>(R) * Hh * F::kTokStride),
                  dalloc<float>(static_cast<std::size_t>(R) * Hh * kKvLora)};
    // old path workspace (per row)
    float *owh = dalloc<float>(kIdxHeads), *osc = dalloc<float>(P + 1), *oqt = dalloc<float>(Hh * kKvLora);
    std::uint32_t *owork = dalloc<std::uint32_t>(P + 1);
    unsigned* flag = new_flag();
    hok(hipDeviceSynchronize(), "s");
    Timer tm;
    unsigned cw = 0, cc = 0;
    auto nw = [&]() -> Lw& { cw = (cw + 1) % LW; return lw[cw]; };
    auto nc = [&]() { cc = (cc + 1) % LC; return cc; };
    const unsigned n_tok = std::min(P, kIdxPools) * kKpool + (ctx % kKpool);
    const double b_dense = (2.0 * kIdxDim + kIdxHeads) * kHidden * 2, b_score = static_cast<double>(R) * P * kIdxDim * 4;
    const double b_kvb_half = static_cast<double>(Hh) * kMlaDim * kKvLora * 2, b_lat = static_cast<double>(R) * n_tok * kKvLora * 2;
    auto fast_dense = [&](Lw& w) {
        const F::DenseJob j3[3] = {{w.wk, x, key, kIdxDim, kHidden, kHidden, kIdxDim}, {w.gate, x, gs, kIdxDim, kHidden, kHidden, kIdxDim},
                                   {w.wp, x, wraw, kIdxHeads, kHidden, kHidden, kIdxHeads}};
        F::dense_f16(j3, 3, R, nullptr);
    };
    auto rows_of = [&](unsigned c) { return F::DsaRows{dslots[c], dt, R}; };
    auto fast_store = [&](Lw& w, unsigned c) { F::dsa_store(rows_of(c), lat, key, gs, D::IndexerDev{w.wk, w.gate, w.wp, w.knw, w.knb, w.ape}, hist0, flag, nullptr); };
    auto fast_score = [&](unsigned c) { F::indexer_score(rows_of(c), qI, wraw, keys, P, hist0, P, nullptr); };
    auto fast_select = [&](unsigned c) { F::indexer_select(rows_of(c), keys, P, hist0, tok, n, nullptr); };
    const unsigned it = 50;
    // prime one selection per cache copy so that the MLA stages have token lists
    for (unsigned c = 0; c < LC; ++c) {
        fast_dense(lw[0]);
        fast_store(lw[0], c);
        fast_score(c);
        fast_select(c);
    }
    bench_line("dsa.dense3 (wk|gate|wproj) fast", R, ctx, tm.us([&] { fast_dense(nw()); }, it), b_dense);
    bench_line("dsa.store fast", R, ctx, tm.us([&] { fast_store(nw(), nc()); }, it), R * (kKvLora + 2 * kIdxDim) * 2.0);
    F::set_indexer_score_rows(1);
    bench_line("dsa.idx_score fast", R, ctx, tm.us([&] { fast_score(nc()); }, it), b_score);
    double same_score_bytes = 0.0;
    for (unsigned t : same_tt) same_score_bytes += static_cast<double>((t + 1) / kKpool) * kIdxDim * 4.0;
    for (unsigned same_slot : {0U, 1U}) {
        for (unsigned rg : {1U, 2U, 3U, 4U, 5U, 1U, 2U, 3U, 4U, 5U, 1U}) {
            F::set_indexer_score_rows(rg);
            cc = 0;
            hok(hipMemset(hist0, 0, static_cast<std::size_t>(R) * 256 * sizeof(std::uint32_t)), "bench score hist");
            const std::string name = std::string(same_slot != 0 ? "dsa.idx_score same-slot rg" : "dsa.idx_score rg") + std::to_string(rg);
            bench_line(name.c_str(), R, ctx, tm.us([&] {
                const unsigned c = nc();
                const F::DsaRows score_rows = same_slot != 0 ? F::DsaRows{same_slots[c], same_dt, R} : rows_of(c);
                F::indexer_score(score_rows, qI, wraw, keys, P, hist0, P, nullptr);
            }, it), same_slot != 0 ? same_score_bytes : b_score);
        }
    }
    F::set_indexer_score_rows(1);
    fast_store(lw[0], 0);  // re-prime: the score timing accumulated hist0; select needs keys + hist of one pass
    fast_score(0);
    bench_line("dsa.idx_select fast", R, ctx, tm.us([&] { fast_select(nc()); }, it), static_cast<double>(R) * P * 4);
    bench_line("mla.absorb fast", R, ctx, tm.us([&] { F::mla_absorb(q, nw().kvb, Hh, mw.qt, mw.mkey, R, nullptr); }, it), b_kvb_half);
    bench_line("mla.scores fast", R, ctx, tm.us([&] { F::mla_scores(rows_of(nc()), tok, n, mw.qt, Hh, mw.sc, mw.mkey, nullptr); }, it), b_lat);
    F::set_scores_schedule(F::DenseSchedule::kStaged);
    bench_line("mla.scores staged", R, ctx, tm.us([&] { F::mla_scores(rows_of(nc()), tok, n, mw.qt, Hh, mw.sc, mw.mkey, nullptr); }, it), b_lat);
    F::set_scores_schedule(F::DenseSchedule::kLane);
    F::mla_absorb(q, lw[0].kvb, Hh, mw.qt, mw.mkey, R, nullptr);  // re-prime: mkey reset + one max pass
    F::mla_scores(rows_of(0), tok, n, mw.qt, Hh, mw.sc, mw.mkey, nullptr);
    bench_line("mla.softmax fast", R, ctx, tm.us([&] { F::mla_softmax(n, mw.sc, mw.mkey, Hh, mw.p, R, nullptr); }, it), 2.0 * R * Hh * n_tok * 4);
    for (unsigned v : {0U, 1U, 2U, 0U, 1U, 2U, 0U}) {
        F::set_mla_pv_variant(v);
        const char* name = v == 0 ? "mla.pv fast" : (v == 1 ? "mla.pv v1" : "mla.pv v2");
        bench_line(name, R, ctx, tm.us([&] { F::mla_pv(rows_of(nc()), tok, n, mw.p, Hh, mw.ob, nullptr); }, it), b_lat);
    }
    bench_line("mla.out fast", R, ctx, tm.us([&] { F::mla_out(mw.ob, nw().kvb, Hh, o, R, nullptr); }, it), b_kvb_half);
    F::set_dense_schedule(F::DenseSchedule::kLane);
    bench_line("mla.out lane", R, ctx, tm.us([&] { F::mla_out(mw.ob, nw().kvb, Hh, o, R, nullptr); }, it), b_kvb_half);
    bench_line("dsa.dense3 lane", R, ctx, tm.us([&] { fast_dense(nw()); }, it), b_dense);
    F::set_dense_schedule(F::DenseSchedule::kStaged);
    F::set_dense_tile(8);
    bench_line("dsa.dense3 tile8", R, ctx, tm.us([&] { fast_dense(nw()); }, it), b_dense);
    bench_line("mla.out tile8", R, ctx, tm.us([&] { F::mla_out(mw.ob, nw().kvb, Hh, o, R, nullptr); }, it), b_kvb_half);
    F::set_dense_tile(16);
    const double b_layer = b_dense + b_score + 2 * b_kvb_half + b_lat;
    auto fast_layer = [&] {
        auto& w = nw();
        const unsigned c = nc();
        fast_dense(w);
        fast_store(w, c);
        fast_score(c);
        fast_select(c);
        F::mla_attention(rows_of(c), tok, n, q, w.kvb, Hh, mw, o, nullptr);
    };
    // old path: dsa_append (rows) + per-row indexer_select + mla_attention
    auto old_layer = [&] {
        auto& w = nw();
        const unsigned c = nc();
        const D::IndexerDev dw{w.wk, w.gate, w.wp, w.knw, w.knb, w.ape};
        D::dsa_append(dslots[c], dt, lat, x, dw, key, gs, flag, R, nullptr);
        for (unsigned r = 0; r < R; ++r) {
            D::indexer_select(slots[c][r], ctx - 1, qI + static_cast<std::size_t>(r) * kIdxHeads * kIdxDim, x + static_cast<std::size_t>(r) * kHidden, dw,
                              owh, osc, owork, tok, n, nullptr);
            D::mla_attention(slots[c][r], tok, n, q + static_cast<std::size_t>(r) * Hh * kMlaDim, w.kvb, Hh, oqt, o, nullptr);
        }
    };
    const unsigned it_old = ctx >= 32768 ? 3 : 10;
    for (unsigned ab = 0; ab < 7; ++ab)  // whole layer: fast old fast old fast old fast (primary numbers)
        if (ab % 2 == 0)
            bench_line("dsa.layer_tierA fast", R, ctx, tm.us(fast_layer, it), b_layer);
        else
            bench_line("dsa.layer_tierA old", R, ctx, tm.us(old_layer, it_old), b_layer);
    // cleanup
    for (auto& w : lw) {
        for (std::uint16_t* p : {w.wk, w.gate, w.wp, w.knw, w.knb, w.kvb}) fr(p);
        fr(w.ape);
    }
    for (unsigned c = 0; c < LC; ++c) {
        for (auto& s : slots[c]) {
            fr(s.latent);
            fr(s.ikey);
            fr(s.igate);
            fr(s.pooled);
        }
        fr(dslots[c]);
        fr(same_slots[c]);
    }
    for (float* p : {x, lat, key, gs, wraw, qI, q, o, mw.qt, mw.sc, mw.p, mw.ob, owh, osc, oqt}) fr(p);
    for (std::uint32_t* p : {keys, hist0, tok, n, mw.mkey, owork}) fr(p);
    fr(dt);
    fr(same_dt);
    fr(flag);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s CASES_DIR [--bench]\n", argv[0]);
        return 2;
    }
    const bool bench_dsa_only = argc > 2 && std::string(argv[2]) == "--bench-dsa";  // DSA / MLA benches only
    const bool bench = (argc > 2 && std::string(argv[2]) == "--bench") || bench_dsa_only;
    hipDeviceProp_t prop{};
    hok(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("device=%s arch=%s CUs=%d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    std::mt19937 g(20261024);
    const auto t0 = std::chrono::steady_clock::now();
    if (argc > 2 && std::string(argv[2]) == "--kda") {  // KDA gate including shared prep, then the KDA bench only
        for (unsigned q : {2U, 160U, 232U}) {  // 2-barrier-tree forms of 2c / cqs (other forms gated in earlier measurements)
            F::set_kda_lanes(q);
            std::printf("KDA step lanes per column: %s\n", q == 160U ? "2ct" : q == 232U ? "cqst" : q == 200U ? "cqs" : q == 128U ? "2c" : q == 194U ? "2sp" : q == 196U ? "4sp" : q == 68U ? "4s" : "2");
            test_kda(argv[1], g);
        }
        F::set_kda_lanes(2);
        std::printf("checks: %d failures: %d\n", checks, failures);
        if (failures == 0)
            for (unsigned R : {1U, 2U, 4U, 8U}) bench_kda(R);
        return failures ? 1 : 0;
    }
    if (argc > 2 && std::string(argv[2]) == "--idx") {  // indexer score variants: DSA gate (rg 1..5 inside), direct gate, DSA bench 8K/32K
        test_dsa(argv[1], g);
        test_large_p(argv[1], g);
        test_idx_score_rows(g);
        std::printf("checks: %d failures: %d\n", checks, failures);
        if (failures == 0)
            for (unsigned ctx : {8192U, 32768U})
                for (unsigned R : {1U, 2U, 4U, 8U}) bench_dsa(R, ctx);
        return failures ? 1 : 0;
    }
    test_f16_decode();
    test_dense(argv[1], g);
    for (unsigned q : {2U, 128U}) {  // KDA schedules: default + column quad (128); split (+64) gated in measurement, 4..56 in attention repeated measurements
        F::set_kda_lanes(q);
        std::printf("KDA step lanes per column: %s\n", q == 128U ? "2c" : "2");
        test_kda(argv[1], g);
    }
    F::set_kda_lanes(2);
    for (unsigned v : {0U, 1U, 2U}) {
        F::set_mla_pv_variant(v);
        std::printf("MLA PV variant: %u\n", v);
        test_dsa(argv[1], g);
        test_large_p(argv[1], g);
    }
    F::set_mla_pv_variant(0);
    test_idx_score_rows(g);
    test_pv_direct(g);
    test_select_keys(g);
    test_select_skip(g);
    test_dense_edges(g);
    test_scores_direct(g);
    std::printf("checks: %d failures: %d (%.1f s)\n", checks, failures,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (bench && failures == 0) {
        if (!bench_dsa_only)
            for (unsigned R : {1U, 2U, 4U, 8U}) bench_kda(R);
        for (unsigned ctx : {1024U, 8192U, 32768U})
            for (unsigned R : {1U, 2U, 4U, 8U}) bench_dsa(R, ctx);
    }
    return failures ? 1 : 0;
}
