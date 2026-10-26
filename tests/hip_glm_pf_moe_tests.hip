// GLM prefill MoE grouped core (src/glm/prefill/glm_pf_moe.*) vs the per-expert library chain, bit for bit.
// Reference per expert e (rows in ascending token order): strix_gemm_pinned(gate_e / up_e, X_e, splits = 1) ->
// dev::swiglu (R15) -> strix_gemm_pinned(down_e, a, 1) -> scatter into eo[t][k] -> dev::moe_combine (R16) + shared.
// Synthetic weights: E = 288 experts, K4 mul1, random trellis bits, suh +-1, svh +-0.02 (F16).
// Gate: T in {1, 7, 64, 1000, 2048} x {uniform, skewed} routes (skewed: a hot expert on every row, experts with exactly
// 1..8, 15-17, 31-33, 63-65, 127-129, 300, 301 rows where T allows). Compared bytewise: out [T][4096] and every route's
// expert output eo [T][8][4096]; finite checks; workspace and outputs poisoned with 0xff; a second run must be bitwise
// equal; workspace bytes <= the bound. Timing (only after a clean gate): T 1024 / 2048 / 4096 uniform routes, grouped
// GPU ms (marker start -> combine), whole-call wall ms and the per-expert library chain, TOPS at 25.17 MFLOP per routed
// row-expert; the timed outputs are gated bitwise too. Exit 0 only if every check passes.
// usage: hip_glm_pf_moe_tests [--no-timing]
#include "glm/kernels/glm_tier_a.hpp"
#include "glm/prefill/glm_pf_moe.hpp"
#include "strix_trellis.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

using namespace strix::glm;
namespace D = strix::glm::dev;
namespace PF = strix::glm::pf;

namespace {

constexpr unsigned kE = 288, kTop = 8, kI = 1024, kH = 4096, kBits = 4;

void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FAIL hip %s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
void sok(strix_status s, const char* w) {
    if (s != STRIX_SUCCESS) {
        std::fprintf(stderr, "FAIL strix %s: %s\n", w, strix_status_string(s));
        std::exit(3);
    }
}

struct Rng {  // splitmix64
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uni() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    unsigned below(unsigned n) { return static_cast<unsigned>(next() % n); }
    float normal() {
        const double u1 = uni() + 1e-300, u2 = uni();
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
};

std::uint16_t f16(float v) {
    const _Float16 h = static_cast<_Float16>(v);
    std::uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}

struct Mat {
    strix_weights w{};
    PF::MoeExl3 p{};
};

Mat make_w(strix_context c, strix_stream* st, unsigned N, unsigned K, Rng& g, bool check_span) {
    std::vector<std::uint64_t> tr64(static_cast<std::size_t>(K) * N * kBits / 64);
    for (auto& v : tr64) v = g.next();
    std::vector<std::uint16_t> suh(K), svh(N);
    for (auto& v : suh) v = f16((g.next() & 1) ? 1.0F : -1.0F);
    for (auto& v : svh) v = f16((g.next() & 1) ? 0.02F : -0.02F);
    strix_exl3_parts_desc p{};
    p.struct_size = sizeof(p);
    p.abi_version = STRIX_TRELLIS_ABI_VERSION;
    p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    p.bits_x2 = 2 * kBits;
    p.rows = N;
    p.columns = K;
    p.codebook = STRIX_CODEBOOK_MUL1;
    p.marker = STRIX_EXL3_MARKER_MUL1;
    p.trellis = {tr64.data(), tr64.size() * 8};
    p.suh = {suh.data(), suh.size() * 2};
    p.svh = {svh.data(), svh.size() * 2};
    Mat m;
    sok(strix_weights_prepare_parts(c, &p, st, &m.w), "prepare");
    m.p = PF::moe_exl3_from_weights(m.w, N, K, kBits);  // throws on a span size mismatch
    if (check_span) {  // the derived pointers hold exactly the uploaded parts
        std::vector<unsigned char> tb(tr64.size() * 8);
        std::vector<std::uint16_t> sb(K), vb(N);
        hok(hipMemcpy(tb.data(), m.p.trellis, tb.size(), hipMemcpyDeviceToHost), "span");
        hok(hipMemcpy(sb.data(), m.p.suh, K * 2, hipMemcpyDeviceToHost), "span");
        hok(hipMemcpy(vb.data(), m.p.svh, N * 2, hipMemcpyDeviceToHost), "span");
        if (std::memcmp(tb.data(), tr64.data(), tb.size()) != 0 || std::memcmp(sb.data(), suh.data(), K * 2) != 0 ||
            std::memcmp(vb.data(), svh.data(), N * 2) != 0) {
            std::fprintf(stderr, "FAIL moe_exl3_from_weights: span pointers do not hold the uploaded parts\n");
            std::exit(1);
        }
    }
    return m;
}

strix_matmul_desc desc(strix_weights w, unsigned N, unsigned K, const float* x, float* y, unsigned rows, void* ws, std::uint64_t wsb) {
    strix_matmul_desc d;
    strix_matmul_desc_init(&d);
    d.weights = w;
    d.input = {sizeof(strix_tensor_desc), STRIX_TRELLIS_ABI_VERSION, const_cast<float*>(x), static_cast<std::uint64_t>(rows) * K * 4,
               STRIX_DTYPE_F32, 2, {rows, K}, {K, 1}, {}};
    d.output = {sizeof(strix_tensor_desc), STRIX_TRELLIS_ABI_VERSION, y, static_cast<std::uint64_t>(rows) * N * 4, STRIX_DTYPE_F32, 2,
                {rows, N}, {N, 1}, {}};
    d.workspace = ws;
    d.workspace_bytes = wsb;
    return d;
}

__global__ void gather_kernel(const float* x, const std::uint32_t* rows, float* xg, unsigned K) {
    const unsigned r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i < K) xg[static_cast<std::size_t>(r) * K + i] = x[static_cast<std::size_t>(rows[r]) * K + i];
}
__global__ void scatter_kernel(const float* y, const std::uint32_t* dst, float* eo) {
    const unsigned r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i < kH) eo[static_cast<std::size_t>(dst[r]) * kH + i] = y[static_cast<std::size_t>(r) * kH + i];
}

struct Routes {
    unsigned T = 0;
    std::vector<std::uint32_t> ids;  // [T][8] ascending per row
    std::vector<float> w;
};

// uniform: 8 distinct uniform experts per row. skewed: expert 0 on every row, expert 1 on ~45 % of the rows, experts
// 2.. with exact row counts (where T and the free slots allow), the rest Zipf-like over experts 40..287.
Routes make_routes(unsigned T, bool skewed, Rng& g, std::string& note) {
    Routes r;
    r.T = T;
    std::vector<std::vector<std::uint32_t>> row(T);
    auto has = [&](unsigned t, unsigned e) { return std::find(row[t].begin(), row[t].end(), e) != row[t].end(); };
    if (skewed) {
        for (unsigned t = 0; t < T; ++t) {
            row[t].push_back(0);
            if (g.uni() < 0.45) row[t].push_back(1);
        }
        const unsigned counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300, 301};
        unsigned e = 2, ptr = 0;
        note = "exact:";
        for (const unsigned c : counts) {
            if (c > T) continue;
            unsigned free_rows = 0;
            for (unsigned t = 0; t < T; ++t) free_rows += row[t].size() < kTop ? 1U : 0U;
            if (free_rows < c) continue;
            unsigned picked = 0;
            for (unsigned s = 0; s < T && picked < c; ++s) {
                const unsigned t = (ptr + s) % T;
                if (row[t].size() < kTop) {
                    row[t].push_back(e);
                    ++picked;
                }
            }
            ptr = (ptr + 37U * c + 11U) % T;
            note += " e" + std::to_string(e) + "=" + std::to_string(c);
            ++e;
        }
        std::vector<double> cdf;
        double acc = 0;
        for (unsigned x = 40; x < kE; ++x) {
            acc += 1.0 / std::pow(static_cast<double>(x - 39), 1.1);
            cdf.push_back(acc);
        }
        for (unsigned t = 0; t < T; ++t)
            while (row[t].size() < kTop) {
                const double u = g.uni() * acc;
                const unsigned x = 40U + static_cast<unsigned>(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
                const unsigned ex = std::min(x, kE - 1);
                if (!has(t, ex)) row[t].push_back(ex);
            }
    } else {
        for (unsigned t = 0; t < T; ++t)
            while (row[t].size() < kTop) {
                const unsigned ex = g.below(kE);
                if (!has(t, ex)) row[t].push_back(ex);
            }
    }
    r.ids.resize(std::size_t{T} * kTop);
    r.w.resize(std::size_t{T} * kTop);
    for (unsigned t = 0; t < T; ++t) {
        std::sort(row[t].begin(), row[t].end());
        for (unsigned k = 0; k < kTop; ++k) {
            r.ids[std::size_t{t} * kTop + k] = row[t][k];
            r.w[std::size_t{t} * kTop + k] = static_cast<float>(0.02 + 0.58 * g.uni());
        }
    }
    return r;
}

template <class T>
T* dmalloc(std::size_t n) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    return p;
}

struct Model {
    strix_context c{};
    hipStream_t hs{};
    strix_stream st{};
    std::vector<Mat> wg, wu, wd;
    std::vector<PF::MoeExpert> experts;
};

// The per-expert library chain. Returns GPU ms (events around the whole chain).
float run_reference(Model& m, const Routes& r, const float* x, const float* shared, const float* wdev, float* eo, float* y) {
    const unsigned T = r.T;
    std::vector<std::vector<std::uint32_t>> rows(kE), dst(kE);
    for (unsigned t = 0; t < T; ++t)
        for (unsigned k = 0; k < kTop; ++k) {
            const unsigned e = r.ids[std::size_t{t} * kTop + k];
            rows[e].push_back(t);
            dst[e].push_back(t * kTop + k);
        }
    unsigned gmax = 0;
    std::vector<unsigned> off(kE + 1, 0);
    std::vector<std::uint32_t> flat_rows, flat_dst;
    for (unsigned e = 0; e < kE; ++e) {
        gmax = std::max<unsigned>(gmax, static_cast<unsigned>(rows[e].size()));
        off[e + 1] = off[e] + static_cast<unsigned>(rows[e].size());
        flat_rows.insert(flat_rows.end(), rows[e].begin(), rows[e].end());
        flat_dst.insert(flat_dst.end(), dst[e].begin(), dst[e].end());
    }
    const std::size_t P = flat_rows.size();
    auto* drows = dmalloc<std::uint32_t>(P);
    auto* ddst = dmalloc<std::uint32_t>(P);
    hok(hipMemcpy(drows, flat_rows.data(), P * 4, hipMemcpyHostToDevice), "c");
    hok(hipMemcpy(ddst, flat_dst.data(), P * 4, hipMemcpyHostToDevice), "c");
    auto* xg = dmalloc<float>(P * kH);
    auto* gg = dmalloc<float>(P * kI);
    auto* uu = dmalloc<float>(P * kI);
    auto* aa = dmalloc<float>(P * kI);
    auto* yd = dmalloc<float>(P * kH);
    std::uint64_t wsb = 0;
    {
        std::uint64_t a = 0, b = 0;
        auto d1 = desc(m.wg[0].w, kI, kH, xg, gg, gmax, nullptr, 0);
        auto d2 = desc(m.wd[0].w, kH, kI, aa, yd, gmax, nullptr, 0);
        sok(strix_gemm_pinned_workspace_bytes(m.c, &d1, 1, &a), "ws");
        sok(strix_gemm_pinned_workspace_bytes(m.c, &d2, 1, &b), "ws");
        wsb = std::max(a, b);
    }
    void* ws = dmalloc<unsigned char>(wsb);
    hipEvent_t e0, e1;
    hok(hipEventCreate(&e0), "e");
    hok(hipEventCreate(&e1), "e");
    hok(hipEventRecord(e0, m.hs), "e");
    for (unsigned e = 0; e < kE; ++e) {
        const unsigned M = off[e + 1] - off[e];
        if (!M) continue;
        float *xe = xg + std::size_t{off[e]} * kH, *ge = gg + std::size_t{off[e]} * kI, *ue = uu + std::size_t{off[e]} * kI;
        float *ae = aa + std::size_t{off[e]} * kI, *ye = yd + std::size_t{off[e]} * kH;
        gather_kernel<<<dim3(kH / 256, M), 256, 0, m.hs>>>(x, drows + off[e], xe, kH);
        hok(hipGetLastError(), "gather");
        auto dg = desc(m.wg[e].w, kI, kH, xe, ge, M, ws, wsb);
        sok(strix_gemm_pinned(m.c, &dg, 1, &m.st), "gate");
        auto du = desc(m.wu[e].w, kI, kH, xe, ue, M, ws, wsb);
        sok(strix_gemm_pinned(m.c, &du, 1, &m.st), "up");
        D::swiglu(ge, ue, ae, std::size_t{M} * kI, m.hs);
        auto dd = desc(m.wd[e].w, kH, kI, ae, ye, M, ws, wsb);
        sok(strix_gemm_pinned(m.c, &dd, 1, &m.st), "down");
        scatter_kernel<<<dim3(kH / 256, M), 256, 0, m.hs>>>(ye, ddst + off[e], eo);
        hok(hipGetLastError(), "scatter");
    }
    D::moe_combine(wdev, eo, shared, y, T, m.hs);
    hok(hipEventRecord(e1, m.hs), "e");
    hok(hipEventSynchronize(e1), "s");
    float ms = 0;
    hok(hipEventElapsedTime(&ms, e0, e1), "t");
    hok(hipEventDestroy(e0), "e");
    hok(hipEventDestroy(e1), "e");
    for (void* p : {static_cast<void*>(drows), static_cast<void*>(ddst), static_cast<void*>(xg), static_cast<void*>(gg),
                    static_cast<void*>(uu), static_cast<void*>(aa), static_cast<void*>(yd), ws})
        hok(hipFree(p), "free");
    return ms;
}

struct Timer {
    hipEvent_t start{}, end{};
    hipEvent_t st[6]{};  // start, gather, gate_up, finish, down, combine (stage split of the best run)
    float stage_best[5] = {1e30F, 1e30F, 1e30F, 1e30F, 1e30F};
};
constexpr const char* kStageLabel[6] = {"start", "gather", "gate_up", "finish", "down", "combine"};
void marker(const char* label, hipStream_t s, void* user) {
    auto* t = static_cast<Timer*>(user);
    if (std::strcmp(label, "start") == 0) hok(hipEventRecord(t->start, s), "e");
    if (std::strcmp(label, "combine") == 0) hok(hipEventRecord(t->end, s), "e");
    for (int i = 0; i < 6; ++i)
        if (t->st[i] != nullptr && std::strcmp(label, kStageLabel[i]) == 0) hok(hipEventRecord(t->st[i], s), "e");
}

struct CaseResult {
    bool pass = false;
    float ref_ms = 0, gpu_ms = 0, wall_ms = 0;
};

std::size_t count_diff(const std::vector<float>& a, const std::vector<float>& b, std::size_t& first) {
    std::size_t n = 0;
    first = ~std::size_t{0};
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            if (n == 0) first = i;
            ++n;
        }
    return n;
}
bool all_finite(const std::vector<float>& v) {
    for (const float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// One case: reference chain, grouped core twice (poisoned), bytewise comparisons. reps > 0: timing repetitions.
CaseResult run_case(Model& m, const Routes& r, const char* label, Rng& g, int reps) {
    CaseResult res;
    const unsigned T = r.T;
    const std::size_t TH = std::size_t{T} * kH, T8H = std::size_t{T} * kTop * kH;
    std::vector<float> xh(TH), sh(TH);
    for (auto& v : xh) v = g.normal();
    for (auto& v : sh) v = g.normal();
    auto* x = dmalloc<float>(TH);
    auto* shared = dmalloc<float>(TH);
    auto* wdev = dmalloc<float>(r.w.size());
    hok(hipMemcpy(x, xh.data(), TH * 4, hipMemcpyHostToDevice), "c");
    hok(hipMemcpy(shared, sh.data(), TH * 4, hipMemcpyHostToDevice), "c");
    hok(hipMemcpy(wdev, r.w.data(), r.w.size() * 4, hipMemcpyHostToDevice), "c");
    auto* eo_ref = dmalloc<float>(T8H);
    auto* y_ref = dmalloc<float>(TH);
    auto* eo = dmalloc<float>(T8H);
    auto* y = dmalloc<float>(TH);
    hok(hipMemset(eo_ref, 0xff, T8H * 4), "poison");
    hok(hipMemset(y_ref, 0xff, TH * 4), "poison");
    // group statistics
    std::vector<unsigned> cnt(kE, 0);
    for (const auto e : r.ids) ++cnt[e];
    unsigned groups = 0, gmin = ~0U, gmax = 0, small = 0;
    for (const unsigned c : cnt)
        if (c) {
            ++groups;
            gmin = std::min(gmin, c);
            gmax = std::max(gmax, c);
            small += c <= 8 ? 1U : 0U;
        }
    res.ref_ms = run_reference(m, r, x, shared, wdev, eo_ref, y_ref);
    std::vector<float> yr(TH), er(T8H);
    hok(hipMemcpy(yr.data(), y_ref, TH * 4, hipMemcpyDeviceToHost), "d");
    hok(hipMemcpy(er.data(), eo_ref, T8H * 4, hipMemcpyDeviceToHost), "d");
    PF::PrefillMoeView v;
    v.rows = T;
    v.x = x;
    v.ids = r.ids.data();
    v.weights = r.w.data();
    v.experts = m.experts.data();
    v.bits_gu = kBits;
    v.bits_dn = kBits;
    v.shared = shared;
    v.out = y;
    v.eo_debug = eo;
    const std::size_t wsb = PF::prefill_moe_workspace_bytes(v), bound = PF::prefill_moe_workspace_bound(T);
    bool ok = wsb <= bound;
    auto* ws = dmalloc<unsigned char>(wsb);
    Timer tm;
    hok(hipEventCreate(&tm.start), "e");
    hok(hipEventCreate(&tm.end), "e");
    for (auto& e : tm.st) hok(hipEventCreate(&e), "e");
    PF::set_prefill_moe_marker(marker, &tm);
    std::vector<float> yo(TH), eg(T8H);
    float best_gpu = 1e30F, best_wall = 1e30F;
    const int runs = 2 + (reps > 0 ? reps : 0);
    for (int run = 0; run < runs; ++run) {
        if (run < 2) {
            hok(hipMemset(ws, 0xff, wsb), "poison");
            hok(hipMemset(eo, 0xff, T8H * 4), "poison");
            hok(hipMemset(y, 0xff, TH * 4), "poison");
            v.eo_debug = eo;
        } else {
            v.eo_debug = nullptr;  // production form for timing
        }
        hok(hipDeviceSynchronize(), "sync");
        const auto w0 = std::chrono::steady_clock::now();
        try {
            PF::enqueue_prefill_moe_glm(v, ws, wsb, m.hs);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "FAIL %s: enqueue threw: %s\n", label, ex.what());
            std::exit(1);
        }
        hok(hipStreamSynchronize(m.hs), "sync");
        const auto w1 = std::chrono::steady_clock::now();
        float gms = 0;
        hok(hipEventElapsedTime(&gms, tm.start, tm.end), "t");
        if (run >= 2) {  // production-form repetitions only (eo_debug off)
            if (gms < best_gpu)
                for (int i = 0; i < 5; ++i) hok(hipEventElapsedTime(&tm.stage_best[i], tm.st[i], tm.st[i + 1]), "t");
            best_gpu = std::min(best_gpu, gms);
            best_wall = std::min(best_wall, static_cast<float>(std::chrono::duration<double, std::milli>(w1 - w0).count()));
        }
        hok(hipMemcpy(yo.data(), y, TH * 4, hipMemcpyDeviceToHost), "d");
        if (run < 2) hok(hipMemcpy(eg.data(), eo, T8H * 4, hipMemcpyDeviceToHost), "d");
        std::size_t f_y = 0, f_e = 0;
        const std::size_t dy = count_diff(yr, yo, f_y);
        const std::size_t de = run < 2 ? count_diff(er, eg, f_e) : 0;
        const bool fin = all_finite(yo) && all_finite(yr) && (run >= 2 || all_finite(eg));
        if (dy || de || !fin) {
            ok = false;
            std::printf("  %s run %d: out mismatches %zu (first %zu: ref %08x got %08x), eo mismatches %zu (first %zu), finite %s\n", label,
                        run, dy, dy ? f_y : 0, dy ? *reinterpret_cast<const std::uint32_t*>(&yr[f_y]) : 0U,
                        dy ? *reinterpret_cast<const std::uint32_t*>(&yo[f_y]) : 0U, de, de ? f_e : 0, fin ? "yes" : "NO");
            std::printf("FAIL %s: aborting at the first mismatch / non-finite result (no timing)\n", label);
            std::exit(1);
        }
    }
    PF::set_prefill_moe_marker(nullptr, nullptr);
    if (reps > 0)
        std::printf("STAGES %s T=%u (best run, ms): gather+transform %.3f | gate/up WMMA %.3f | finish (R15 + down transform) %.3f | down WMMA %.3f | combine %.3f\n",
                    label, T, tm.stage_best[0], tm.stage_best[1], tm.stage_best[2], tm.stage_best[3], tm.stage_best[4]);
    res.gpu_ms = best_gpu;
    res.wall_ms = best_wall;
    res.pass = ok;
    std::printf("%s %s T=%u groups=%u rows/group min %u max %u, groups<=8 rows %u, ws %.1f MB (bound %.1f MB): out+eo bitwise x2 %s\n",
                ok ? "PASS" : "FAIL", label, T, groups, gmin, gmax, small, static_cast<double>(wsb) / 1048576.0, static_cast<double>(bound) / 1048576.0, ok ? "yes" : "NO");
    hok(hipEventDestroy(tm.start), "e");
    hok(hipEventDestroy(tm.end), "e");
    for (void* p : {static_cast<void*>(x), static_cast<void*>(shared), static_cast<void*>(wdev), static_cast<void*>(eo_ref),
                    static_cast<void*>(y_ref), static_cast<void*>(eo), static_cast<void*>(y), static_cast<void*>(ws)})
        hok(hipFree(p), "free");
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    const bool timing = !(argc > 1 && std::strcmp(argv[1], "--no-timing") == 0);
    Model m;
    strix_context_desc cd{};
    cd.struct_size = sizeof(cd);
    cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    sok(strix_context_create(&cd, &m.c), "ctx");
    hok(hipStreamCreate(&m.hs), "stream");
    m.st = {sizeof(strix_stream), STRIX_TRELLIS_ABI_VERSION, m.hs, {}};
    Rng g{0x5eed61e5ULL};
    m.wg.resize(kE);
    m.wu.resize(kE);
    m.wd.resize(kE);
    m.experts.resize(kE);
    for (unsigned e = 0; e < kE; ++e) {
        const bool check = e == 0 || e == 1 || e == 143 || e == kE - 1;
        m.wg[e] = make_w(m.c, &m.st, kI, kH, g, check);
        m.wu[e] = make_w(m.c, &m.st, kI, kH, g, check);
        m.wd[e] = make_w(m.c, &m.st, kH, kI, g, check);
        m.experts[e] = {m.wg[e].p, m.wu[e].p, m.wd[e].p};
    }
    std::printf("weights: %u experts x (gate, up [1024 x 4096], down [4096 x 1024]) K%u mul1; span pointers checked\n", kE, kBits);
    // fail-closed API checks
    bool all = true;
    {
        PF::PrefillMoeView bad;
        bool threw = false;
        try {
            (void)PF::prefill_moe_workspace_bytes(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        std::vector<std::uint32_t> ids(kTop, kE);  // out of range
        std::vector<float> w(kTop, 1.0F);
        float* dummy = dmalloc<float>(kH);
        PF::PrefillMoeView v2;
        v2.rows = 1;
        v2.x = dummy;
        v2.shared = dummy;
        v2.out = dummy;
        v2.ids = ids.data();
        v2.weights = w.data();
        v2.experts = m.experts.data();
        bool threw2 = false;
        try {
            (void)PF::prefill_moe_workspace_bytes(v2);
        } catch (const std::invalid_argument&) {
            threw2 = true;
        }
        hok(hipFree(dummy), "free");
        std::printf("%s API rejects empty view / out-of-range id\n", threw && threw2 ? "PASS" : "FAIL");
        all = all && threw && threw2;
    }
    for (const unsigned T : {1U, 7U, 64U, 1000U, 2048U})
        for (const bool skewed : {false, true}) {
            std::string note;
            const Routes r = make_routes(T, skewed, g, note);
            const std::string label = std::string(skewed ? "skewed" : "uniform") + " T" + std::to_string(T);
            const CaseResult c = run_case(m, r, label.c_str(), g, 0);
            if (skewed) std::printf("  %s\n", note.c_str());
            all = all && c.pass;
        }
    if (!all) {
        std::printf("GATE FAIL\n");
        return 1;
    }
    std::printf("GATE PASS (every case bitwise)\n");
    if (!timing) return 0;
    for (const unsigned T : {1024U, 2048U, 4096U}) {
        std::string note;
        const Routes r = make_routes(T, false, g, note);
        const std::string label = "timing uniform T" + std::to_string(T);
        const CaseResult c = run_case(m, r, label.c_str(), g, 5);
        const double mflop = static_cast<double>(T) * kTop * 2.0 * (2.0 * kH * kI + static_cast<double>(kI) * kH) / 1e6;  // 25.17 / pair
        std::printf("TIME T=%u: grouped GPU %.3f ms (%.1f TOPS), whole call %.3f ms (%.1f TOPS); library chain %.3f ms (%.1f TOPS); "
                    "chain/grouped %.2fx\n",
                    T, c.gpu_ms, mflop / c.gpu_ms / 1e3, c.wall_ms, mflop / c.wall_ms / 1e3, c.ref_ms, mflop / c.ref_ms / 1e3,
                    c.ref_ms / c.gpu_ms);
        all = all && c.pass;
    }
    std::printf("%s\n", all ? "ALL PASS" : "FAIL");
    return all ? 0 : 1;
}
