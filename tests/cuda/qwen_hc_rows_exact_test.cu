// Byte-identity + timing of the decode-row HC / router kernels: incumbent vs latency-hiding forms (rows 1..8).
// Includes qwen_hc.cu for the private HC and router-logit kernels. --route-exact compares the public selector only.
// Build:  tests\cuda\build_qwen_test.bat <worktree> qwen_hc_rows_exact_test <outdir>   (adds gguf_reader + amd_math_tables)
// Run:    qwen_hc_rows_exact_test.exe <tables dir, e.g. D:/tmp/qrtxpf-build/tables>   (GPU: via pro.sh)
#include "../../src/cuda/models/qwen/qwen_hc.cu"

#include "strix/cuda/amd_math_tables.hpp"
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

namespace strix::cuda::qwen_hc {
namespace {
#include "../cuda_core/f3g_route_frozen.inc"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)
int g_pass = 0, g_fail = 0;
void* g_flush = nullptr;
constexpr std::size_t kFlush = 256u << 20;

template <class F> float time_us(F&& launch, bool cold, int reps = 50) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float total = 0.0F;
    for (int i = 0; i < reps + 3; ++i) {
        if (cold) CK(cudaMemsetAsync(g_flush, i, kFlush, 0));
        CK(cudaEventRecord(a, 0)); launch(); CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); if (i >= 3) total += ms;
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return total * 1000.0F / reps;
}
// kernel-only time: 20 back-to-back launches between two events (launch latency hidden, warm L2)
template <class F> float time_b2b(F&& launch, int reps = 10) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) launch();
    CK(cudaDeviceSynchronize()); float total = 0.0F;
    for (int k = 0; k < reps; ++k) {
        CK(cudaEventRecord(a, 0)); for (int i = 0; i < 20; ++i) launch(); CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); total += ms;
    }
    return total * 1000.0F / (20.0F * reps);
}
template <class T> T* dev(const std::vector<T>& h) {
    T* d; CK(cudaMalloc(&d, h.size() * sizeof(T))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
template <class T> std::vector<T> host(const T* d, std::size_t n) {
    std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T)); }
void verdict(const char* what, unsigned R, bool ok) { (ok ? g_pass : g_fail)++; if (!ok) std::printf("FAIL %s rows=%u\n", what, R); }

std::mt19937 rng(12345);
float nrm(float sd) { return std::normal_distribution<float>(0.0F, sd)(rng); }
std::uint16_t bf16(float v) { std::uint32_t b; std::memcpy(&b, &v, 4); return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16); }

std::vector<std::uint32_t> test_route_select(const std::vector<float>& logits, unsigned rows, unsigned top_k,
                                             bool normalize, const am::Tables& tables, bool timing = false) {
    const std::size_t slots = std::size_t{rows} * top_k;
    constexpr std::uint32_t id_guard = 0xfeedU;
    constexpr float weight_guard = 123.0F;
    const std::vector<std::uint32_t> init_ids(slots + 1U, id_guard);
    const std::vector<float> init_weights(slots + 1U, weight_guard);
    auto* ld = dev(logits);
    auto* frozen_ids = dev(init_ids);
    auto* frozen_weights = dev(init_weights);
    auto* bound_ids = dev(init_ids);
    auto* bound_weights = dev(init_weights);
    const auto block = [&] {
        frozen::route_select_kernel<<<rows, 256U>>>(ld, top_k, normalize, frozen_ids, frozen_weights, tables);
        CK(cudaGetLastError());
    };
    const auto warp = [&] {
        frozen::route_select_warp_kernel<<<rows, 256U>>>(ld, top_k, normalize, frozen_ids, frozen_weights, tables);
        CK(cudaGetLastError());
    };
    const auto bound = [&] { route_select(ld, rows, top_k, normalize, tables, bound_ids, bound_weights, nullptr); };
    bound();
    const auto ids = host(bound_ids, slots + 1U);
    const auto weights = host(bound_weights, slots + 1U);
    const auto matches = [&] {
        return same(ids, host(frozen_ids, slots + 1U)) && same(weights, host(frozen_weights, slots + 1U));
    };
    block();
    const bool block_exact = matches();
    CK(cudaMemcpy(frozen_ids, init_ids.data(), init_ids.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(frozen_weights, init_weights.data(), init_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    warp();
    const bool warp_exact = matches();
    if (!block_exact || !warp_exact) {
        std::printf("route top_k=%u normalize=%d block_exact=%d warp_exact=%d\n",
                    top_k, normalize, block_exact, warp_exact);
    }
    verdict("select", rows, block_exact && warp_exact && ids.back() == id_guard && weights.back() == weight_guard);
    if (timing) std::printf("select   R=%u block %.2f us  warp %.2f us  bound %.2f us (b2b)\n",
                            rows, time_b2b(block), time_b2b(warp), time_b2b(bound));
    CK(cudaFree(bound_weights)); CK(cudaFree(bound_ids)); CK(cudaFree(frozen_weights)); CK(cudaFree(frozen_ids));
    CK(cudaFree(ld));
    return ids;
}

void test_route_exact(const am::Tables& tables) {
    constexpr unsigned rows = 32U;
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 random(0xF351U);
    std::vector<float> logits(rows * kExperts);
    for (float& v : logits) v = static_cast<float>(static_cast<int>(random() % 4096U) - 2048) / 64.0F;
    std::fill_n(logits.data(), kExperts, neg_inf);
    std::fill_n(logits.data() + kExperts, kExperts, nan);
    logits[2U * kExperts + 319U] = nan;
    logits[3U * kExperts + 511U] = -neg_inf;
    std::fill_n(logits.data() + 4U * kExperts, kExperts, neg_inf);
    constexpr unsigned tied[]{31U, 32U, 255U, 256U, 511U};
    for (unsigned expert : tied) logits[4U * kExperts + expert] = 4.0F;
    std::fill_n(logits.data() + 5U * kExperts, kExperts, -0.0F);
    logits[5U * kExperts + 256U] = 0.0F;

    unsigned cases = 0;
    for (unsigned top_k = 1; top_k <= moe::kGfx1151RouteMaxTopK; ++top_k) {
        for (bool normalize : {false, true}) {
            const auto ids = test_route_select(logits, rows, top_k, normalize, tables);
            bool valid = true;
            for (unsigned row = 0; row < rows; ++row) {
                for (unsigned slot = 0; slot < top_k; ++slot) {
                    const auto id = ids[row * top_k + slot];
                    valid &= row < 4U ? id == kExperts : id < kExperts;
                }
            }
            verdict("invalid_route sentinel", rows, valid);
            for (unsigned slot = 0; slot < top_k; ++slot) {
                const unsigned sparse_id = slot < std::size(tied) ? tied[slot] : slot - static_cast<unsigned>(std::size(tied));
                verdict("ties", rows, ids[4U * top_k + slot] == sparse_id && ids[5U * top_k + slot] == slot);
            }
            cases += rows;
        }
    }
    if (!g_fail) std::printf("Qwen HC route: %u row/case comparisons to both frozen selectors EXACT, invalid_route PASS\n", cases);
}

template <unsigned R> void test_router(const float* wd, const std::uint16_t* hd, float* l_old, float* l_new, const am::Tables& tab, unsigned top_k, bool timing) {
    constexpr unsigned smem_old = kRtePerBlock * kRtePitch * 4U + R * (kHidden + 2U) * 2U;
    constexpr unsigned smem_new = kRtePerBlock * kRtvWP * 4U + R * kRtvXP * 2U;
    CK(cudaFuncSetAttribute(route_logits_rows_kernel<R, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_old));
    CK(cudaFuncSetAttribute(route_logits_pipe_kernel<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_new));
    auto run_old = [&] { route_logits_rows_kernel<R, true><<<kExperts / kRtePerBlock, 256U, smem_old>>>(hd, wd, l_old); };
    auto run_new = [&] { route_logits_pipe_kernel<R><<<kExperts / kRtePerBlock, 128U, smem_new>>>(hd, wd, l_new); };
    run_old(); run_new(); CK(cudaDeviceSynchronize());
    verdict("router logits", R, same(host(l_old, R * kExperts), host(l_new, R * kExperts)));
    if (timing) std::printf("router   R=%u old %.2f us  new %.2f us (b2b) | event cold old %.2f new %.2f\n", R, time_b2b(run_old), time_b2b(run_new), time_us(run_old, true), time_us(run_new, true));
    // select on the logits (three variants: plain, quantised ties, one NaN row)
    for (int variant = 0; variant < 3; ++variant) {
        auto lg = host(l_old, R * kExperts);
        if (variant == 1) for (auto& v : lg) v = std::round(v * 4.0F) / 4.0F;
        if (variant == 2) lg[(R - 1) * kExperts + 7] = std::nanf("");
        for (bool normalize : {false, true})
            test_route_select(lg, R, top_k, normalize, tab, timing && variant == 0 && normalize);
    }
}

template <unsigned R, unsigned RG, unsigned FB> void test_hc_out(const float* nd, const std::uint32_t* up, const float* us, std::uint16_t* o_old, std::uint16_t* o_new, const am::Tables& tab, bool timing, bool sweep) {
    auto run_old = [&] { hc_output_rows_kernel<R, RG><<<dim3(kHidden / 32U, (R + RG - 1U) / RG), 128U>>>(nd, up, us, o_old, tab); };
    auto run_new = [&] { hc_output_rows_reg_kernel<R, RG, FB><<<dim3(kHidden / FB, (R + RG - 1U) / RG), FB * 4U>>>(nd, up, us, o_new, tab); };
    CK(cudaMemset(o_old, 0, R * kHidden * 2)); CK(cudaMemset(o_new, 0, R * kHidden * 2));
    run_old(); run_new(); CK(cudaDeviceSynchronize());
    verdict("hc_output", R, same(host(o_old, R * kHidden), host(o_new, R * kHidden)));
    if (timing) std::printf("hc_out   R=%u RG=%u FB=%u old %.2f us  new %.2f us (b2b) | event cold old %.2f new %.2f\n", R, RG, FB, time_b2b(run_old), time_b2b(run_new), time_us(run_old, true), time_us(run_new, true));
}

template <unsigned R> void test_all(const std::vector<float>& wv, const std::vector<std::uint16_t>& hv, const float* wd, const std::uint16_t* hd, const float* nd, const std::uint32_t* up, const float* us, const am::Tables& tab, bool timing) {
    (void)wv; (void)hv;
    float *lo, *ln; std::uint16_t *oo, *on;
    CK(cudaMalloc(&lo, R * kExperts * 4)); CK(cudaMalloc(&ln, R * kExperts * 4)); CK(cudaMalloc(&oo, R * kHidden * 2)); CK(cudaMalloc(&on, R * kHidden * 2));
    test_router<R>(wd, hd, lo, ln, tab, 10U, timing);
    test_router<R>(wd, hd, lo, ln, tab, 3U, false);
    constexpr unsigned RO = R >= 4U ? 2U : 1U;
    test_hc_out<R, RO, 16U>(nd, up, us, oo, on, tab, timing, false);
    test_hc_out<R, RO, 32U>(nd, up, us, oo, on, tab, timing, true);
    test_hc_out<R, R, 16U>(nd, up, us, oo, on, tab, false, false);
    cudaFree(lo); cudaFree(ln); cudaFree(oo); cudaFree(on);
}
}  // namespace
}  // namespace strix::cuda::qwen_hc

int main(int argc, char** argv) {
    using namespace strix::cuda::qwen_hc;
    if (argc == 3 && std::string_view(argv[1]) == "--route-exact") {
        auto tables = am::load_tables(argv[2]);
        test_route_exact(tables);
        am::free_tables(tables);
        return g_fail ? 1 : 0;
    }
    const std::string dir = argc > 1 ? argv[1] : ".";
    am::Tables tab = am::load_tables(dir);
    CK(cudaMalloc(&g_flush, kFlush));
    // router inputs: 8 BF16 rows (hidden activations, mixed magnitudes) and 512 x 2560 FP32 weights
    std::vector<std::uint16_t> hv(8 * kHidden); for (auto& v : hv) v = bf16(nrm(1.5F) * (rng() % 50 == 0 ? 30.0F : 1.0F));
    std::vector<float> wv(std::size_t{kExperts} * kHidden); for (auto& v : wv) v = nrm(0.03F);
    // HC: normalized rows [8][kNormalizedPitch] (norm part ~1, low part ~ +-3), packed up codes/scales
    std::vector<float> nv(8 * std::size_t{kNormalizedPitch}); for (auto& v : nv) v = nrm(1.0F) * 1.5F;
    std::vector<std::uint32_t> up(std::size_t{kRank / 4U} * kStreams * kHidden); for (auto& v : up) v = static_cast<std::uint32_t>(rng());
    std::vector<float> us(std::size_t{kRank / 32U} * kStreams * kHidden); for (auto& v : us) v = 0.0004F * (1.0F + (rng() % 100) / 50.0F);
    auto* hd = dev(hv); auto* wd = dev(wv); auto* nd = dev(nv); auto* upd = dev(up); auto* usd = dev(us);
    for (int pass = 0; pass < 2; ++pass) {  // pass 0: verify + timing at rows 1,4,8; pass 1: verify all rows
        test_all<1>(wv, hv, wd, hd, nd, upd, usd, tab, true);
        test_all<2>(wv, hv, wd, hd, nd, upd, usd, tab, pass == 1 ? false : false);
        test_all<3>(wv, hv, wd, hd, nd, upd, usd, tab, false);
        test_all<4>(wv, hv, wd, hd, nd, upd, usd, tab, true);
        test_all<5>(wv, hv, wd, hd, nd, upd, usd, tab, false);
        test_all<6>(wv, hv, wd, hd, nd, upd, usd, tab, false);
        test_all<7>(wv, hv, wd, hd, nd, upd, usd, tab, false);
        test_all<8>(wv, hv, wd, hd, nd, upd, usd, tab, true);
        if (pass == 0) { for (auto& v : hv) v = bf16(nrm(0.2F)); for (auto& v : wv) v = nrm(0.5F); CK(cudaMemcpy(hd, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(wd, wv.data(), wv.size() * 4, cudaMemcpyHostToDevice)); }
    }
    std::printf("PASS %d FAIL %d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
