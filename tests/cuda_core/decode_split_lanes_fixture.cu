// No-model identity fixture of attention_decode_prepared_split_lanes_bf16: every lane of a lanes launch must give, bit
// for bit, the output of the one-row launch attention_decode_prepared_split_bf16 of that row (its Q row, K / V planes,
// device position, bias row and sink row). Cases: distinct caches across chunk boundaries, verify rows sharing one
// cache, wrapped window caches, and more than one 16-lane block. Prints "decode-split-lanes EXACT PASS cases=N".
#include "strix/cuda_core/attention.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace cc = strix::cuda_core;
namespace ln = strix::attn::lanes;

namespace {

constexpr unsigned kHeads = 24, kKv = 8, kDim = 128, kExtent = 1024;
constexpr float kScale = 0.08838834764831845f;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    check(cudaMalloc(&p, h.size() * sizeof(T) + 16), "malloc");
    check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return p;
}

struct Case {
    const char* name;
    unsigned slots;               // cache slots of every cache (= d.max_positions of the one-row launch)
    unsigned window;
    bool wrapped;
    bool shared_cache;            // verify rows: every lane reads cache 0
    std::vector<unsigned> positions;
};

bool run(const Case& c, std::mt19937& rng) {
    const unsigned lanes = static_cast<unsigned>(c.positions.size());
    const unsigned caches = c.shared_cache ? 1U : lanes;
    const std::size_t cache_elems = static_cast<std::size_t>(c.slots) * kKv * kDim;
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::uniform_int_distribution<unsigned> bits(0, 0xffffU);
    std::vector<std::uint16_t*> k(caches), v(caches);
    for (unsigned i = 0; i < caches; ++i) {
        std::vector<std::uint16_t> hk(cache_elems), hv(cache_elems);
        // finite BF16: random mantissa, exponent around 1
        for (auto& x : hk) x = static_cast<std::uint16_t>(0x3e00U | (bits(rng) & 0x81ffU));
        for (auto& x : hv) x = static_cast<std::uint16_t>(0x3e00U | (bits(rng) & 0x81ffU));
        k[i] = dev(hk);
        v[i] = dev(hv);
    }
    std::vector<float> hq(static_cast<std::size_t>(lanes) * kHeads * kDim), hb(static_cast<std::size_t>(lanes) * kHeads * kExtent),
        hs(static_cast<std::size_t>(lanes) * kHeads);
    for (auto& x : hq) x = uni(rng);
    for (auto& x : hb) x = uni(rng);
    for (auto& x : hs) x = uni(rng) * 2.0f;
    float* dq = dev(hq);
    float* db = dev(hb);
    float* ds = dev(hs);
    std::vector<float> zeros(hq.size(), 0.0f);
    float* out_one = dev(zeros);
    float* out_lanes = dev(zeros);
    std::uint32_t* dpos = dev(std::vector<std::uint32_t>(c.positions.begin(), c.positions.end()));

    const cc::AttentionDims dims{kHeads, kKv, kDim, kScale, c.slots};
    const std::size_t row_ws = cc::decode_split_workspace_bytes(dims, c.window);
    const std::size_t lane_ws = cc::decode_split_lanes_workspace_bytes(dims, c.window);
    void* ws1 = nullptr;
    void* wsl = nullptr;
    check(cudaMalloc(&ws1, row_ws), "ws1");
    check(cudaMalloc(&wsl, lane_ws), "wsl");
    check(cc::decode_split_workspace_init(ws1, dims, c.window, nullptr), "init1");
    check(cc::decode_split_lanes_workspace_init(wsl, dims, c.window, nullptr), "initl");

    std::vector<ln::PreparedBf16Lane> table(lanes);
    for (unsigned i = 0; i < lanes; ++i) {
        const unsigned cache = c.shared_cache ? 0U : i;
        table[i] = {k[cache], v[cache], c.positions[i], c.slots, c.wrapped ? 1U : 0U};
        const cc::PreparedBf16Flash row_prepared{db + static_cast<std::size_t>(i) * kHeads * kExtent,
                                                 ds + static_cast<std::size_t>(i) * kHeads, kExtent, c.window,
                                                 c.wrapped, false};
        check(cc::attention_decode_prepared_split_bf16(
                  dims, row_prepared, dq + static_cast<std::size_t>(i) * kHeads * kDim,
                  reinterpret_cast<const __nv_bfloat16*>(k[cache]), reinterpret_cast<const __nv_bfloat16*>(v[cache]),
                  dpos + i, out_one + static_cast<std::size_t>(i) * kHeads * kDim, ws1, row_ws, nullptr),
              "one-row");
    }
    const cc::PreparedBf16Flash prepared{db, ds, kExtent, c.window, c.wrapped, false};
    check(cc::attention_decode_prepared_split_lanes_bf16(dims, prepared, dq, table, out_lanes, wsl, lane_ws, nullptr),
          "lanes");
    check(cudaDeviceSynchronize(), "sync");
    std::vector<float> a(hq.size()), b(hq.size());
    check(cudaMemcpy(a.data(), out_one, a.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h");
    check(cudaMemcpy(b.data(), out_lanes, b.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h");
    std::size_t diff = 0, nonfinite = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++diff;
        if (!(a[i] == a[i]) || a[i] > 1e30f || a[i] < -1e30f) ++nonfinite;
    }
    std::printf("case %-14s lanes=%2u window=%4u wrapped=%d shared=%d diff=%zu nonfinite=%zu\n", c.name, lanes, c.window,
                c.wrapped ? 1 : 0, c.shared_cache ? 1 : 0, diff, nonfinite);
    for (unsigned i = 0; i < caches; ++i) { cudaFree(k[i]); cudaFree(v[i]); }
    cudaFree(dq); cudaFree(db); cudaFree(ds); cudaFree(out_one); cudaFree(out_lanes); cudaFree(dpos);
    cudaFree(ws1); cudaFree(wsl);
    return diff == 0 && nonfinite == 0;
}

}  // namespace

int main() {
    std::mt19937 rng(1234);
    const std::vector<Case> cases = {
        {"boundaries", 4096, 0, false, false, {0, 1, 255, 256, 257, 511, 512, 1000, 2520, 4095}},
        {"verify-rows", 4096, 0, false, true, {2517, 2518, 2519, 2520}},
        {"window-wrap", 1024, 1024, true, false, {300, 1023, 1024, 1500, 3000, 7777}},
        {"window-flat", 4096, 1024, false, false, {100, 1024, 2047, 4000}},
        {"two-blocks", 3072, 0, false, false,
         {5, 260, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3071, 17, 33, 290, 770, 1290, 2050, 2600, 3001}},
    };
    unsigned pass = 0;
    for (const auto& c : cases) pass += run(c, rng) ? 1U : 0U;
    const bool ok = pass == cases.size();
    std::printf("decode-split-lanes %s cases=%u/%zu\n", ok ? "EXACT PASS" : "MISMATCH", pass, cases.size());
    return ok ? 0 : 1;
}
