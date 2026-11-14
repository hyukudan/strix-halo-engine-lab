// Load-time converter byte oracle: every source width, shape and expert stride.
#include "moe_core/exl3_trellis.hpp"
#include "core/activation_quant.hpp"
#include "core/wmma_tiled.hpp"  // strix-trellis convert_exl3_iu4 (STRIX_TRELLIS_IU4_DIR/src)

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mc = strix::moe_core;

namespace {
template <int Bits>
__global__ __launch_bounds__(256) void reference_kernel(const std::uint32_t* const* trellis, unsigned K, unsigned N, std::uint8_t* wq,
                                                        _Float16* ws, std::size_t wq_expert, std::size_t ws_expert) {
    const unsigned lane = threadIdx.x & 31U, item = blockIdx.x * 8U + (threadIdx.x >> 5U), ntiles = N / 16U;
    const unsigned expert = blockIdx.y;
    const std::uint32_t* src = trellis[expert];
    if (src == nullptr || item >= (K / 64U) * ntiles) return;
    const unsigned g = item / ntiles, nt = item % ntiles;
    const auto* packed = reinterpret_cast<const unsigned char*>(src);
    const mc::LaneGeom<Bits> geo(lane);
    float v[4][8];
#pragma unroll
    for (unsigned u = 0; u < 4U; ++u) {
        const auto* tile = packed + (std::size_t{g * 4U + u} * ntiles + nt) * Bits * 32U;
        const mc::TileStates<Bits> st(tile, geo);
        const auto put = [&](unsigned j, auto h) {
            v[u][2U * j] = static_cast<float>(h[0]);
            v[u][2U * j + 1U] = static_cast<float>(h[1]);
        };
        put(0U, mc::mul1_pair(st.template state<0>(), st.template state<1>()));
        put(1U, mc::mul1_pair(st.template state<2>(), st.template state<3>()));
        put(2U, mc::mul1_pair(st.template state<4>(), st.template state<5>()));
        put(3U, mc::mul1_pair(st.template state<6>(), st.template state<7>()));
    }
    const unsigned q = lane & 3U, groups = K / 64U;
    std::uint8_t* base = wq + std::size_t{expert} * wq_expert;
    _Float16* sbase = ws + std::size_t{expert} * ws_expert;
#pragma unroll
    for (unsigned half = 0; half < 2U; ++half) {
        const unsigned n = nt * 16U + lane / 4U + 8U * half;
        float amax = 0.0F;
#pragma unroll
        for (unsigned u = 0; u < 4U; ++u)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) amax = fmaxf(amax, fabsf(v[u][4U * half + j]));
        amax = fmaxf(amax, __shfl_xor(amax, 1));
        amax = fmaxf(amax, __shfl_xor(amax, 2));
        const _Float16 s = static_cast<_Float16>(amax == 0.0F ? 1.0F : fmaxf(amax / 7.0F, 0x1p-24F));
        const float inv = 1.0F / static_cast<float>(s);
        if (q == 0U) sbase[std::size_t{n} * groups + g] = s;
#pragma unroll
        for (unsigned u = 0; u < 4U; ++u)
#pragma unroll
            for (unsigned p = 0; p < 2U; ++p) {
                const int lo = max(-7, min(7, static_cast<int>(__builtin_rintf(v[u][4U * half + 2U * p] * inv))));
                const int hi = max(-7, min(7, static_cast<int>(__builtin_rintf(v[u][4U * half + 2U * p + 1U] * inv))));
                const unsigned k = (g * 4U + u) * 16U + 8U * p + 2U * q;
                const std::size_t byte = ((std::size_t{k / 16U} * ntiles + n / 16U) * 16U + n % 16U) * 8U + (k % 16U) / 2U;
                base[byte] = static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 15U) | ((static_cast<unsigned>(hi) & 15U) << 4U));
            }
    }
}

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

template <int Bits>
void launch(bool ref, const std::uint32_t* const* table, unsigned K, unsigned N, unsigned experts, std::uint8_t* wq, _Float16* ws,
            std::size_t wqe, std::size_t wse, hipStream_t st) {
    if (ref)
        hipLaunchKernelGGL(reference_kernel<Bits>, dim3(((K / 64U) * (N / 16U) + 7U) / 8U, experts), dim3(256), 0, st, table, K, N, wq, ws, wqe,
                           wse);
    else
        ok(strix::core::convert_exl3_iu4(table, Bits, static_cast<int>(experts), static_cast<int>(K), static_cast<int>(N), wq, ws, wqe, wse, st),
           "convert_exl3_iu4");
    ok(hipGetLastError(), "launch");
}

template <int Bits>
int check(unsigned K, unsigned N, bool interleaved, std::mt19937& rng) {
    constexpr unsigned E = 3;
    const std::size_t tbytes = std::size_t{K} * N * Bits / 8U;
    const std::size_t wqe = interleaved ? strix::core::iu4_weight_bytes(K, N) + strix::core::iu4_scale_bytes(K, N) : strix::core::iu4_weight_bytes(K, N);
    const std::size_t wse = interleaved ? wqe / 2U : std::size_t{N} * (K / 64U);
    const std::size_t wq_bytes = E * wqe, ws_bytes = (interleaved ? E * wqe : E * strix::core::iu4_scale_bytes(K, N)) + 16;
    std::vector<std::uint32_t> h(E * tbytes / 4U);
    for (auto& x : h) x = rng();
    std::uint32_t* tr = nullptr;
    ok(hipMalloc(&tr, E * tbytes), "malloc");
    ok(hipMemcpy(tr, h.data(), E * tbytes, hipMemcpyHostToDevice), "copy");
    std::vector<const std::uint32_t*> tab{tr, nullptr, tr + 2 * tbytes / 4U};  // expert 1 skipped
    const std::uint32_t** td = nullptr;
    ok(hipMalloc(&td, E * sizeof(void*)), "malloc");
    ok(hipMemcpy(td, tab.data(), E * sizeof(void*), hipMemcpyHostToDevice), "copy");
    std::vector<std::uint8_t> out[2][2];
    for (int ref = 0; ref < 2; ++ref) {
        std::uint8_t* wq = nullptr;
        std::uint8_t* wsb = nullptr;
        ok(hipMalloc(&wq, wq_bytes), "malloc");
        ok(hipMemset(wq, 0xA5, wq_bytes), "memset");
        if (interleaved) wsb = wq + strix::core::iu4_weight_bytes(K, N);
        else {
            ok(hipMalloc(&wsb, ws_bytes), "malloc");
            ok(hipMemset(wsb, 0xA5, ws_bytes), "memset");
        }
        launch<Bits>(ref != 0, td, K, N, E, wq, reinterpret_cast<_Float16*>(wsb), wqe, wse, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        out[ref][0].resize(wq_bytes);
        ok(hipMemcpy(out[ref][0].data(), wq, wq_bytes, hipMemcpyDeviceToHost), "copy");
        if (!interleaved) {
            out[ref][1].resize(ws_bytes);
            ok(hipMemcpy(out[ref][1].data(), wsb, ws_bytes, hipMemcpyDeviceToHost), "copy");
            ok(hipFree(wsb), "free");
        }
        ok(hipFree(wq), "free");
    }
    ok(hipFree(tr), "free");
    ok(hipFree(td), "free");
    std::size_t diff = 0;
    for (int b = 0; b < 2; ++b)
        for (std::size_t i = 0; i < out[0][b].size(); ++i) diff += out[0][b][i] != out[1][b][i] ? 1U : 0U;
    std::printf("iu4 convert bits %d K %u N %u %s: %zu differing bytes of %zu -> %s\n", Bits, K, N, interleaved ? "interleaved" : "plain", diff,
                out[0][0].size() + out[0][1].size(), diff == 0 ? "PASS" : "FAIL");
    return diff == 0 ? 0 : 1;
}

// One MiniMax rank layer: 129 experts x (gate, up: 6144 -> 1536; down: 1536 -> 6144), all `Bits`.
}  // namespace

int main() {
    std::mt19937 rng(20261110U);
    int fail = 0;
    for (bool il : {false, true}) {
        fail += check<3>(6144, 1536, il, rng);
        fail += check<3>(1536, 6144, il, rng);
        fail += check<4>(6144, 1536, il, rng);
        fail += check<5>(6144, 1536, il, rng);
        fail += check<5>(1536, 6144, il, rng);
        fail += check<6>(1536, 6144, il, rng);
    }

    std::printf("iu4 convert tests: %s\n", fail == 0 ? "PASS" : "FAIL");
    return fail == 0 ? 0 : 1;
}
