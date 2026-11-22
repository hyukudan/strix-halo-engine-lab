// Decode lanes of the shared prepared BF16 flash (attn_core::dense::append_prepared_bf16_lanes) == one
// append_prepared_bf16<Policy, true> per lane, bitwise: attention output, both caches and the failure word, for cohorts
// of 1, 3, 16 and 20 lanes (two launches) at mixed positions (one split .. 16 splits, full cache and sliding ring).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "attn_core/prepared_bf16_attention.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace dense = strix::attn_core::dense;

namespace {

constexpr unsigned kDim = 128U, kKvHeads = 8U, kGroup = 3U, kHeads = kKvHeads * kGroup, kExtent = 64U;
constexpr unsigned kWidth = kKvHeads * kDim;

struct Policy : dense::PreparedBf16Policy<kDim, kGroup, true, true> {
    static constexpr float kScaleLog2 = 0.08838834764831845F * 1.4426950408889634F;
};

#define HIP_OK(expr)                                                                       \
    do {                                                                                   \
        const hipError_t status_ = (expr);                                                 \
        if (status_ != hipSuccess) {                                                       \
            std::printf("FAIL %s: %s\n", #expr, hipGetErrorString(status_));              \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    HIP_OK(hipMalloc(&device, host.size() * sizeof(T)));
    HIP_OK(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    return device;
}

template <class T>
std::vector<T> download(const T* device, std::size_t count) {
    std::vector<T> host(count);
    HIP_OK(hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost));
    return host;
}

std::uint16_t to_bf16(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

struct Lane {
    unsigned position;
    unsigned window;  // 0: full cache; otherwise a wrapped ring of kRing slots
};
constexpr unsigned kRing = 1040U;

// Returns the number of mismatching arrays.
int run_cohort(const std::vector<Lane>& spec, unsigned window, std::mt19937& rng) {
    const auto n = static_cast<unsigned>(spec.size());
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    const auto fill = [&](std::size_t count, float scale) {
        std::vector<float> v(count);
        for (auto& x : v) x = u(rng) * scale;
        return v;
    };
    const auto queries = fill(std::size_t{n} * kHeads * kDim, 1.0F);
    const auto keys = fill(std::size_t{n} * kWidth, 1.0F);
    const auto values = fill(std::size_t{n} * kWidth, 1.0F);
    const auto bias = fill(std::size_t{n} * kHeads * kExtent, 0.5F);
    const auto sinks = fill(std::size_t{n} * kHeads, 0.5F);
    auto* d_queries = upload(queries);
    auto* d_keys = upload(keys);
    auto* d_values = upload(values);
    auto* d_bias = upload(bias);
    auto* d_sinks = upload(sinks);
    std::vector<unsigned> slots(n);
    std::vector<std::vector<std::uint16_t>> caches(2U * n);
    for (unsigned l = 0; l < n; ++l) {
        slots[l] = window == 0U ? spec[l].position + 1U : kRing;
        for (unsigned c = 0; c < 2U; ++c) {
            auto& cache = caches[2U * l + c];
            cache.resize(std::size_t{slots[l]} * kWidth);
            for (auto& b : cache) b = to_bf16(u(rng));
        }
    }
    const std::vector<std::uint16_t> zero_row(kDim, 0U);
    auto* d_zero = upload(zero_row);
    const std::size_t out_floats = std::size_t{n} * kHeads * kDim;
    float* out_ref = nullptr;
    float* out_lanes = nullptr;
    HIP_OK(hipMalloc(&out_ref, out_floats * sizeof(float)));
    HIP_OK(hipMalloc(&out_lanes, out_floats * sizeof(float)));
    HIP_OK(hipMemset(out_ref, 0xff, out_floats * sizeof(float)));
    HIP_OK(hipMemset(out_lanes, 0xff, out_floats * sizeof(float)));
    std::vector<std::uint16_t*> ref_caches(2U * n), lane_caches(2U * n);
    for (std::size_t i = 0; i < caches.size(); ++i) {
        ref_caches[i] = upload(caches[i]);
        lane_caches[i] = upload(caches[i]);
    }
    const std::size_t ws_floats = dense::key_split_workspace_floats(16U, kHeads, kDim);
    float* workspace = nullptr;
    HIP_OK(hipMalloc(&workspace, ws_floats * sizeof(float)));
    std::vector<unsigned> zero_flags(2U, 0U);
    unsigned* flags = upload(zero_flags);

    // Reference: the incumbent one-lane append per lane, serially.
    for (unsigned l = 0; l < n; ++l) {
        const strix::attn_core::PreparedGqaAttentionRows rows{
            d_queries + std::size_t{l} * kHeads * kDim, d_keys + std::size_t{l} * kWidth,
            d_values + std::size_t{l} * kWidth, d_bias + std::size_t{l} * kHeads * kExtent,
            d_sinks + std::size_t{l} * kHeads, out_ref + std::size_t{l} * kHeads * kDim, spec[l].position, 1U};
        const dense::PreparedBf16Cache cache{ref_caches[2U * l], ref_caches[2U * l + 1U], d_zero, slots[l],
                                             window != 0U};
        const unsigned splits = dense::small_row_key_splits(spec[l].position, 1U, window);
        const auto ws = splits > 1U ? std::span<float>(workspace, ws_floats) : std::span<float>{};
        HIP_OK((dense::append_prepared_bf16<Policy, true>(kKvHeads, window, kExtent, rows, cache, flags, nullptr, ws,
                                                          splits)));
    }
    // Lanes: one store / flash / combine per 16 lanes.
    std::vector<dense::PreparedBf16Lane> lanes(n);
    for (unsigned l = 0; l < n; ++l)
        lanes[l] = {lane_caches[2U * l], lane_caches[2U * l + 1U], spec[l].position, slots[l], window != 0U ? 1U : 0U};
    const strix::attn_core::PreparedGqaAttentionRows image{d_queries, d_keys, d_values, d_bias, d_sinks, out_lanes, 0U, n};
    HIP_OK(dense::append_prepared_bf16_lanes<Policy>(kKvHeads, window, kExtent, image, lanes, d_zero, flags + 1,
                                                     nullptr, std::span<float>(workspace, ws_floats)));
    HIP_OK(hipDeviceSynchronize());

    int bad = 0;
    const auto a = download(out_ref, out_floats);
    const auto b = download(out_lanes, out_floats);
    if (std::memcmp(a.data(), b.data(), out_floats * sizeof(float)) != 0) {
        std::printf("  output differs (lanes %u, window %u)\n", n, window);
        ++bad;
    }
    for (std::size_t i = 0; i < caches.size(); ++i) {
        const auto x = download(ref_caches[i], caches[i].size());
        const auto y = download(lane_caches[i], caches[i].size());
        if (x != y) {
            std::printf("  cache %zu differs\n", i);
            ++bad;
        }
        HIP_OK(hipFree(ref_caches[i]));
        HIP_OK(hipFree(lane_caches[i]));
    }
    const auto f = download(flags, 2U);
    if (f[0] != f[1]) {
        std::printf("  failure word %u vs %u\n", f[0], f[1]);
        ++bad;
    }
    for (void* p : {static_cast<void*>(d_queries), static_cast<void*>(d_keys), static_cast<void*>(d_values),
                    static_cast<void*>(d_bias), static_cast<void*>(d_sinks), static_cast<void*>(d_zero),
                    static_cast<void*>(out_ref), static_cast<void*>(out_lanes), static_cast<void*>(workspace),
                    static_cast<void*>(flags)})
        HIP_OK(hipFree(p));
    std::printf("lanes %2u window %4u: %s\n", n, window, bad == 0 ? "BIT-IDENTICAL" : "DIFFER");
    return bad;
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP: no HIP device\n");
        return 77;
    }
    std::mt19937 rng(1122U);
    int bad = 0;
    // Full cache: 1 split (< 128 keys) up to 16 splits (>= 1921 keys).
    bad += run_cohort({{77U, 0U}}, 0U, rng);
    bad += run_cohort({{5U, 0U}, {300U, 0U}, {2500U, 0U}}, 0U, rng);
    std::vector<Lane> wide, wider;
    for (unsigned l = 0; l < 16U; ++l) wide.push_back({40U + 131U * l, 0U});
    for (unsigned l = 0; l < 20U; ++l) wider.push_back({200U + 97U * l, 0U});
    bad += run_cohort(wide, 0U, rng);
    bad += run_cohort(wider, 0U, rng);
    // Sliding ring of 1024 (wrapped slots), positions before and past the wrap.
    std::vector<Lane> ring;
    for (unsigned l = 0; l < 6U; ++l) ring.push_back({100U + 700U * l, 1024U});
    bad += run_cohort(ring, 1024U, rng);
    std::printf("prepared_bf16_lanes: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
