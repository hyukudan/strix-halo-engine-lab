// Device identity of the shared round folds (no model): (1) the causal convolution with RoundInput on unrounded
// input and dynamic taps equals Bf16Round in place followed by the incumbent convolution (output and history bitwise,
// decode and prefill, static and dynamic taps, both output stores); (2) a BF16 tight-row projection with round_bf16
// equals the plain projection followed by Bf16Round (bitwise on finite values).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/fma_bf16_rows.hpp"
#include "core/row_transforms.hpp"
#include "core/short_convolution.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace core = strix::core;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FAIL round-fold-hip %s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* upload(const std::vector<T>& host) {
    T* d = nullptr;
    ok(hipMalloc(&d, host.size() * sizeof(T) + 16U), "malloc");
    ok(hipMemcpy(d, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> host(n);
    ok(hipMemcpy(host.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return host;
}

std::vector<float> noise(std::size_t n, std::mt19937& rng, float scale) {
    std::normal_distribution<float> dist(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// One convolution case: arm A rounds input and dynamic in place then convolves (RoundInput off), arm B convolves the
// unrounded copies with RoundInput on. Both start from the same (already rounded) history.
template <bool RoundOutput>
bool conv_case(unsigned channels, unsigned taps, unsigned groups, unsigned rows, std::mt19937& rng) {
    const core::ShortConvolution plan{channels, taps, groups, 1.0F, groups ? 3.0F : 2.0F};
    const auto input = noise(std::size_t{rows} * channels, rng, 1.7F);
    const auto weights = noise(std::size_t{taps} * channels, rng, 0.6F);
    const auto dynamic = noise(std::size_t{rows} * (groups ? groups : 1U) * taps, rng, 0.3F);
    auto history = noise(std::size_t{taps - 1U} * channels, rng, 1.1F);
    for (auto& h : history) h = strix::weight::scalar_float::bf16_round(h);
    const std::vector<float> zeros(std::size_t{rows} * channels, 0.0F);

    float* in_a = upload(input);
    float* in_b = upload(input);
    float* dyn_a = groups ? upload(dynamic) : nullptr;
    float* dyn_b = groups ? upload(dynamic) : nullptr;
    float* w = upload(weights);
    float* hist_a = upload(history);
    float* hist_b = upload(history);
    float* out_a = upload(zeros);
    float* out_b = upload(zeros);
    namespace rt = core::row_transforms;
    ok(rt::elements(input.size(), rt::Bf16Round{in_a}, nullptr), "round input");
    if (groups) ok(rt::elements(dynamic.size(), rt::Bf16Round{dyn_a}, nullptr), "round taps");
    ok(core::causal_conv::launch_with_store<RoundOutput>(plan, {rows, in_a, w, dyn_a, hist_a, out_a}, nullptr), "A");
    ok(core::causal_conv::launch_with_store<RoundOutput, true>(plan, {rows, in_b, w, dyn_b, hist_b, out_b}, nullptr),
       "B");
    ok(hipDeviceSynchronize(), "sync");
    const auto oa = download(out_a, zeros.size()), ob = download(out_b, zeros.size());
    const auto ha = download(hist_a, history.size()), hb = download(hist_b, history.size());
    const bool same = std::memcmp(oa.data(), ob.data(), oa.size() * 4U) == 0 &&
                      std::memcmp(ha.data(), hb.data(), ha.size() * 4U) == 0;
    std::printf("%s round-fold-hip conv channels=%u taps=%u groups=%u rows=%u round_output=%d\n",
                same ? "PASS" : "FAIL", channels, taps, groups, rows, RoundOutput ? 1 : 0);
    for (void* p : {static_cast<void*>(in_a), static_cast<void*>(in_b), static_cast<void*>(dyn_a),
                    static_cast<void*>(dyn_b), static_cast<void*>(w), static_cast<void*>(hist_a),
                    static_cast<void*>(hist_b), static_cast<void*>(out_a), static_cast<void*>(out_b)})
        if (p) (void)hipFree(p);
    return same;
}

bool projection_case(unsigned width, unsigned outputs, unsigned rows, std::mt19937& rng) {
    const auto x = noise(std::size_t{rows} * width, rng, 1.0F);
    std::vector<std::uint16_t> wbits(std::size_t{outputs} * width);
    for (std::size_t i = 0; i < wbits.size(); ++i)
        wbits[i] = strix::weight::scalar_float::bf16_encode(noise(1, rng, 0.05F)[0]);
    const std::vector<float> zeros(std::size_t{rows} * outputs, 0.0F);
    float* dx = upload(x);
    std::uint16_t* dw = upload(wbits);
    float* ya = upload(zeros);
    float* yb = upload(zeros);
    const core::Bf16RowProjection plain{dw, outputs, ya};
    core::Bf16RowProjection stored{dw, outputs, yb};
    stored.round_bf16 = true;
    ok(core::fma_rows::enqueue_bf16_rows(dx, width, rows, {&plain, 1U}, outputs, nullptr), "plain projection");
    namespace rt = core::row_transforms;
    ok(rt::elements(zeros.size(), rt::Bf16Round{ya}, nullptr), "round projection");
    ok(core::fma_rows::enqueue_bf16_rows(dx, width, rows, {&stored, 1U}, outputs, nullptr), "stored projection");
    ok(hipDeviceSynchronize(), "sync");
    const auto a = download(ya, zeros.size()), b = download(yb, zeros.size());
    const bool same = std::memcmp(a.data(), b.data(), a.size() * 4U) == 0;
    std::printf("%s round-fold-hip projection width=%u outputs=%u rows=%u admitted=%d\n", same ? "PASS" : "FAIL", width,
                outputs, rows, core::fma_rows::rounds_in_store(rows) ? 1 : 0);
    (void)hipFree(dx);
    (void)hipFree(dw);
    (void)hipFree(ya);
    (void)hipFree(yb);
    return same;
}
}  // namespace

int main() {
    std::mt19937 rng(20261122U);
    bool all = true;
    for (const unsigned rows : {1U, 5U}) {
        all = conv_case<false>(512, 4, 0, rows, rng) && all;
        all = conv_case<true>(512, 4, 0, rows, rng) && all;
        all = conv_case<false>(512, 4, 8, rows, rng) && all;
        all = conv_case<true>(512, 4, 8, rows, rng) && all;
    }
    for (const unsigned rows : {1U, 4U, 16U}) all = projection_case(2304, 2304, rows, rng) && all;
    std::puts(all ? "round_fold_hip: PASS" : "round_fold_hip: FAIL");
    return all ? 0 : 1;
}
