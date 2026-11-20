// Device checks of two decode-stage forms that keep the incumbent bits, synthetic data, no model:
//  1. launch_hip_dense_rows rows 1: wave form (strix/dense_rows_wave.hpp, default) == block form (block_rows = true,
//     lm_head_kernel), BITWISE: BF16 120832 x 3072 fp32 input (Hy4 head), F32 512 x 3072 (Hy4 router), F16 777 x 1032,
//     BF16 100 x 4104 bf16 input, F32 33 x 8; inadmissible width 3004 falls back to the block form (equal by identity).
//  2. rmsnorm_row (src/ops/norm/norm_device.hpp) batched form == the loop form it replaced (kept here as the reference),
//     BITWISE: n 128 / 512 / 1024 / 3072 / 5120 / 6144 (batched) and 7000 (loop form on both sides), plain and
//     bf16_double finish, with and without the residual add, 1 and 37 rows.
//  3. timing (printed, not asserted): head and router geometry, block vs wave, median of 15; one-block norm n 3072 /
//     1024 / 512, 2000 back-to-back launches, loop vs batched (per launch, boundary included).
// Exit 77 when no HIP device is present.
#include "strix/hip_lm_head_launcher.hpp"
#include "ops/norm/norm_device.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
void hip(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint32_t lcg(std::uint32_t& s) {
    s = s * 1664525U + 1013904223U;
    return s;
}
float random_float(std::uint32_t& s) {
    const auto u = lcg(s);
    const float m = static_cast<float>(u >> 8U) / 16777216.0F - 0.5F;
    return (u & 7U) == 0U ? m * 16.0F : m;
}
std::uint16_t bf16_bits(float f) {
    std::uint32_t b = 0;
    std::memcpy(&b, &f, 4);
    return static_cast<std::uint16_t>(b >> 16U);
}
template <class T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    hip(hipMalloc(&d, h.size() * sizeof(T)), "malloc");
    hip(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}
template <class T>
std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hip(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}

// ---- 1. dense rows ----------------------------------------------------------------------------------------------
struct Shape {
    const char* name;
    std::uint32_t type;  // GGUF: 0 F32, 1 F16, 30 BF16
    std::uint32_t width, outputs;
    bool bf16_input;
};
std::vector<std::uint8_t> weights_of(const Shape& sh, std::uint32_t& seed) {
    const std::size_t n = std::size_t{sh.width} * sh.outputs;
    std::vector<std::uint8_t> w(n * (sh.type == 0U ? 4U : 2U));
    for (std::size_t i = 0; i < n; ++i) {
        const float f = random_float(seed);
        if (sh.type == 0U) std::memcpy(w.data() + 4U * i, &f, 4);
        else {
            std::uint16_t b = bf16_bits(f);
            if (sh.type == 1U) {
                const auto h = static_cast<_Float16>(f);
                std::memcpy(&b, &h, 2);
            }
            std::memcpy(w.data() + 2U * i, &b, 2);
        }
    }
    return w;
}
float run_rows(const Shape& sh, const void* x, const void* w, float* out, bool block, int reps) {
    strix::HipDenseRowsView v;
    v.x = x;
    v.input = sh.bf16_input ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;
    v.weights = w;
    v.ggml_type = sh.type;
    v.width = sh.width;
    v.outputs = sh.outputs;
    v.out = out;
    v.block_rows = block;
    std::vector<float> ms;
    hipEvent_t a, b;
    hip(hipEventCreate(&a), "event");
    hip(hipEventCreate(&b), "event");
    for (int r = 0; r < reps; ++r) {
        hip(hipEventRecord(a, nullptr), "record");
        strix::launch_hip_dense_rows(v, nullptr);
        hip(hipEventRecord(b, nullptr), "record");
        hip(hipEventSynchronize(b), "sync");
        float t = 0.0F;
        hip(hipEventElapsedTime(&t, a, b), "elapsed");
        ms.push_back(t);
    }
    hip(hipEventDestroy(a), "event");
    hip(hipEventDestroy(b), "event");
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2U];
}
void dense_rows_case(const Shape& sh, std::uint32_t& seed, bool timing) {
    const auto w = weights_of(sh, seed);
    std::vector<float> xf(sh.width);
    for (auto& f : xf) f = random_float(seed);
    std::vector<std::uint16_t> xb(sh.width);
    for (std::size_t i = 0; i < xb.size(); ++i) xb[i] = bf16_bits(xf[i]);
    void* dx = sh.bf16_input ? static_cast<void*>(upload(xb)) : static_cast<void*>(upload(xf));
    auto* dw = upload(w);
    float *o1 = nullptr, *o2 = nullptr;
    hip(hipMalloc(&o1, sh.outputs * 4U), "malloc");
    hip(hipMalloc(&o2, sh.outputs * 4U), "malloc");
    hip(hipMemset(o1, 0xff, sh.outputs * 4U), "memset");
    hip(hipMemset(o2, 0x7f, sh.outputs * 4U), "memset");
    const int reps = timing ? 15 : 1;
    const float t_block = run_rows(sh, dx, dw, o1, true, reps);
    const float t_wave = run_rows(sh, dx, dw, o2, false, reps);
    const auto a = download(o1, sh.outputs), b = download(o2, sh.outputs);
    check(std::memcmp(a.data(), b.data(), a.size() * 4U) == 0, (std::string("dense rows wave == block bitwise: ") + sh.name).c_str());
    if (timing) {
        const double gb = static_cast<double>(w.size()) / 1e9;
        std::printf("TIME %s block %.3f ms (%.0f GB/s) wave %.3f ms (%.0f GB/s)\n", sh.name, t_block, gb / (t_block / 1e3),
                    t_wave, gb / (t_wave / 1e3));
    }
    hip(hipFree(dx), "free");
    hip(hipFree(dw), "free");
    hip(hipFree(o1), "free");
    hip(hipFree(o2), "free");
}

// ---- 2. norm rows -----------------------------------------------------------------------------------------------
template <strix::ops::norm::OutputForm Form>
struct TestTraits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    static constexpr auto reduction_order = strix::ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    static constexpr auto output_form = Form;
    template <unsigned L> __device__ __forceinline__ static float reduce(float* s, float part) {
        const unsigned l = threadIdx.x;
        s[l] = part;
        __syncthreads();
        for (unsigned h = L / 2; h > 0; h >>= 1U) {
            if (l < h) s[l] = s[l] + s[l + h];
            __syncthreads();
        }
        const float r = s[0];
        __syncthreads();
        return r;
    }
    __device__ __forceinline__ static float inv_sqrt(float x) { return 1.0F / sqrtf(x); }
    __device__ __forceinline__ static float load(const float* w, int, unsigned i) { return w[i]; }
};

// The loop form of rmsnorm_row as it was before the batched form (reference, test-only copy).
template <class Traits, unsigned L, bool Residual>
__device__ void loop_row(float* s, const float* xr, float* xw, const float* rr, const float* w, unsigned n, float eps, float* yr) {
    namespace nm = strix::ops::norm;
    float a = 0.0F;
    for (unsigned i = threadIdx.x; i < n; i += L) {
        if constexpr (Residual) {
            const float v = xr[i] + rr[i];
            xw[i] = v;
            a = __builtin_fmaf(v, v, a);
        } else {
            a = __builtin_fmaf(xr[i], xr[i], a);
        }
    }
    const float ss = Traits::template reduce<L>(s, a);
    const float r = Traits::inv_sqrt(ss / static_cast<float>(n) + eps);
    constexpr nm::OutputForm form = nm::output_form_of<Traits>();
    for (unsigned i = threadIdx.x; i < n; i += L) {
        const float v = Residual ? xw[i] : xr[i];
        float y = 0.0F;
        if constexpr (form == nm::OutputForm::plain) y = w ? (v * r) * Traits::load(w, 0, i) : v * r;
        else y = nm::finish<form>(v * r, w ? Traits::load(w, 0, i) : 1.0F, w != nullptr);
        yr[i] = y;
    }
}
template <class Traits, bool Residual, bool Batched>
__global__ __launch_bounds__(256) void norm_kernel(float* x, const float* r, const float* w, unsigned n, float eps, float* y) {
    __shared__ float s[256];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;
    if constexpr (Batched)
        strix::ops::norm::rmsnorm_row<Traits, 256, Residual>(s, x + base, x + base, Residual ? r + base : nullptr, w, 0, n, eps,
                                                             y + base, base, strix::ops::norm::NoStore{});
    else
        loop_row<Traits, 256, Residual>(s, x + base, x + base, Residual ? r + base : nullptr, w, n, eps, y + base);
}
template <class Traits, bool Residual>
void norm_case(unsigned n, unsigned rows, bool weight, std::uint32_t& seed, const char* form) {
    std::vector<float> x(std::size_t{n} * rows), r(x.size()), w(n);
    for (auto& v : x) v = random_float(seed);
    for (auto& v : r) v = random_float(seed);
    for (auto& v : w) v = random_float(seed) + 1.0F;
    float *x1 = upload(x), *x2 = upload(x), *dr = upload(r), *dw = upload(w), *y1 = nullptr, *y2 = nullptr;
    hip(hipMalloc(&y1, x.size() * 4U), "malloc");
    hip(hipMalloc(&y2, x.size() * 4U), "malloc");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_kernel<Traits, Residual, false>), dim3(rows), dim3(256), 0, nullptr, x1, dr,
                       weight ? dw : nullptr, n, 1e-6F, y1);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_kernel<Traits, Residual, true>), dim3(rows), dim3(256), 0, nullptr, x2, dr,
                       weight ? dw : nullptr, n, 1e-6F, y2);
    hip(hipDeviceSynchronize(), "norm sync");
    const auto a = download(y1, x.size()), b = download(y2, x.size()), xa = download(x1, x.size()), xb = download(x2, x.size());
    const bool eq = std::memcmp(a.data(), b.data(), a.size() * 4U) == 0 && std::memcmp(xa.data(), xb.data(), xa.size() * 4U) == 0;
    const std::string what = "norm batched == loop bitwise (y and x): n " + std::to_string(n) + " rows " + std::to_string(rows) +
                             " " + form + (Residual ? " residual" : "") + (weight ? "" : " no-weight");
    check(eq, what.c_str());
    for (float* p : {x1, x2, dr, dw, y1, y2}) hip(hipFree(p), "free");
}
template <bool Batched>
double norm_time(unsigned n, std::uint32_t& seed) {
    using T = TestTraits<strix::ops::norm::OutputForm::bf16_double>;
    std::vector<float> x(n), w(n);
    for (auto& v : x) v = random_float(seed);
    for (auto& v : w) v = random_float(seed);
    float *dx = upload(x), *dw = upload(w), *dy = nullptr;
    hip(hipMalloc(&dy, n * 4U), "malloc");
    constexpr int kLaunches = 2000;
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        hip(hipDeviceSynchronize(), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kLaunches; ++i)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_kernel<T, false, Batched>), dim3(1), dim3(256), 0, nullptr, dx, nullptr, dw, n,
                               1e-6F, dy);
        hip(hipDeviceSynchronize(), "sync");
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kLaunches);
    }
    for (float* p : {dx, dw, dy}) hip(hipFree(p), "free");
    return best;
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("no HIP device: skip\n");
        return 77;
    }
    try {
        std::uint32_t seed = 0x00d0e57U;
        const Shape shapes[] = {
            {"BF16 120832 x 3072 fp32 in (Hy4 head)", 30U, 3072U, 120832U, false},
            {"F32 512 x 3072 fp32 in (Hy4 router)", 0U, 3072U, 512U, false},
            {"F16 777 x 1032 fp32 in", 1U, 1032U, 777U, false},
            {"BF16 100 x 4104 bf16 in", 30U, 4104U, 100U, true},
            {"F32 33 x 8 fp32 in", 0U, 8U, 33U, false},
            {"BF16 50 x 3004 fp32 in (inadmissible width: block form)", 30U, 3004U, 50U, false},
        };
        for (std::size_t i = 0; i < std::size(shapes); ++i) dense_rows_case(shapes[i], seed, i < 2U);

        using Plain = TestTraits<strix::ops::norm::OutputForm::plain>;
        using Bf16d = TestTraits<strix::ops::norm::OutputForm::bf16_double>;
        for (const unsigned n : {128U, 512U, 1024U, 3072U, 5120U, 6144U, 7000U}) {
            norm_case<Bf16d, false>(n, 1U, true, seed, "bf16_double");
            norm_case<Plain, false>(n, 37U, true, seed, "plain");
        }
        norm_case<Plain, false>(3072U, 5U, false, seed, "plain");
        norm_case<Bf16d, true>(3072U, 7U, true, seed, "bf16_double");
        norm_case<Plain, true>(1024U, 3U, true, seed, "plain");
        norm_case<Plain, true>(7000U, 2U, true, seed, "plain");
        for (const unsigned n : {3072U, 1024U, 512U})
            std::printf("TIME norm 1 block n %u: loop %.2f us batched %.2f us per launch (2000 back-to-back)\n", n,
                        norm_time<false>(n, seed), norm_time<true>(n, seed));
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
