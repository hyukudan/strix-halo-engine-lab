// A8 rows stored by their producer (src/ops/a8/a8_quant_device.hpp) == the separate passes they replace, bitwise:
//  1. rmsnorm with the A8 NormStore == rmsnorm_kernel, then ds41::enqueue_quantize_a8 of its output (FP32 rows + codes /
//     deltas / sums, plain and aliasing row pitches);
//  2. quantize_rows_kernel<SigmoidGate> == sigmoid_gate_kernel, then the quantiser;
//  3. quantize_rows_kernel<Swiglu> == swiglu_kernel, then the quantiser;
//  4. rmsnorm_strided_kernel in place on the first n values of wider rows == gather, rmsnorm_kernel, scatter;
//  5. the norm row's register form (loads before the FMA chain) == its loop form (Regs = 0): y and the residual x, both
//     output forms, with and without weights, widths up to and past the register capacity (L * kRowRegs).
// Data: random FP32 with all-zero and tiny groups and a wide exponent range. Exit 77 when no HIP device is present.
#include "ops/a8/a8_quant_device.hpp"
#include "ops/glu/glu_device.hpp"
#include "ops/norm/norm_device.hpp"
#include "ops/norm/norm_presets.hpp"
#include "strix/a8_row_pitch.hpp"
#include "strix/hip_ds41_formats.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace a8 = strix::ops::a8;
namespace glu = strix::ops::glu;
namespace nm = strix::ops::norm;
using Traits = nm::DeclaredF32Traits<nm::OutputForm::bf16_double>;

int failures = 0;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <class T>
struct Buf {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buf(std::size_t count) : n(count) {
        ok(hipMalloc(&p, count * sizeof(T) + 1U), "malloc");
        ok(hipMemset(p, 0x5a, count * sizeof(T) + 1U), "memset");
    }
    ~Buf() { (void)hipFree(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    void put(const std::vector<T>& v) { ok(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> get() const {
        std::vector<T> v(n);
        ok(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return v;
    }
};

template <class T>
void same(const char* what, const Buf<T>& a, const Buf<T>& b) {
    const auto x = a.get(), y = b.get();
    if (std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) != 0) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

std::vector<float> data(std::size_t rows, std::size_t width, std::uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::uniform_int_distribution<int> e(-12, 6);
    std::vector<float> v(rows * width);
    for (auto& x : v) x = std::ldexp(u(g), e(g));
    for (std::size_t r = 0; r < rows; r += 5)  // an all-zero group and a tiny group in some rows
        for (std::size_t i = 0; i < 32U && 64U <= width; ++i) {
            v[r * width + i] = 0.0F;
            v[r * width + 32U + i] = std::ldexp(u(g), -125);
        }
    return v;
}

struct A8 {
    Buf<std::int8_t> codes;
    Buf<float> deltas;
    Buf<std::int32_t> sums;
    unsigned pitch;
    A8(std::size_t count, unsigned p) : codes(count * p), deltas(count * p / 32U), sums(count * p / 32U), pitch(p) {}
    a8::Rows rows() const { return {codes.p, deltas.p, sums.p, pitch}; }
};

void compare(const char* what, const A8& a, const A8& b) {
    char name[160];
    std::snprintf(name, sizeof name, "%s codes", what);
    same(name, a.codes, b.codes);
    std::snprintf(name, sizeof name, "%s deltas", what);
    same(name, a.deltas, b.deltas);
    std::snprintf(name, sizeof name, "%s sums", what);
    same(name, a.sums, b.sums);
}

void norm_case(unsigned rows, unsigned n) {
    const unsigned pitch = strix::a8_row_pitch(n);
    Buf<float> x(std::size_t{rows} * n), w(n), y0(std::size_t{rows} * n), y1(std::size_t{rows} * n);
    x.put(data(rows, n, 7U + n));
    w.put(data(1, n, 11U + n));
    A8 q0(rows, pitch), q1(rows, pitch);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nm::rmsnorm_kernel<Traits, 256>), dim3(rows), dim3(256), 0, nullptr, x.p, w.p, 0, n, 1e-5F, y0.p);
    strix::ds41::enqueue_quantize_a8(y0.p, n, rows, pitch, q0.codes.p, q0.deltas.p, q0.sums.p, nullptr);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nm::rmsnorm_fused_kernel<Traits, 256, false, a8::NormStore>), dim3(rows), dim3(256), 0, nullptr, x.p,
                       nullptr, w.p, 0, n, 1e-5F, y1.p, a8::NormStore{q1.rows(), n});
    ok(hipDeviceSynchronize(), "norm case");
    char name[64];
    std::snprintf(name, sizeof name, "norm n %u y", n);
    same(name, y0, y1);
    std::snprintf(name, sizeof name, "norm n %u", n);
    compare(name, q0, q1);
}

template <class Pre, class Separate>
void pre_case(const char* what, unsigned rows, unsigned width, const Buf<float>& sep, Pre pre, Separate separate) {
    const unsigned pitch = strix::a8_row_pitch(width);
    A8 q0(rows, pitch), q1(rows, pitch);
    separate();
    strix::ds41::enqueue_quantize_a8(sep.p, width, rows, pitch, q0.codes.p, q0.deltas.p, q0.sums.p, nullptr);
    a8::launch_quantize_rows(pre, width, rows, q1.rows(), nullptr);
    ok(hipDeviceSynchronize(), what);
    compare(what, q0, q1);
}

void strided_case(unsigned rows, unsigned n, unsigned stride) {
    Buf<float> a(std::size_t{rows} * stride), b(std::size_t{rows} * stride), tight(std::size_t{rows} * n), w(n);
    const auto v = data(rows, stride, 23U);
    a.put(v);
    b.put(v);
    w.put(data(1, n, 29U));
    ok(hipMemcpy2D(tight.p, n * 4U, a.p, stride * 4U, n * 4U, rows, hipMemcpyDeviceToDevice), "gather");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nm::rmsnorm_kernel<Traits, 256>), dim3(rows), dim3(256), 0, nullptr, tight.p, w.p, 0, n, 1e-5F, tight.p);
    ok(hipMemcpy2D(a.p, stride * 4U, tight.p, n * 4U, n * 4U, rows, hipMemcpyDeviceToDevice), "scatter");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nm::rmsnorm_strided_kernel<Traits, 256>), dim3(rows), dim3(256), 0, nullptr, b.p, std::size_t{stride},
                       w.p, 0, n, 1e-5F, b.p, std::size_t{stride});
    ok(hipDeviceSynchronize(), "strided case");
    same("strided norm in place", a, b);
}

// One block per row over rmsnorm_row with an explicit register capacity (0 = the loop form for every width).
template <class T, bool Residual, unsigned Regs>
__global__ void row_form_kernel(float* x, const float* r, const float* w, unsigned n, float eps, float* y) {
    __shared__ float s[256];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;
    nm::rmsnorm_row<T, 256, Residual, nm::NoStore, Regs>(s, x + base, x + base, Residual ? r + base : nullptr, w, 0, n, eps,
                                                         y + base, base, nm::NoStore{});
}

template <class T, bool Residual>
void row_form_case(const char* form, unsigned rows, unsigned n, bool weighted) {
    const std::size_t count = std::size_t{rows} * n;
    Buf<float> x0(count), x1(count), r(count), w(n), y0(count), y1(count);
    const auto xv = data(rows, n, 43U + n);
    x0.put(xv);
    x1.put(xv);
    r.put(data(rows, n, 47U + n));
    w.put(data(1, n, 53U + n));
    const float* wp = weighted ? w.p : nullptr;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(row_form_kernel<T, Residual, 0U>), dim3(rows), dim3(256), 0, nullptr, x0.p, r.p, wp, n, 1e-6F, y0.p);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(row_form_kernel<T, Residual, nm::kRowRegs>), dim3(rows), dim3(256), 0, nullptr, x1.p, r.p, wp, n,
                       1e-6F, y1.p);
    ok(hipDeviceSynchronize(), "row form case");
    char name[96];
    std::snprintf(name, sizeof name, "row form %s residual %d weighted %d n %u y", form, Residual ? 1 : 0, weighted ? 1 : 0, n);
    same(name, y0, y1);
    std::snprintf(name, sizeof name, "row form %s residual %d weighted %d n %u x", form, Residual ? 1 : 0, weighted ? 1 : 0, n);
    same(name, x0, x1);
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("no HIP device: skipped\n");
        return 77;
    }
    const unsigned rows = 37U;
    for (const unsigned n : {512U, 1536U, 3072U, 4096U}) norm_case(rows, n);  // 4096: aliasing pitch (+128)
    for (const unsigned width : {3072U, 4096U, 8192U}) {
        const std::size_t count = std::size_t{rows} * width;
        Buf<float> x(count), g(count), u(count), y(count);
        x.put(data(rows, width, 31U + width));
        g.put(data(rows, width, 37U + width));
        u.put(data(rows, width, 41U + width));
        char name[64];
        std::snprintf(name, sizeof name, "sigmoid gate width %u", width);
        pre_case(name, rows, width, y, a8::SigmoidGate<glu::DeclaredExp, glu::Round::bf16>{x.p, g.p},
                 [&] { glu::launch_sigmoid_gate<glu::DeclaredExp, glu::Round::bf16>(x.p, g.p, count, y.p, nullptr); });
        std::snprintf(name, sizeof name, "swiglu width %u", width);
        pre_case(name, rows, width, y, a8::Swiglu<glu::DeclaredExp, glu::Round::bf16>{g.p, u.p, glu::no_clamp},
                 [&] { glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(g.p, u.p, count, glu::no_clamp, y.p, nullptr); });
    }
    strided_case(rows, 512U, 576U);
    using Plain = nm::DeclaredF32Traits<nm::OutputForm::plain>;
    for (const unsigned n : {64U, 300U, 512U, 3072U, 5120U, 7168U, 256U * nm::kRowRegs, 256U * nm::kRowRegs + 96U})
        for (const bool weighted : {true, false}) {
            row_form_case<Traits, false>("bf16_double", 5U, n, weighted);
            row_form_case<Traits, true>("bf16_double", 5U, n, weighted);
            row_form_case<Plain, false>("plain", 5U, n, weighted);
            row_form_case<Plain, true>("plain", 5U, n, weighted);
        }
    if (failures != 0) {
        std::printf("hip a8 producer tests: %d FAILED\n", failures);
        return 1;
    }
    std::printf("hip a8 producer tests: all passed\n");
    return 0;
}
