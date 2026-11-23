// Device checks of the MODEL_B dense-side shared pieces (synthetic data, no model):
//  - launch_hip_dense_rows (head family): F32 weights 512 x 3072 (router geometry), BF16 weights 1024 x 3072 (head rows),
//    BF16 and FP32 inputs, 2 rows with strides: FP32 out vs a double host sum, |err| <= 1e-5 * sum|w x| + 1e-6;
//  - ops::rope Round::bf16 on 64 heads x 256 (trailing 64), theta 1e4 / 1e7 native tables: bitwise == reference_apply;
//    Round::none (the DS41 instantiation): |err| <= 2^-20 relative (contraction only);
//  - ops::glu swiglu (limit 10 and none) and sigmoid gate, Round::bf16 and none, DeclaredExp: bitwise == host
//    (this TU compiles without contraction, as a MODEL_B binder must).
// Exit 77 when no HIP device is present.
// Compile-time regression: shared norm presets must compile without importing DS41 model headers.
#include "ops/norm/norm_presets.hpp"
#include "ops/glu/glu_device.hpp"
#include "ops/rope/rope_device.hpp"
#include "strix/hip_lm_head_launcher.hpp"

#include <hip/hip_runtime.h>

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
float bf(float x) { return strix::numeric::round_to_bf16(x); }
std::uint16_t bf_bits(float x) {
    const float r = bf(x);
    std::uint32_t u;
    std::memcpy(&u, &r, 4);
    return static_cast<std::uint16_t>(u >> 16U);
}
std::uint32_t lcg = 777U;
float next_value(float scale) {
    lcg = lcg * 1664525U + 1013904223U;
    return (static_cast<float>(lcg >> 8U) / 16777216.0F - 0.5F) * 2.0F * scale;
}
template <class T> T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    hip(hipMalloc(reinterpret_cast<void**>(&d), h.size() * sizeof(T)), "malloc");
    hip(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}
template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hip(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}

void dense_rows(std::uint32_t ggml_type, std::uint32_t outputs, bool bf16_input, const char* what) {
    constexpr std::uint32_t width = 3072, rows = 2, x_stride = 3200, out_stride = 0;
    std::vector<float> w(std::size_t{outputs} * width), x(std::size_t{rows} * x_stride, 0.0F);
    for (auto& v : w) v = ggml_type == 30 ? bf(next_value(0.05F)) : next_value(0.05F);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t k = 0; k < width; ++k) x[std::size_t{r} * x_stride + k] = bf(next_value(2.0F));
    void* dw = nullptr;
    if (ggml_type == 30) {
        std::vector<std::uint16_t> wb(w.size());
        for (std::size_t i = 0; i < w.size(); ++i) wb[i] = bf_bits(w[i]);
        dw = upload(wb);
    } else {
        dw = upload(w);
    }
    void* dx = nullptr;
    if (bf16_input) {
        std::vector<std::uint16_t> xb(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) xb[i] = bf_bits(x[i]);
        dx = upload(xb);
    } else {
        dx = upload(x);
    }
    float* dout = nullptr;
    hip(hipMalloc(reinterpret_cast<void**>(&dout), std::size_t{rows} * outputs * sizeof(float)), "malloc out");
    strix::HipDenseRowsView v;
    v.x = dx;
    v.input = bf16_input ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;
    v.weights = dw;
    v.ggml_type = ggml_type;
    v.width = width;
    v.outputs = outputs;
    v.rows = rows;
    v.x_stride = x_stride;
    v.out_stride = out_stride;
    v.out = dout;
    strix::launch_hip_dense_rows(v, nullptr);
    hip(hipDeviceSynchronize(), "dense rows sync");
    const auto got = download(dout, std::size_t{rows} * outputs);
    bool ok = true;
    double worst = 0.0;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < outputs; ++o) {
            double sum = 0.0, mag = 0.0;
            for (std::uint32_t k = 0; k < width; ++k) {
                const double p = double{w[std::size_t{o} * width + k]} * double{x[std::size_t{r} * x_stride + k]};
                sum += p;
                mag += std::fabs(p);
            }
            const double err = std::fabs(double{got[std::size_t{r} * outputs + o]} - sum);
            worst = std::fmax(worst, err / (mag + 1e-30));
            ok = ok && err <= 1e-5 * mag + 1e-6;
        }
    std::printf("     %s worst rel err %.3g\n", what, worst);
    check(ok, what);
    hip(hipFree(dw), "free");
    hip(hipFree(dx), "free");
    hip(hipFree(dout), "free");
}

void rope(strix::ops::rope::Round round, float theta, const char* what) {
    using strix::ops::rope::Round;
    constexpr std::uint32_t heads = 64, head = 256, pairs = 32, nope = 192, rows = 4;
    const std::uint32_t positions[rows] = {0U, 1U, 511U, 4095U};
    std::vector<float> ct(4096U * pairs), st(4096U * pairs);
    strix::ops::rope::build_native_table(theta, pairs, 0U, 4096U, Round::bf16, ct.data(), st.data());
    std::vector<float> x(std::size_t{rows} * heads * head);
    for (auto& v : x) v = bf(next_value(4.0F));
    std::vector<std::uint32_t> pos(positions, positions + rows);
    auto* dx = upload(x);
    auto* dc = upload(ct);
    auto* ds = upload(st);
    auto* dp = upload(pos);
    strix::ops::rope::View v{dx, rows, heads, heads * head, head, nope, pairs, dc, ds, dp, false};
    if (round == Round::bf16) strix::ops::rope::launch<Round::bf16>(v, nullptr);
    else strix::ops::rope::launch<Round::none>(v, nullptr);
    hip(hipGetLastError(), "rope launch");
    hip(hipDeviceSynchronize(), "rope sync");
    const auto got = download(dx, x.size());
    strix::ops::rope::View h{x.data(), rows, heads, heads * head, head, nope, pairs, ct.data(), st.data(), pos.data(), false};
    strix::ops::rope::reference_apply(h, round);
    bool ok = true;
    for (std::size_t i = 0; i < x.size(); ++i)
        ok = ok && (round == Round::bf16 ? std::memcmp(&got[i], &x[i], 4) == 0
                                         : std::fabs(got[i] - x[i]) <= 9.6e-7F * (std::fabs(x[i]) + 4.0F));
    check(ok, what);
    hip(hipFree(dx), "free");
    hip(hipFree(dc), "free");
    hip(hipFree(ds), "free");
    hip(hipFree(dp), "free");
}

template <strix::ops::glu::Round R> void glu(float limit, const char* what) {
    using strix::ops::glu::DeclaredExp;
    constexpr std::size_t n = 16U * 1536U;
    std::vector<float> g(n), u(n), x(n);
    for (std::size_t i = 0; i < n; ++i) {
        g[i] = bf(next_value(24.0F));
        u[i] = bf(next_value(24.0F));
        x[i] = next_value(3.0F);
    }
    auto* dg = upload(g);
    auto* du = upload(u);
    auto* dx = upload(x);
    float* da = nullptr;
    hip(hipMalloc(reinterpret_cast<void**>(&da), n * sizeof(float)), "malloc");
    strix::ops::glu::launch_swiglu<DeclaredExp, R>(dg, du, n, limit, da, nullptr);
    strix::ops::glu::launch_sigmoid_gate<DeclaredExp, R>(dx, dg, n, dx, nullptr);  // in place
    hip(hipGetLastError(), "glu launch");
    hip(hipDeviceSynchronize(), "glu sync");
    const auto a = download(da, n), y = download(dx, n);
    bool ok_a = true, ok_y = true;
    for (std::size_t i = 0; i < n; ++i) {
        const float wa = strix::ops::glu::swiglu<DeclaredExp, R>(g[i], u[i], limit);
        const float wy = strix::ops::glu::sigmoid_gate<DeclaredExp, R>(x[i], g[i]);
        ok_a = ok_a && std::memcmp(&a[i], &wa, 4) == 0;
        ok_y = ok_y && std::memcmp(&y[i], &wy, 4) == 0;
    }
    check(ok_a, what);
    check(ok_y, "sigmoid gate (same round) == host");
    hip(hipFree(dg), "free");
    hip(hipFree(du), "free");
    hip(hipFree(dx), "free");
    hip(hipFree(da), "free");
}

// Linear epilogue (bias add + one BF16 rounding, row-broadcast bias; unit bias; in place) == host formula, bitwise.
void epilogue() {
    using strix::ops::glu::Round;
    constexpr std::size_t width = 1536U, n = 4U * width;
    std::vector<float> x(n), b(width);
    for (auto& v : x) v = next_value(30.0F);
    for (auto& v : b) v = next_value(2.0F);
    auto* dx = upload(x);
    auto* db = upload(b);
    float* dy = nullptr;
    hip(hipMalloc(reinterpret_cast<void**>(&dy), n * sizeof(float)), "malloc");
    strix::ops::glu::launch_epilogue<Round::bf16>(dx, db, width, n, dy, nullptr);
    strix::ops::glu::launch_epilogue<Round::bf16>(dx, nullptr, width, n, dx, nullptr);  // in place, unit bias
    hip(hipGetLastError(), "epilogue launch");
    hip(hipDeviceSynchronize(), "epilogue sync");
    const auto y = download(dy, n), r = download(dx, n);
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) {
        const float wy = strix::ops::glu::epilogue<Round::bf16>(x[i], b.data(), i % width);
        const float wr = strix::ops::glu::epilogue<Round::bf16>(x[i], nullptr, 0U);
        ok = ok && std::memcmp(&y[i], &wy, 4) == 0 && std::memcmp(&r[i], &wr, 4) == 0;
    }
    check(ok, "linear epilogue bf16 (bias, unit, in place) == host (bitwise)");
    hip(hipFree(dx), "free");
    hip(hipFree(db), "free");
    hip(hipFree(dy), "free");
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP no HIP device\n");
        return 77;
    }
    try {
        dense_rows(0, 512, true, "dense rows F32 512x3072 bf16 input (router)");
        dense_rows(0, 16, false, "dense rows F32 16x3072 fp32 input (indexer.proj)");
        dense_rows(30, 1024, true, "dense rows BF16 1024x3072 bf16 input (head rows)");
        rope(strix::ops::rope::Round::bf16, 1.0e4F, "rope bf16 theta 1e4 == reference (bitwise)");
        rope(strix::ops::rope::Round::bf16, 1.0e7F, "rope bf16 theta 1e7 == reference (bitwise)");
        rope(strix::ops::rope::Round::none, 1.0e7F, "rope none (DS41 form) ~ reference");
        glu<strix::ops::glu::Round::bf16>(10.0F, "swiglu bf16 limit 10 == host (bitwise)");
        glu<strix::ops::glu::Round::bf16>(strix::ops::glu::no_clamp, "swiglu bf16 no clamp == host (bitwise)");
        glu<strix::ops::glu::Round::none>(10.0F, "swiglu none limit 10 == host (bitwise)");
        epilogue();
    } catch (const std::exception& e) {
        std::printf("FAIL exception %s\n", e.what());
        return 1;
    }
    if (failures) return 1;
    std::printf("PASS modelb dense ops (dense rows F32/BF16, rope, swiglu, sigmoid gate)\n");
    return 0;
}
