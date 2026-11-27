// q8_rows head rows (strix/decode_head_rows.hpp + q8_rows::launch_heads), synthetic Q8_0 data, no model:
//  1. identity: ONE launch_heads launch == H per-head q8_rows::launch launches of the same Site, bitwise, at the MLA absorb
//     geometries (K_b: width 192, 512 outputs / head, input stride 256; V_b: width 512, 256 outputs / head, stride 512;
//     32 and 64 heads); plus a host-reference sanity bound.
//  2. timing (printed, not asserted): per layer, the incumbent per-head launch_hip_dense_rows loop, the per-head q8_rows
//     loop and the one-launch head form; and the streaming rate of launch_hip_dense_rows vs q8_rows on the large model-B
//     Q8_0 projection shapes. Weights rotate over copies larger than the 32 MB last-level cache.
//  3. verify windows (q8_rows::launch_heads_rows_f32, the stock HeadsF32 arithmetic): R = 1..8 input rows in ONE launch ==
//     launch_heads_f32 on each row alone, bitwise, at the same geometries; --bench times R = 8 as the former pairs of rows
//     (four HeadsRowsF32<2> launches) against the one-launch window.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "../src/hip_q8_rows_gemv.hpp"
#include "../src/hip_q8_rows_sites.hpp"
#include "strix/hip_lm_head_launcher.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
#define CHECK(x)                                                                              \
    do {                                                                                      \
        const hipError_t e_ = (x);                                                            \
        if (e_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                     \
        }                                                                                     \
    } while (0)

struct PlainSite {
    using Input = float;
    static constexpr unsigned rows = 1U;
    const void* input[1];
    strix::q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* out;
    __device__ static float coefficient(unsigned scale, std::int8_t code) {
        return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(scale))) * static_cast<float>(code);
    }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) { return lower + upper; }
    __device__ void store(std::uint32_t o, unsigned, float value) const { out[o] = value; }
};
struct HeadSite : PlainSite {
    static constexpr bool head_input = true;
    std::uint32_t head_outputs, input_stride;
};

std::uint32_t rng = 0x2545F491U;
std::uint32_t next_u32() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
std::uint16_t f16_bits(float v) {
    const _Float16 h = static_cast<_Float16>(v);
    std::uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
float f16_value(std::uint16_t b) {
    _Float16 h;
    std::memcpy(&h, &b, 2);
    return static_cast<float>(h);
}
std::vector<unsigned char> q8_rows_host(std::uint64_t rows, std::uint32_t width) {
    std::vector<unsigned char> w(rows * (width / 32U) * 34U);
    for (std::size_t at = 0; at < w.size(); at += 34U) {
        const std::uint16_t s = f16_bits(0.001F + 0.02F * static_cast<float>(next_u32() % 1000U) / 1000.0F);
        std::memcpy(&w[at], &s, 2);
        for (unsigned i = 0; i < 32U; ++i) w[at + 2U + i] = static_cast<unsigned char>(static_cast<int>(next_u32() % 255U) - 127);
    }
    return w;
}
template <class T> T* to_device(const std::vector<T>& h) {
    T* d = nullptr;
    CHECK(hipMalloc(&d, h.size() * sizeof(T)));
    CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

struct Geometry {
    const char* name;
    std::uint32_t heads, head_outputs, width, stride;
};

void per_head_q8(const unsigned char* w, const float* x, float* y, const Geometry& g, hipStream_t s) {
    const std::uint64_t head_bytes = std::uint64_t{g.head_outputs} * (g.width / 32U) * 34U;
    for (std::uint32_t h = 0; h < g.heads; ++h) {
        PlainSite p{};
        p.input[0] = x + std::size_t{h} * g.stride;
        p.segments = strix::q8_rows::segments({{w + h * head_bytes, g.head_outputs}});
        p.width = g.width;
        p.outputs = g.head_outputs;
        p.out = y + std::size_t{h} * g.head_outputs;
        strix::q8_rows::launch(p, false, s);
    }
}
void heads_q8(const unsigned char* w, const float* x, float* y, const Geometry& g, hipStream_t s) {
    HeadSite p{};
    p.input[0] = x;
    p.segments = strix::q8_rows::segments({{w, g.heads * g.head_outputs}});
    p.width = g.width;
    p.outputs = g.heads * g.head_outputs;
    p.out = y;
    p.head_outputs = g.head_outputs;
    p.input_stride = g.stride;
    strix::q8_rows::launch_heads(p, s);
}
void per_head_dense(const unsigned char* w, const float* x, float* y, const Geometry& g, hipStream_t s) {
    const std::uint64_t head_bytes = std::uint64_t{g.head_outputs} * (g.width / 32U) * 34U;
    for (std::uint32_t h = 0; h < g.heads; ++h) {
        strix::HipDenseRowsView v;
        v.x = x + std::size_t{h} * g.stride;
        v.input = strix::HipActivationStorage::fp32;
        v.weights = w + h * head_bytes;
        v.ggml_type = 8U;
        v.width = g.width;
        v.outputs = g.head_outputs;
        v.out = y + std::size_t{h} * g.head_outputs;
        strix::launch_hip_dense_rows(v, s);
    }
}

template <class F> double time_us(F&& f, unsigned reps, hipStream_t s) {
    f(0U);
    CHECK(hipStreamSynchronize(s));
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    CHECK(hipEventRecord(a, s));
    for (unsigned i = 0; i < reps; ++i) f(i);
    CHECK(hipEventRecord(b, s));
    CHECK(hipEventSynchronize(b));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b));
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    return 1000.0 * ms / reps;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    hipStream_t s;
    CHECK(hipStreamCreate(&s));
    int failures = 0;
    const Geometry geos[] = {{"kb64", 64U, 512U, 192U, 256U}, {"kb32", 32U, 512U, 192U, 256U},
                             {"vb64", 64U, 256U, 512U, 512U}, {"vb32", 32U, 256U, 512U, 512U}};
    for (const auto& g : geos) {
        const std::uint64_t outputs = std::uint64_t{g.heads} * g.head_outputs;
        const auto wh = q8_rows_host(outputs, g.width);
        std::vector<float> xh(std::size_t{g.heads} * g.stride);
        for (auto& v : xh) v = static_cast<float>(static_cast<std::int32_t>(next_u32() % 2001U) - 1000) / 250.0F;
        const auto* w = to_device(wh);
        const auto* x = to_device(xh);
        float *ya = nullptr, *yb = nullptr;
        CHECK(hipMalloc(&ya, outputs * 4U));
        CHECK(hipMalloc(&yb, outputs * 4U));
        CHECK(hipMemset(ya, 0xff, outputs * 4U));
        CHECK(hipMemset(yb, 0x7f, outputs * 4U));
        heads_q8(w, x, ya, g, s);
        per_head_q8(w, x, yb, g, s);
        CHECK(hipStreamSynchronize(s));
        std::vector<float> a(outputs), b(outputs);
        CHECK(hipMemcpy(a.data(), ya, outputs * 4U, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(), yb, outputs * 4U, hipMemcpyDeviceToHost));
        const bool bitwise = std::memcmp(a.data(), b.data(), outputs * 4U) == 0;
        double worst = 0;  // host reference, |err| / sum |terms|
        for (std::uint64_t o = 0; o < outputs; ++o) {
            const float* xr = xh.data() + (o / g.head_outputs) * g.stride;
            const unsigned char* row = wh.data() + o * (g.width / 32U) * 34U;
            double ref = 0, mag = 0;
            for (std::uint32_t k = 0; k < g.width; ++k) {
                const unsigned char* blk = row + (k / 32U) * 34U;
                std::uint16_t sb;
                std::memcpy(&sb, blk, 2);
                const double t = static_cast<double>(f16_value(sb)) * static_cast<std::int8_t>(blk[2U + k % 32U]) * xr[k];
                ref += t;
                mag += std::fabs(t);
            }
            worst = std::max(worst, std::fabs(a[o] - ref) / (mag + 1e-30));
        }
        const bool close = worst < 1e-5;
        std::printf("HEADS %s heads=%u outputs/head=%u width=%u bitwise_vs_per_head=%d host_rel_err=%.2e %s\n", g.name, g.heads,
                    g.head_outputs, g.width, bitwise ? 1 : 0, worst, bitwise && close ? "PASS" : "FAIL");
        if (!bitwise || !close) ++failures;
        if (bench) {
            constexpr unsigned kCopies = 8U;  // 8 layers' worth of weights: > 32 MB last-level cache
            std::vector<unsigned char*> ws(kCopies);
            for (auto& c : ws) c = to_device(wh);
            const auto bytes = static_cast<double>(wh.size());
            const double t_dense = time_us([&](unsigned i) { per_head_dense(ws[i % kCopies], x, yb, g, s); }, 40U, s);
            const double t_loop = time_us([&](unsigned i) { per_head_q8(ws[i % kCopies], x, yb, g, s); }, 40U, s);
            const double t_heads = time_us([&](unsigned i) { heads_q8(ws[i % kCopies], x, ya, g, s); }, 200U, s);
            std::printf("TIME %s us/layer: dense_rows_loop=%.1f q8_rows_loop=%.1f heads=%.1f (heads %.1f GB/s)\n", g.name, t_dense,
                        t_loop, t_heads, bytes / (t_heads * 1e3));
            for (auto* c : ws) CHECK(hipFree(c));
        }
        CHECK(hipFree(const_cast<unsigned char*>(w)));
        CHECK(hipFree(const_cast<float*>(x)));
        CHECK(hipFree(ya));
        CHECK(hipFree(yb));
    }
    // 3. verify windows: R rows in one head launch == each row's one-row launch (stock HeadsF32 arithmetic), bitwise.
    for (const auto& g : geos) {
        namespace qr = strix::q8_rows;
        const std::uint64_t outputs = std::uint64_t{g.heads} * g.head_outputs;
        const std::size_t x_row = std::size_t{g.heads} * g.stride, y_row = outputs;
        constexpr std::uint32_t kR = 8U;
        const auto wh = q8_rows_host(outputs, g.width);
        std::vector<float> xh(kR * x_row);
        for (auto& v : xh) v = static_cast<float>(static_cast<std::int32_t>(next_u32() % 2001U) - 1000) / 250.0F;
        const auto* w = to_device(wh);
        const auto* x = to_device(xh);
        float *ya = nullptr, *yb = nullptr;
        CHECK(hipMalloc(&ya, kR * y_row * 4U));
        CHECK(hipMalloc(&yb, kR * y_row * 4U));
        int bad = 0;
        for (std::uint32_t R = 1U; R <= kR; ++R) {
            CHECK(hipMemset(ya, 0xff, kR * y_row * 4U));
            CHECK(hipMemset(yb, 0x7f, kR * y_row * 4U));
            qr::launch_heads_rows_f32(w, g.heads, g.width, g.head_outputs, x, g.stride, x_row, ya, g.head_outputs, y_row, R, s);
            for (std::uint32_t r = 0; r < R; ++r)
                qr::launch_heads_f32(w, g.heads, g.width, g.head_outputs, x + r * x_row, g.stride, yb + r * y_row, g.head_outputs, s);
            CHECK(hipStreamSynchronize(s));
            std::vector<float> a(R * y_row), b(R * y_row);
            CHECK(hipMemcpy(a.data(), ya, R * y_row * 4U, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(b.data(), yb, R * y_row * 4U, hipMemcpyDeviceToHost));
            if (std::memcmp(a.data(), b.data(), R * y_row * 4U) != 0) ++bad;
        }
        std::printf("WINDOW %s heads=%u R=1..%u bitwise_vs_rows %d/%u %s\n", g.name, g.heads, kR, static_cast<int>(kR) - bad, kR, bad ? "FAIL" : "PASS");
        failures += bad;
        if (bench) {
            constexpr unsigned kCopies = 8U;  // > 32 MB last-level cache
            std::vector<unsigned char*> ws(kCopies);
            for (auto& c : ws) c = to_device(wh);
            const auto pairs = [&](const unsigned char* wt) {
                for (std::uint32_t r = 0; r < kR; r += 2U) {
                    qr::HeadsRowsF32<2U> site{};
                    site.input[0] = x + r * x_row;
                    site.input[1] = x + (r + 1U) * x_row;
                    site.segments = qr::segments({{wt, g.heads * g.head_outputs}});
                    site.width = g.width;
                    site.outputs = g.heads * g.head_outputs;
                    site.head_outputs = g.head_outputs;
                    site.input_stride = g.stride;
                    site.out_stride = g.head_outputs;
                    site.row_stride = y_row;
                    site.out = yb + r * y_row;
                    qr::launch_heads(site, s);
                }
            };
            const double t_pairs = time_us([&](unsigned i) { pairs(ws[i % kCopies]); }, 100U, s);
            const double t_window = time_us([&](unsigned i) {
                qr::launch_heads_rows_f32(ws[i % kCopies], g.heads, g.width, g.head_outputs, x, g.stride, x_row, ya, g.head_outputs,
                                          y_row, kR, s); }, 100U, s);
            std::printf("TIME_R8 %s us/layer: pairs=%.1f window=%.1f (window %.1f GB/s)\n", g.name, t_pairs, t_window,
                        static_cast<double>(wh.size()) / (t_window * 1e3));
            for (auto* c : ws) CHECK(hipFree(c));
        }
        CHECK(hipFree(const_cast<unsigned char*>(w)));
        CHECK(hipFree(const_cast<float*>(x)));
        CHECK(hipFree(ya));
        CHECK(hipFree(yb));
    }
    if (bench) {
        // U2: one-row streaming rate of the incumbent head-family dense rows vs the q8_rows direct form, model-B Q8_0 shapes.
        struct Shape { const char* name; std::uint32_t outputs, width; };
        const Shape shapes[] = {{"attn_gate64_16384x3072", 16384U, 3072U}, {"attn_out64_3072x16384", 3072U, 16384U},
                                {"attn_q_b64_16384x1024", 16384U, 1024U}, {"q_a+kv_a_1600x3072", 1600U, 3072U},
                                {"shexp_up_1536x3072", 1536U, 3072U}, {"latent_up_3072x1536", 3072U, 1536U}};
        for (const auto& sh : shapes) {
            const auto wh = q8_rows_host(sh.outputs, sh.width);
            std::vector<float> xh(sh.width, 0.5F);
            const unsigned copies = static_cast<unsigned>(std::max<std::size_t>(2U, (96U << 20) / wh.size() + 1U));
            std::vector<unsigned char*> ws(copies);
            for (auto& c : ws) c = to_device(wh);
            const auto* x = to_device(xh);
            float* y = nullptr;
            CHECK(hipMalloc(&y, std::size_t{sh.outputs} * 4U));
            const double t_dense = time_us([&](unsigned i) {
                strix::HipDenseRowsView v;
                v.x = x;
                v.input = strix::HipActivationStorage::fp32;
                v.weights = ws[i % copies];
                v.ggml_type = 8U;
                v.width = sh.width;
                v.outputs = sh.outputs;
                v.out = y;
                strix::launch_hip_dense_rows(v, s);
            }, 50U, s);
            const double t_q8 = time_us([&](unsigned i) {
                PlainSite p{};
                p.input[0] = x;
                p.segments = strix::q8_rows::segments({{ws[i % copies], sh.outputs}});
                p.width = sh.width;
                p.outputs = sh.outputs;
                p.out = y;
                strix::q8_rows::launch(p, false, s);
            }, 50U, s);
            const double bytes = static_cast<double>(wh.size());
            std::printf("BW %s MB=%.2f dense_rows=%.1fus %.1fGB/s q8_rows=%.1fus %.1fGB/s\n", sh.name, bytes / 1e6, t_dense,
                        bytes / (t_dense * 1e3), t_q8, bytes / (t_q8 * 1e3));
            for (auto* c : ws) CHECK(hipFree(c));
            CHECK(hipFree(const_cast<float*>(x)));
            CHECK(hipFree(y));
        }
    }
    CHECK(hipStreamDestroy(s));
    if (failures) {
        std::printf("q8_rows heads: %d FAIL\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("q8_rows heads: identity PASS (4 geometries)\n");
    return EXIT_SUCCESS;
}
