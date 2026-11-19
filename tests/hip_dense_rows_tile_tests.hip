// Device checks of the rows tile of launch_hip_dense_rows (strix/dense_rows_tile.hpp), synthetic data, no model:
//  1. identity: rows form (one launch) == one single-row launch per row (row_loop = true), BITWISE, on
//     F32 512 x 3072 fp32 input rows 2048 (model-B router), F32 16 x 3072 fp32 rows 37 (indexer proj, ragged rows),
//     BF16 1003 x 3000 bf16 input rows 13 with strides (ragged outputs, width not a multiple of 256),
//     Q8_0 300 x 2048 bf16 rows 9, F16 64 x 768 fp32 rows 2; outputs outside the stored range stay untouched.
//  2. timing (printed, not asserted): router geometry, rows 2048: per-row loop vs rows tile, median of 5.
// Exit 77 when no HIP device is present.
#include "strix/hip_lm_head_launcher.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
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
std::uint16_t f16_bits(std::uint32_t& s) {  // finite half, exponent 9 .. 16
    const auto u = lcg(s);
    return static_cast<std::uint16_t>(((u >> 31U) << 15U) | ((9U + (u >> 4U) % 8U) << 10U) | ((u >> 8U) & 0x3ffU));
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    hip(hipMalloc(reinterpret_cast<void**>(&d), h.size() * sizeof(T)), "malloc");
    hip(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}
std::vector<float> download(const float* d, std::size_t n) {
    std::vector<float> h(n);
    hip(hipMemcpy(h.data(), d, n * sizeof(float), hipMemcpyDeviceToHost), "download");
    return h;
}

struct Case {
    std::uint32_t ggml_type, outputs, width, rows, x_stride, out_stride;
    bool bf16_input;
    const char* what;
};

// Weight bytes for [outputs][width] of the given type, random.
std::vector<std::uint8_t> weights(const Case& c, std::uint32_t& s) {
    std::vector<std::uint8_t> w;
    const std::size_t n = std::size_t{c.outputs} * c.width;
    if (c.ggml_type == 0U) {
        w.resize(n * 4U);
        for (std::size_t i = 0; i < n; ++i) {
            const float f = random_float(s);
            std::memcpy(&w[i * 4U], &f, 4);
        }
    } else if (c.ggml_type == 1U || c.ggml_type == 30U) {
        w.resize(n * 2U);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint16_t b = c.ggml_type == 1U ? f16_bits(s) : bf16_bits(random_float(s));
            std::memcpy(&w[i * 2U], &b, 2);
        }
    } else {  // Q8_0: {f16 d, int8 q[32]} per 32 values
        const std::size_t blocks = n / 32U;
        w.resize(blocks * 34U);
        for (std::size_t b = 0; b < blocks; ++b) {
            const std::uint16_t d = f16_bits(s);
            std::memcpy(&w[b * 34U], &d, 2);
            for (std::size_t q = 0; q < 32U; ++q) w[b * 34U + 2U + q] = static_cast<std::uint8_t>(lcg(s) >> 24U);
        }
    }
    return w;
}

void identity(const Case& c, bool time_it) {
    std::uint32_t s = 0x9e3779b9U ^ c.outputs ^ (c.rows << 12U);
    const auto w = weights(c, s);
    std::vector<float> xf(std::size_t{c.rows} * c.x_stride);
    for (auto& f : xf) f = random_float(s);
    std::vector<std::uint16_t> xb(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) xb[i] = bf16_bits(xf[i]);
    auto* dw = upload(w);
    void* dx = c.bf16_input ? static_cast<void*>(upload(xb)) : static_cast<void*>(upload(xf));
    const std::size_t on = std::size_t{c.rows} * c.out_stride;
    const std::vector<float> sentinel(on, -12345.0F);
    auto* da = upload(sentinel);
    auto* db = upload(sentinel);
    strix::HipDenseRowsView v;
    v.x = dx;
    v.input = c.bf16_input ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;
    v.weights = dw;
    v.ggml_type = c.ggml_type;
    v.width = c.width;
    v.outputs = c.outputs;
    v.rows = c.rows;
    v.x_stride = c.x_stride;
    v.out_stride = c.out_stride;
    v.out = da;
    v.row_loop = true;
    strix::launch_hip_dense_rows(v, nullptr);
    v.out = db;
    v.row_loop = false;
    strix::launch_hip_dense_rows(v, nullptr);
    hip(hipDeviceSynchronize(), "sync");
    const auto a = download(da, on), b = download(db, on);
    bool same = std::memcmp(a.data(), b.data(), on * sizeof(float)) == 0;
    bool written = true;
    for (std::uint32_t r = 0; r < c.rows; ++r)
        for (std::uint32_t o = 0; o < c.out_stride; ++o) {
            const bool stored = std::memcmp(&b[std::size_t{r} * c.out_stride + o], &sentinel[0], 4) != 0;
            if (stored != (o < c.outputs)) written = false;
        }
    check(same && written, c.what);
    if (time_it) {
        hipEvent_t e0, e1;
        hip(hipEventCreate(&e0), "event");
        hip(hipEventCreate(&e1), "event");
        float ms[2][5] = {};
        for (int rep = 0; rep < 5; ++rep)
            for (int form = 0; form < 2; ++form) {
                v.row_loop = form == 0;
                hip(hipEventRecord(e0, nullptr), "record");
                strix::launch_hip_dense_rows(v, nullptr);
                hip(hipEventRecord(e1, nullptr), "record");
                hip(hipEventSynchronize(e1), "sync");
                hip(hipEventElapsedTime(&ms[form][rep], e0, e1), "elapsed");
            }
        std::sort(ms[0], ms[0] + 5);
        std::sort(ms[1], ms[1] + 5);
        std::printf("TIMING router F32 %u x %u rows %u: per-row loop %.3f ms, rows tile %.3f ms (median of 5)\n",
                    c.outputs, c.width, c.rows, static_cast<double>(ms[0][2]), static_cast<double>(ms[1][2]));
        hip(hipEventDestroy(e0), "event");
        hip(hipEventDestroy(e1), "event");
    }
    hip(hipFree(dw), "free");
    hip(hipFree(dx), "free");
    hip(hipFree(da), "free");
    hip(hipFree(db), "free");
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("no HIP device: skipped\n");
        return 77;
    }
    try {
        identity({0U, 512U, 3072U, 2048U, 3072U, 512U, false,
                  "rows tile == per-row, bitwise: F32 512 x 3072 fp32 rows 2048 (router)"}, true);
        identity({0U, 16U, 3072U, 37U, 3072U, 16U, false,
                  "rows tile == per-row, bitwise: F32 16 x 3072 fp32 rows 37 (indexer proj)"}, false);
        identity({30U, 1003U, 3000U, 13U, 3072U, 1010U, true,
                  "rows tile == per-row, bitwise: BF16 1003 x 3000 bf16 rows 13, strides"}, false);
        identity({8U, 300U, 2048U, 9U, 2048U, 300U, true,
                  "rows tile == per-row, bitwise: Q8_0 300 x 2048 bf16 rows 9"}, false);
        identity({1U, 64U, 768U, 2U, 768U, 64U, false,
                  "rows tile == per-row, bitwise: F16 64 x 768 fp32 rows 2"}, false);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
