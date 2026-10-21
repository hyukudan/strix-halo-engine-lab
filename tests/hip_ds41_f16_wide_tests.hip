// The wide-load F16 rows kernel against the plain one, bit for bit: the decode shapes of the deepseek41 F16 sites
// (Engram kv, compressor kv and gate, indexer q_b and weights) at 1..8 rows, plus partial last blocks and an offset
// first row; then the time of both kernels on the Engram shape.
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

struct Case {
    std::uint32_t width, first, count, rows;
};

std::vector<float> run(const std::uint16_t* w, const float* x, float* out, const Case& c, bool wide) {
    strix::ds41::set_f16_rows_wide(wide);
    check(hipMemset(out, 0xff, std::size_t{c.rows} * c.count * 4U), "clear");
    strix::ds41::F16RowsView v;
    v.weight = w;
    v.width = c.width;
    v.first = c.first;
    v.count = c.count;
    v.x = x;
    v.rows = c.rows;
    v.x_stride = c.width;
    v.out_stride = c.count;
    v.out = out;
    strix::ds41::enqueue_f16_rows(v, nullptr);
    check(hipDeviceSynchronize(), "run");
    std::vector<float> h(std::size_t{c.rows} * c.count);
    check(hipMemcpy(h.data(), out, h.size() * 4U, hipMemcpyDeviceToHost), "download");
    return h;
}

}  // namespace

int main() try {
    const std::size_t max_w = std::size_t{6144} * 25600U + 4096U * 1536U;
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> wh(max_w);
    for (auto& v : wh) {  // finite F16 values of both signs, including subnormals
        const float f = normal(rng) * 0.05F;
        _Float16 h = static_cast<_Float16>(f);
        std::memcpy(&v, &h, 2);
    }
    std::vector<float> xh(std::size_t{8} * 6144U);
    for (auto& v : xh) v = normal(rng);
    std::uint16_t* w = nullptr;
    float *x = nullptr, *out = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&w), wh.size() * 2U), "w");
    check(hipMalloc(reinterpret_cast<void**>(&x), xh.size() * 4U), "x");
    check(hipMalloc(reinterpret_cast<void**>(&out), std::size_t{8} * 25600U * 4U), "out");
    check(hipMemcpy(w, wh.data(), wh.size() * 2U, hipMemcpyHostToDevice), "w");
    check(hipMemcpy(x, xh.data(), xh.size() * 4U, hipMemcpyHostToDevice), "x");
    const Case shapes[] = {{6144, 0, 25600, 0}, {5120, 0, 512, 0}, {1536, 0, 4096, 0}, {5120, 0, 32, 0},
                           {5120, 3, 2051, 0},  {256, 5, 37, 0},    {1536, 1, 4095, 0}};
    int failures = 0;
    for (const auto& s : shapes)
        for (std::uint32_t rows = 1; rows <= 8; ++rows) {
            Case c = s;
            c.rows = rows;
            const auto a = run(w, x, out, c, false), b = run(w, x, out, c, true);
            const bool same = std::memcmp(a.data(), b.data(), a.size() * 4U) == 0;
            if (!same) ++failures;
            if (!same || rows == 1U || rows == 8U)
                std::printf("width %5u first %u count %5u rows %u: %s\n", c.width, c.first, c.count, rows, same ? "bit-exact" : "DIFFER");
        }
    for (const std::uint32_t rows : {1U, 4U, 8U}) {
        const Case c{6144, 0, 25600, rows};
        double t[2];
        for (int wide = 0; wide < 2; ++wide) {
            (void)run(w, x, out, c, wide != 0);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 20; ++i) (void)run(w, x, out, c, wide != 0);
            t[wide] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20.0 * 1e3;
        }
        std::printf("Engram shape rows %u: plain %.3f ms, wide %.3f ms (with a sync and download each)\n", rows, t[0], t[1]);
    }
    std::printf("%s\n", failures == 0 ? "ALL BIT-EXACT" : "FAILURES");
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
