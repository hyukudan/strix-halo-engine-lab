// Probe (measurement only, not linked into any engine): the IU4 decode projection (core/dense_linear_iu4.hpp) split by
// output columns as a tensor-parallel rank computes it (run_cols), on the Qwen 27B decode shapes.
//   - bitwise: the column shares of an equal split and of a rate-weighted split (rates 1 : 1.12), joined in place, must
//     equal the full launch word for word;
//   - time: full launch vs each share, rotating over enough copies of the matrix (>= 160 MB) that the 32 MB MALL cannot
//     hold them, so the read rate is the DRAM rate a decode step sees.
// Usage: hip-dense-iu4-cols-probe [ROWS...]   (default 1 8 16). One line per (shape, rows); exit 1 on any bit mismatch.
#include "core/dense_linear_iu4.hpp"
#include "strix/pair/column_split.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

namespace {
namespace iu4 = strix::core::dense_iu4;
namespace tp = strix::pair::tp;

struct Shape { const char* name; unsigned n, k; };

template <class F> double time_us(F&& f, int iters) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) f(i);
    CK(hipEventRecord(a, nullptr));
    for (int i = 0; i < iters; ++i) f(i);
    CK(hipEventRecord(b, nullptr));
    CK(hipEventSynchronize(b));
    float ms = 0.0F;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return double(ms) * 1000.0 / iters;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<unsigned> rows_list;
    for (int i = 1; i < argc; ++i) rows_list.push_back(unsigned(std::atoi(argv[i])));
    if (rows_list.empty()) rows_list = {1U, 8U, 16U};
    const Shape shapes[] = {{"gdn_qkv", 10240, 5120}, {"gdn_z", 6144, 5120}, {"attn_q", 12288, 5120},
                            {"out_o", 5120, 6144},    {"gate_up", 17408, 5120}, {"down", 5120, 17408}};
    std::mt19937_64 rng(20261118);
    int failures = 0;
    for (const auto& sh : shapes) {
        const std::size_t tb = iu4::tensor_bytes(sh.n, sh.k), pb = iu4::packed_bytes(sh.n, sh.k);
        const unsigned copies = unsigned(std::max<std::size_t>(2U, (std::size_t(160) << 20U) / tb + 1U));
        std::vector<std::uint8_t> host(tb);
        for (std::size_t i = 0; i < pb; ++i) host[i] = std::uint8_t(rng());
        auto* sc = reinterpret_cast<_Float16*>(host.data() + pb);
        std::uniform_real_distribution<float> sd(1.0F / 512.0F, 1.0F / 32.0F);
        for (std::size_t i = 0; i < std::size_t(sh.n) * (sh.k / 64U); ++i) sc[i] = static_cast<_Float16>(sd(rng));
        std::uint8_t* w = nullptr;
        CK(hipMalloc(&w, tb * copies));
        for (unsigned c = 0; c < copies; ++c) CK(hipMemcpy(w + std::size_t(c) * tb, host.data(), tb, hipMemcpyHostToDevice));
        // equal split and a rate-weighted split, in 64-column units
        const double eq[2] = {1.0, 1.0}, uneq[2] = {1.0, 1.12};
        const auto se = tp::split_units(sh.n / 64U, eq), su = tp::split_units(sh.n / 64U, uneq);
        for (const unsigned rows : rows_list) {
            std::vector<float> xs(std::size_t(16) * sh.k);
            std::normal_distribution<float> xd(0.0F, 1.0F);
            for (auto& v : xs) v = xd(rng);
            float *x = nullptr, *full = nullptr, *split = nullptr;
            _Float16* xh = nullptr;
            CK(hipMalloc(&x, xs.size() * 4U));
            CK(hipMalloc(&xh, xs.size() * 2U));
            CK(hipMalloc(&full, std::size_t(16) * sh.n * 4U));
            CK(hipMalloc(&split, std::size_t(16) * sh.n * 4U));
            CK(hipMemcpy(x, xs.data(), xs.size() * 4U, hipMemcpyHostToDevice));
            iu4::prepare(x, xh, sh.k, 16U, nullptr);
            iu4::run(w, sh.k, sh.n, xh, full, rows, nullptr);
            std::vector<float> a(std::size_t(rows) * sh.n), b(a.size());
            CK(hipMemcpy(a.data(), full, a.size() * 4U, hipMemcpyDeviceToHost));
            bool same = true;
            for (const auto* s : {&se, &su}) {
                CK(hipMemset(split, 0xFF, std::size_t(16) * sh.n * 4U));
                for (const auto& r : *s) {
                    const auto c = tp::scaled(r, 64U);
                    iu4::run_cols(w, sh.k, sh.n, c.begin, c.size(), xh, split + c.begin, sh.n, rows, nullptr);
                }
                CK(hipMemcpy(b.data(), split, b.size() * 4U, hipMemcpyDeviceToHost));
                same = same && std::memcmp(a.data(), b.data(), a.size() * 4U) == 0;
            }
            if (!same) ++failures;
            const int iters = 40;
            const double t_full = time_us([&](int i) { iu4::run(w + std::size_t(unsigned(i) % copies) * tb, sh.k, sh.n, xh, full, rows, nullptr); }, iters);
            double t_half[2];
            for (int h = 0; h < 2; ++h) {
                const auto c = tp::scaled(se[std::size_t(h)], 64U);
                t_half[h] = time_us([&](int i) {
                    iu4::run_cols(w + std::size_t(unsigned(i) % copies) * tb, sh.k, sh.n, c.begin, c.size(), xh, split + c.begin, sh.n, rows, nullptr);
                }, iters);
            }
            const double gb_full = double(tb) / t_full / 1e3, gb_half = double(tb) / 2.0 / std::max(t_half[0], t_half[1]) / 1e3;
            std::printf("cols %-8s n %5u k %5u rows %2u full %8.1f us %6.1f GB/s | half0 %8.1f half1 %8.1f us %6.1f GB/s | "
                        "max(half)/full %.3f | bitwise %s (eq %u|%u, rate 1:1.12 %u|%u blocks)\n",
                        sh.name, sh.n, sh.k, rows, t_full, gb_full, t_half[0], t_half[1], gb_half,
                        std::max(t_half[0], t_half[1]) / t_full, same ? "OK" : "FAIL", se[0].size(), se[1].size(), su[0].size(), su[1].size());
            CK(hipFree(x));
            CK(hipFree(xh));
            CK(hipFree(full));
            CK(hipFree(split));
        }
        CK(hipFree(w));
    }
    std::printf("cols probe %s (%d shape/rows mismatches)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
