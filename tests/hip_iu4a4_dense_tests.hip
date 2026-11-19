// Device fixture of the shared A4W4 class (core/iu4a4_dense.hpp) against the contract dense-iu4a4-rot-v1
// (core/iu4a4_rot.hpp), bit for bit: the weight requantisation (codes + steps), the activation quantiser (codes, group
// scales, row base; pad rows; a zero row; NaN / Inf rows) and every GEMM output (NaN rows: NaN on both sides).
// Shapes cover partial CTAs in m and n, an odd count of 128 blocks and a single tile.
#include "core/iu4a4_dense.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace ad = strix::core::iu4a4_dense;
namespace rot = strix::core::iu4a4_rot;
#define CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

std::uint32_t bits(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}
bool same(float a, float b) { return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b); }

int run_case(unsigned rows, unsigned in, unsigned n, unsigned seed) {
    std::mt19937 g(seed);
    const unsigned mp = ad::padded_rows(rows), blocks = in / rot::kBlock;
    // Weights: random nibbles, FP16 scales 2^-10 .. 2^-4 with random mantissas.
    std::vector<std::uint8_t> pack(std::size_t(n) * in / 2U);
    for (auto& e : pack) e = static_cast<std::uint8_t>(g());
    std::vector<std::uint16_t> sc(std::size_t(n) * (in / 64U));
    for (auto& e : sc) e = static_cast<std::uint16_t>(((5U + g() % 7U) << 10U) | (g() & 1023U));
    // Activations: normal values, a few outlier channels, a zero row, NaN / Inf rows.
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(std::size_t(rows) * in);
    for (auto& e : x) e = nd(g);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned k = 7; k < in; k += 211U) x[std::size_t(r) * in + k] *= 40.0f;
    if (rows > 3U) std::fill(x.begin() + 3 * std::ptrdiff_t(in), x.begin() + 4 * std::ptrdiff_t(in), 0.0f);
    if (rows > 5U) x[5U * std::size_t(in) + in / 2U] = std::nanf("");
    if (rows > 7U) x[7U * std::size_t(in) + 3U] = INFINITY, x[7U * std::size_t(in) + in - 1U] = std::nanf("");

    std::uint8_t *dpack = nullptr, *dw = nullptr, *dcodes = nullptr;
    std::uint16_t* dsc = nullptr;
    float *dstep = nullptr, *dx = nullptr, *dbase = nullptr, *dy = nullptr;
    std::uint32_t* du = nullptr;
    CHECK(hipMalloc(&dpack, pack.size()));
    CHECK(hipMalloc(&dsc, sc.size() * 2U));
    CHECK(hipMalloc(&dw, ad::weight_code_bytes(in, n)));
    CHECK(hipMalloc(&dstep, ad::weight_step_bytes(in, n)));
    CHECK(hipMalloc(&dx, x.size() * 4U));
    CHECK(hipMalloc(&dcodes, ad::act_code_bytes(rows, in)));
    CHECK(hipMalloc(&du, ad::act_u_bytes(rows, in)));
    CHECK(hipMalloc(&dbase, ad::act_base_bytes(rows)));
    CHECK(hipMalloc(&dy, std::size_t(rows) * n * 4U));
    CHECK(hipMemcpy(dpack, pack.data(), pack.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dsc, sc.data(), sc.size() * 2U, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice));
    CHECK(hipMemset(dy, 0xff, std::size_t(rows) * n * 4U));
    const ad::Act act{dcodes, du, dbase, rows, in};
    const ad::Weights wt{dw, dstep, in, n};
    ad::enqueue_requantise(dpack, dsc, in, n, dw, dstep, nullptr);
    ad::enqueue_quantise(dx, in, act, nullptr);
    ad::enqueue_gemm(wt, act, ad::OutCols{dy, n, n / 16U}, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<std::uint8_t> gw(ad::weight_code_bytes(in, n)), gcodes(ad::act_code_bytes(rows, in));
    std::vector<float> gstep(ad::weight_step_bytes(in, n) / 4U), gbase(mp), gy(std::size_t(rows) * n);
    std::vector<std::uint32_t> gu(ad::act_u_bytes(rows, in) / 4U);
    CHECK(hipMemcpy(gw.data(), dw, gw.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gstep.data(), dstep, gstep.size() * 4U, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gcodes.data(), dcodes, gcodes.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gu.data(), du, gu.size() * 4U, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gbase.data(), dbase, gbase.size() * 4U, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gy.data(), dy, gy.size() * 4U, hipMemcpyDeviceToHost));
    for (void* p : {static_cast<void*>(dpack), static_cast<void*>(dsc), static_cast<void*>(dw), static_cast<void*>(dstep), static_cast<void*>(dx),
                    static_cast<void*>(dcodes), static_cast<void*>(du), static_cast<void*>(dbase), static_cast<void*>(dy)})
        CHECK(hipFree(p));

    unsigned bad_w = 0, bad_s = 0, bad_a = 0, bad_u = 0, bad_b = 0, bad_y = 0, nan_y = 0;
    // Reference weights.
    std::vector<std::uint8_t> hw(gw.size(), 0);
    std::vector<float> hstep(std::size_t(n) * blocks);
    for (unsigned b = 0; b < blocks; ++b)
        for (unsigned c = 0; c < n; ++c) {
            rot::requantise_block(pack.data(), sc.data(), in, n, c, b, hw.data(), hstep.data());
            bad_s += bits(gstep[ad::step_index(b, c, n)]) != bits(hstep[std::size_t(c) * blocks + b]);
        }
    for (std::size_t i = 0; i < hw.size(); ++i) bad_w += hw[i] != gw[i];
    // Reference activations (pad rows: codes 0, u 1, base 0).
    std::vector<std::int8_t> hc(std::size_t(mp) * in, 0);
    std::vector<std::uint8_t> hu(std::size_t(mp) * (in / rot::kGroupA), 1);
    std::vector<float> hb(mp, 0.0f), row(in);
    for (unsigned r = 0; r < rows; ++r) {
        std::copy(x.begin() + std::ptrdiff_t(r) * in, x.begin() + std::ptrdiff_t(r + 1U) * in, row.begin());
        hb[r] = rot::quantise_row(row.data(), in, hc.data() + std::size_t(r) * in, hu.data() + std::size_t(r) * (in / rot::kGroupA));
    }
    for (unsigned r = 0; r < mp; ++r) {
        bad_b += !same(gbase[r], hb[r]);
        for (unsigned k = 0; k < in; ++k) bad_a += rot::pack_code(gcodes.data(), k, r, mp) != hc[std::size_t(r) * in + k];
        for (unsigned b = 0; b < blocks; ++b)
            for (unsigned q = 0; q < rot::kGroupsPerBlock; ++q)
                bad_u += ((gu[std::size_t(b) * mp + r] >> (8U * q)) & 255U) != hu[std::size_t(r) * (in / rot::kGroupA) + b * 4U + q];
    }
    // Reference outputs.
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < n; ++c) {
            const float ref = rot::dot(hc.data() + std::size_t(r) * in, hu.data() + std::size_t(r) * (in / rot::kGroupA), hb[r], hw.data(), hstep.data(),
                                       in, n, c);
            const float dev = gy[std::size_t(r) * n + c];
            if (!same(dev, ref)) {
                if (bad_y < 4U) std::printf("  y[%u][%u] dev %.9g (0x%08x) ref %.9g (0x%08x)\n", r, c, double(dev), bits(dev), double(ref), bits(ref));
                ++bad_y;
            }
            nan_y += std::isnan(ref) ? 1U : 0U;
        }
    const bool ok = !(bad_w | bad_s | bad_a | bad_u | bad_b | bad_y);
    std::printf("%s rows %u in %u n %u: wcodes %u steps %u acodes %u u %u base %u y %u differ (NaN outputs %u)\n", ok ? "PASS" : "FAIL", rows, in, n,
                bad_w, bad_s, bad_a, bad_u, bad_b, bad_y, nan_y);
    return ok ? 0 : 1;
}
}  // namespace

int main() {
    int fails = 0;
    fails += run_case(37, 512, 96, 1U);     // partial m tile, 6 W tiles (partial CTA in n)
    fails += run_case(130, 1280, 272, 2U);  // 2 m CTAs (second partial), 17 W tiles
    fails += run_case(16, 384, 16, 3U);     // odd block count, one tile each way
    fails += run_case(300, 2048, 400, 4U);  // 3 m CTAs, 25 W tiles
    std::printf("%s\n", fails ? "iu4a4_dense tests FAILED" : "iu4a4_dense tests PASS");
    return fails ? 1 : 0;
}
