// Device fixture of the shared A4W4 class (core/iu4a4_dense.hpp) against the contract dense-iu4a4-rot-v1
// (core/iu4a4_rot.hpp), bit for bit: the weight requantisation (codes + steps), the activation quantiser (codes, group
// scales, row base; pad rows; a zero row; NaN / Inf rows) and every GEMM output (NaN rows: NaN on both sides).
// Shapes cover partial CTAs in m and n, an odd count of 128 blocks and a single tile; outputs at a row pitch, through
// the default store (incumbent epilogue and grid) and through the opt-in StagedRawStore (staged epilogue, column
// blocks first).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/iu4a4_dense.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm>
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
struct Mul {  // map2 functor of the fixture (an engine passes its gated unit)
    __device__ __forceinline__ static float apply(float a, float b) { return a * b; }
};

int run_case(unsigned rows, unsigned in, unsigned n, unsigned seed, bool map2 = false, unsigned pitch = 0U, bool staged = false) {
    const unsigned ys = pitch ? pitch : n;  // output row pitch (floats); columns n .. ys - 1 must stay untouched
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
    // map2: the device gets (x, x2) and quantises x * x2; the reference quantises the stored FP32 product.
    std::vector<float> x2(map2 ? x.size() : 0U), xq = x;
    for (std::size_t i = 0; i < x2.size(); ++i) x2[i] = 0.5f + nd(g) * 0.25f, xq[i] = x[i] * x2[i];

    std::uint8_t *dpack = nullptr, *dw = nullptr, *dcodes = nullptr;
    std::uint16_t* dsc = nullptr;
    float *dstep = nullptr, *dx = nullptr, *dx2 = nullptr, *dbase = nullptr, *dy = nullptr;
    std::uint32_t* du = nullptr;
    CHECK(hipMalloc(&dpack, pack.size()));
    CHECK(hipMalloc(&dsc, sc.size() * 2U));
    CHECK(hipMalloc(&dw, ad::weight_code_bytes(in, n)));
    CHECK(hipMalloc(&dstep, ad::weight_step_bytes(in, n)));
    CHECK(hipMalloc(&dx, x.size() * 4U));
    CHECK(hipMalloc(&dx2, std::max<std::size_t>(x2.size(), 1U) * 4U));
    CHECK(hipMalloc(&dcodes, ad::act_code_bytes(rows, in)));
    CHECK(hipMalloc(&du, ad::act_u_bytes(rows, in)));
    CHECK(hipMalloc(&dbase, ad::act_base_bytes(rows)));
    CHECK(hipMalloc(&dy, std::size_t(rows) * ys * 4U));
    CHECK(hipMemcpy(dpack, pack.data(), pack.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dsc, sc.data(), sc.size() * 2U, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice));
    if (map2) CHECK(hipMemcpy(dx2, x2.data(), x2.size() * 4U, hipMemcpyHostToDevice));
    CHECK(hipMemset(dy, 0xff, std::size_t(rows) * ys * 4U));
    const ad::Act act{dcodes, du, dbase, rows, in};
    const ad::Weights wt{dw, dstep, in, n};
    ad::enqueue_requantise(dpack, dsc, in, n, dw, dstep, nullptr);
    if (map2) ad::enqueue_quantise_map2<Mul>(dx, dx2, in, act, nullptr);
    else ad::enqueue_quantise(dx, in, act, nullptr);
    if (staged) ad::enqueue_gemm(wt, act, ad::OutCols{dy, ys, n / 16U}, nullptr, ad::StagedRawStore{});
    else ad::enqueue_gemm(wt, act, ad::OutCols{dy, ys, n / 16U}, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<std::uint8_t> gw(ad::weight_code_bytes(in, n)), gcodes(ad::act_code_bytes(rows, in));
    std::vector<float> gstep(ad::weight_step_bytes(in, n) / 4U), gbase(mp), gy(std::size_t(rows) * ys);
    std::vector<std::uint32_t> gu(ad::act_u_bytes(rows, in) / 4U);
    CHECK(hipMemcpy(gw.data(), dw, gw.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gstep.data(), dstep, gstep.size() * 4U, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gcodes.data(), dcodes, gcodes.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gu.data(), du, gu.size() * 4U, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gbase.data(), dbase, gbase.size() * 4U, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gy.data(), dy, gy.size() * 4U, hipMemcpyDeviceToHost));
    for (void* p : {static_cast<void*>(dpack), static_cast<void*>(dsc), static_cast<void*>(dw), static_cast<void*>(dstep), static_cast<void*>(dx), static_cast<void*>(dx2),
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
        std::copy(xq.begin() + std::ptrdiff_t(r) * in, xq.begin() + std::ptrdiff_t(r + 1U) * in, row.begin());
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
            const float dev = gy[std::size_t(r) * ys + c];
            if (!same(dev, ref)) {
                if (bad_y < 4U) std::printf("  y[%u][%u] dev %.9g (0x%08x) ref %.9g (0x%08x)\n", r, c, double(dev), bits(dev), double(ref), bits(ref));
                ++bad_y;
            }
            nan_y += std::isnan(ref) ? 1U : 0U;
        }
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = n; c < ys; ++c) bad_y += bits(gy[std::size_t(r) * ys + c]) != 0xffffffffU;  // pitch padding
    const bool ok = !(bad_w | bad_s | bad_a | bad_u | bad_b | bad_y);
    std::printf("%s%s%s pitch %u rows %u in %u n %u: wcodes %u steps %u acodes %u u %u base %u y %u differ (NaN outputs %u)\n", ok ? "PASS" : "FAIL", map2 ? " map2" : "", staged ? " staged" : "", ys, rows, in, n,
                bad_w, bad_s, bad_a, bad_u, bad_b, bad_y, nan_y);
    return ok ? 0 : 1;
}

// Weight rule trait (contract v2): the device image built with WeightRule::mse_clip is the host rule bit for bit
// (codes and steps), and differs from the v1 image (the trait is live, not ignored).
int run_rule_case(unsigned in, unsigned n, unsigned seed) {
    std::mt19937 g(seed);
    const unsigned blocks = in / rot::kBlock;
    std::vector<std::uint8_t> pack(std::size_t(n) * in / 2U);
    for (auto& e : pack) e = static_cast<std::uint8_t>(g());
    std::vector<std::uint16_t> sc(std::size_t(n) * (in / 64U));
    for (auto& e : sc) e = static_cast<std::uint16_t>(((5U + g() % 7U) << 10U) | (g() & 1023U));
    std::uint8_t *dpack = nullptr, *dw = nullptr, *dw1 = nullptr;
    std::uint16_t* dsc = nullptr;
    float *dstep = nullptr, *dstep1 = nullptr;
    CHECK(hipMalloc(&dpack, pack.size()));
    CHECK(hipMalloc(&dsc, sc.size() * 2U));
    CHECK(hipMalloc(&dw, ad::weight_code_bytes(in, n)));
    CHECK(hipMalloc(&dw1, ad::weight_code_bytes(in, n)));
    CHECK(hipMalloc(&dstep, ad::weight_step_bytes(in, n)));
    CHECK(hipMalloc(&dstep1, ad::weight_step_bytes(in, n)));
    CHECK(hipMemcpy(dpack, pack.data(), pack.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dsc, sc.data(), sc.size() * 2U, hipMemcpyHostToDevice));
    CHECK(hipMemset(dw, 0, ad::weight_code_bytes(in, n)));
    CHECK(hipMemset(dw1, 0, ad::weight_code_bytes(in, n)));
    ad::enqueue_requantise(dpack, dsc, in, n, dw, dstep, nullptr, rot::WeightRule::mse_clip);
    ad::enqueue_requantise(dpack, dsc, in, n, dw1, dstep1, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<std::uint8_t> gw(ad::weight_code_bytes(in, n)), gw1(gw.size());
    std::vector<float> gstep(ad::weight_step_bytes(in, n) / 4U), gstep1(gstep.size());
    CHECK(hipMemcpy(gw.data(), dw, gw.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gw1.data(), dw1, gw1.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gstep.data(), dstep, gstep.size() * 4U, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gstep1.data(), dstep1, gstep1.size() * 4U, hipMemcpyDeviceToHost));
    for (void* p : {static_cast<void*>(dpack), static_cast<void*>(dsc), static_cast<void*>(dw), static_cast<void*>(dw1),
                    static_cast<void*>(dstep), static_cast<void*>(dstep1)})
        CHECK(hipFree(p));
    std::vector<std::uint8_t> hw(gw.size(), 0);
    std::vector<float> hstep(std::size_t(n) * blocks);
    unsigned bad_s = 0, bad_w = 0, clipped = 0;
    for (unsigned b = 0; b < blocks; ++b)
        for (unsigned c = 0; c < n; ++c) {
            rot::requantise_block(pack.data(), sc.data(), in, n, c, b, hw.data(), hstep.data(), rot::WeightRule::mse_clip);
            bad_s += bits(gstep[ad::step_index(b, c, n)]) != bits(hstep[std::size_t(c) * blocks + b]);
            clipped += gstep[ad::step_index(b, c, n)] < gstep1[ad::step_index(b, c, n)];
        }
    for (std::size_t i = 0; i < hw.size(); ++i) bad_w += hw[i] != gw[i];
    const bool ok = bad_s == 0 && bad_w == 0 && clipped > 0 && gw != gw1;
    std::printf("%s weight rule %s in %u n %u: wcodes %u steps %u differ from the host rule; %u of %u blocks clipped\n",
                ok ? "PASS" : "FAIL", rot::identity(rot::WeightRule::mse_clip), in, n, bad_w, bad_s, clipped, n * blocks);
    return ok ? 0 : 1;
}
}  // namespace

int main() {
    int fails = 0;
    fails += run_rule_case(1280, 272, 11U);  // weight rule v2: device image == host rule
    fails += run_rule_case(384, 16, 12U);
    fails += run_case(37, 512, 96, 1U);     // partial m tile, 6 W tiles (partial CTA in n)
    fails += run_case(130, 1280, 272, 2U);  // 2 m CTAs (second partial), 17 W tiles
    fails += run_case(16, 384, 16, 3U);     // odd block count, one tile each way
    fails += run_case(300, 2048, 400, 4U);  // 3 m CTAs, 25 W tiles
    fails += run_case(40, 17408, 32, 5U, true);  // map2 (gated-unit form), the longest CTA-quantiser row of the bound shapes
    fails += run_case(20, 24704, 32, 6U);        // past kCtaInputs: the wave-per-row quantiser
    fails += run_case(70, 640, 96, 7U, false, 1024U);  // incumbent epilogue at a 4 KiB pitch
    fails += run_case(37, 256, 48, 8U, false, 51U);    // incumbent epilogue at an odd pitch
    fails += run_case(70, 640, 96, 7U, false, 1024U, true);  // staged epilogue: 16 B aligned rows at a 4 KiB pitch
    fails += run_case(37, 256, 48, 8U, false, 51U, true);    // staged epilogue: odd pitch (unaligned chunks, scalar stores)
    fails += run_case(300, 2048, 400, 4U, false, 0U, true);  // staged epilogue: 3 m CTAs, 25 W tiles (partial n CTA)
    std::printf("%s\n", fails ? "iu4a4_dense tests FAILED" : "iu4a4_dense tests PASS");
    return fails ? 1 : 0;
}
