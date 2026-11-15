// CPU backend scaffold: the reference and the selected table against the existing host references.
//  - quantize_a8 and gemv_q8_0_a8 against strix::ds41::quantize_a8 / product_q8_0 (bitwise)
//  - rmsnorm through the shared strix::ops::norm::Norm<> interface against ops::norm::reference (bitwise)
#include "cpu/core/backend.hpp"
#include "ops/norm/reference.hpp"
#include "strix/ds41_formats.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

namespace {
unsigned failures = 0, checks = 0;
void expect(bool ok, const char* what, std::size_t i) {
    ++checks;
    if (!ok && failures++ < 10) std::printf("FAIL %s at %zu\n", what, i);
}
bool same(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

struct HostNormTraits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    using Stream = void*;
    static constexpr float epsilon = 1e-6F;
    static constexpr auto reduction_order = strix::ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    static constexpr unsigned lanes(unsigned n) { return strix::cpu::NormTraits::lanes(n); }
    static float inv_sqrt(float x) { return 1.0F / std::sqrt(x); }
    static float load(const float* w, int, std::size_t i) { return w[i]; }
};

std::uint16_t float_to_half_bits(float f) {  // fixture helper: normal finite scales only
    const std::uint32_t b = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (b >> 16U) & 0x8000U;
    const int e = static_cast<int>((b >> 23U) & 0xffU) - 127 + 15;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10U) | ((b >> 13U) & 0x3ffU));
}

void test_q8_0(const strix::cpu::Kernels& k, std::mt19937& rng, std::uint32_t width, std::uint32_t count,
               std::uint32_t rows) {
    namespace d = strix::ds41;
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> code(-128, 127);
    std::uniform_real_distribution<float> sc(1e-3F, 2e-2F);
    const std::uint32_t row_bytes = width / 32U * 34U;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(count) * row_bytes);
    for (std::size_t b = 0; b < w.size() / 34U; ++b) {
        const std::uint16_t h = float_to_half_bits(sc(rng) * (b % 7U == 0 ? -1.0F : 1.0F));
        w[b * 34U] = static_cast<std::uint8_t>(h & 0xffU);
        w[b * 34U + 1] = static_cast<std::uint8_t>(h >> 8U);
        for (std::size_t j = 0; j < 32; ++j) w[b * 34U + 2 + j] = static_cast<std::uint8_t>(code(rng));
    }
    std::vector<float> x(static_cast<std::size_t>(rows) * width);
    for (auto& v : x) v = nd(rng);
    for (std::size_t j = 0; j < 32 && rows > 1; ++j) x[width + j] = 0.0F;  // an all-zero group
    std::vector<std::int8_t> codes(x.size());
    std::vector<float> deltas(x.size() / 32U);
    std::vector<std::int32_t> sums(x.size() / 32U);
    std::vector<d::A8Activation> ref_x;
    for (std::uint32_t r = 0; r < rows; ++r) {
        k.quantize_a8(x.data() + static_cast<std::size_t>(r) * width, width, codes.data() + static_cast<std::size_t>(r) * width,
                      deltas.data() + static_cast<std::size_t>(r) * width / 32U, sums.data() + static_cast<std::size_t>(r) * width / 32U);
        ref_x.push_back(d::quantize_a8(std::span<const float>(x.data() + static_cast<std::size_t>(r) * width, width)));
        for (std::size_t i = 0; i < width; ++i) expect(codes[r * width + i] == ref_x[r].codes[i], "a8 code", i);
        for (std::size_t g = 0; g < width / 32U; ++g) {
            expect(same(deltas[r * width / 32U + g], ref_x[r].deltas[g]), "a8 delta", g);
            expect(sums[r * width / 32U + g] == ref_x[r].sums[g], "a8 sum", g);
        }
    }
    std::vector<float> y(static_cast<std::size_t>(rows) * count, 0.0F);
    const strix::cpu::Q8a8View v{w.data(), codes.data(), deltas.data(), y.data(), row_bytes, width, count, rows, width, count, 0};
    k.gemv_q8_0_a8(v, strix::cpu::Stream{});
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < count; ++o) {
            const float ref = d::product_q8_0(std::span<const std::uint8_t>(w.data() + static_cast<std::size_t>(o) * row_bytes, row_bytes),
                                              ref_x[r], width);
            expect(same(y[static_cast<std::size_t>(r) * count + o], ref), "q8_0 gemv", o);
        }
    // accumulate: out += product, one rounding after the product
    std::vector<float> y2 = y;
    const strix::cpu::Q8a8View va{w.data(), codes.data(), deltas.data(), y2.data(), row_bytes, width, count, rows, width, count, 1};
    k.gemv_q8_0_a8(va, strix::cpu::Stream{});
    for (std::size_t i = 0; i < y.size(); ++i) expect(same(y2[i], y[i] + y[i]), "q8_0 accumulate", i);
}

void test_norm(std::mt19937& rng, unsigned n, unsigned rows) {
    std::normal_distribution<float> nd(0.0F, 3.0F);
    std::vector<float> x(static_cast<std::size_t>(n) * rows), w(n), y(x.size()), ref(x.size());
    for (auto& v : x) v = nd(rng);
    for (auto& v : w) v = 1.0F + 0.1F * nd(rng);
    const strix::cpu::Stream s{};
    strix::cpu::Norm::run(x.data(), w.data(), 0, n, 1e-6F, y.data(), rows, &s);
    strix::ops::norm::reference<HostNormTraits>(x.data(), w.data(), 0, n, 1e-6F, ref.data(), rows);
    for (std::size_t i = 0; i < y.size(); ++i) expect(same(y[i], ref[i]), "rmsnorm", i);
    strix::cpu::Norm::run(x.data(), nullptr, 0, n, 1e-6F, y.data(), rows, &s);
    strix::ops::norm::reference<HostNormTraits>(x.data(), nullptr, 0, n, 1e-6F, ref.data(), rows);
    for (std::size_t i = 0; i < y.size(); ++i) expect(same(y[i], ref[i]), "rmsnorm unscaled", i);
}
}  // namespace

int main() {
    std::mt19937 rng(1112U);
    const auto host = strix::cpu::caps();
    const auto& selected = strix::cpu::select(host);
    std::printf("cpu backend: selected table %s (avx2=%d avx512f=%d vnni=%d amx=%d neon=%d sve=%d)\n",
                strix::cpu::isa_name(selected.isa), host.avx2, host.avx512f, host.avx512vnni, host.amx_int8,
                host.neon, host.sve);
    for (const auto* k : {&strix::cpu::reference_kernels(), &selected}) {
        test_q8_0(*k, rng, 32, 5, 1);
        test_q8_0(*k, rng, 2048, 64, 3);
        test_q8_0(*k, rng, 7168 + 32, 17, 2);  // groups not a multiple of the lane count
    }
    test_norm(rng, 96, 3);
    test_norm(rng, 4096, 2);
    test_norm(rng, 7168, 1);
    std::printf("%u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
