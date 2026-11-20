// core/q8_0_transpose.hpp: the transposed Q8_0 image of an MLA K_b weight (expanded-MHA prefill).
// Invariants: (1) exact transposition when each output block's amax is 127 x a shared power-of-two scale; (2) for any
// source, every output value is within half an output step (+ the f16 rounding of the scale) of the source value at the
// transposed index; (3) zero blocks give scale 0 and codes 0; (4) bad shapes are refused.
#include "core/q8_0_transpose.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

namespace q8 = strix::core::q8_0;
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

// batch x [rows][cols] Q8_0 with the given per-block scale function and codes.
template <class Scale, class Code>
std::vector<std::uint8_t> make(std::size_t batch, std::size_t rows, std::size_t cols, Scale scale, Code code) {
    std::vector<std::uint8_t> m(batch * rows * q8::row_bytes(cols));
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t g = 0; g < cols / q8::kBlock; ++g) {
                std::uint8_t* blk = m.data() + (b * rows + r) * q8::row_bytes(cols) + g * q8::kBlockBytes;
                const std::uint16_t h = strix::core::onfly::f32_to_f16_bits(scale(b, r, g));
                blk[0] = static_cast<std::uint8_t>(h & 0xffU);
                blk[1] = static_cast<std::uint8_t>(h >> 8U);
                for (std::size_t j = 0; j < q8::kBlock; ++j)
                    blk[2U + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(b, r, g * q8::kBlock + j)));
            }
    return m;
}

void exact_case() {
    const std::size_t B = 3, R = 64, C = 96;  // the K_b shape class: [head][kv_lora][qk_nope], smaller
    // one power-of-two scale per batch; code 127 on the diagonal of every 32 x 32 tile so every output block has amax 127 d
    const auto scale = [](std::size_t b, std::size_t, std::size_t) { return std::ldexp(1.0F, -7 - static_cast<int>(b)); };
    const auto code = [](std::size_t b, std::size_t r, std::size_t c) -> int {
        if (r % 32U == c % 32U) return (r + c) % 2U ? -127 : 127;
        return static_cast<int>((r * 7U + c * 13U + b * 5U) % 255U) - 127;
    };
    const auto src = make(B, R, C, scale, code);
    const auto t = q8::transpose(src.data(), B, R, C);
    bool same = t.size() == src.size();
    for (std::size_t b = 0; b < B && same; ++b)
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c)
                same &= q8::value(t.data() + b * C * q8::row_bytes(R), R, c, r) == q8::value(src.data() + b * R * q8::row_bytes(C), C, r, c);
    expect(same, "exact transposition (power-of-two shared scale, amax 127)");
}

void bound_case() {
    const std::size_t B = 2, R = 512, C = 192;  // Hy4 K_b per head: [kv_lora 512][qk_nope 192]
    std::mt19937 g(20261120U);
    std::uniform_real_distribution<float> sd(1e-5F, 4e-3F);
    std::uniform_int_distribution<int> cd(-127, 127);
    std::vector<float> scales(B * R * (C / q8::kBlock));
    for (auto& s : scales) s = sd(g);
    std::vector<int> codes(B * R * C);
    for (auto& c : codes) c = cd(g);
    const auto src = make(B, R, C, [&](std::size_t b, std::size_t r, std::size_t k) { return scales[(b * R + r) * (C / q8::kBlock) + k]; },
                          [&](std::size_t b, std::size_t r, std::size_t c) { return codes[(b * R + r) * C + c]; });
    const auto t = q8::transpose(src.data(), B, R, C);
    double worst = 0.0;
    bool ok = true;
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r0 = 0; r0 < R; r0 += q8::kBlock) {
                const std::uint8_t* tb = t.data() + b * C * q8::row_bytes(R) + c * q8::row_bytes(R) + r0 / q8::kBlock * q8::kBlockBytes;
                const float d = strix::core::onfly::f16_bits_to_f32(static_cast<std::uint16_t>(tb[0] | (tb[1] << 8U)));
                for (std::size_t j = 0; j < q8::kBlock; ++j) {
                    const float v = q8::value(src.data() + b * R * q8::row_bytes(C), C, r0 + j, c);
                    const float w = q8::value(t.data() + b * C * q8::row_bytes(R), R, c, r0 + j);
                    // half a step of the FP32 scale, plus the f16 rounding of the scale times the code (<= 2^-11 relative)
                    const double bound = 0.5 * d * (1.0 + 0x1p-10) + 127.0 * std::fmax(d * 0x1p-11, 0x1p-25);
                    const double err = std::fabs(static_cast<double>(w) - v);
                    ok &= err <= bound;
                    if (d > 0.0F) worst = std::fmax(worst, err / d);
                }
            }
    expect(ok, "re-quantised values within half a step of the source");
    std::printf("bound case: worst |error| / step = %.4f\n", worst);
}

void zero_and_refuse() {
    const auto src = make(1, 32, 32, [](std::size_t, std::size_t, std::size_t) { return 0.5F; }, [](std::size_t, std::size_t, std::size_t) { return 0; });
    const auto t = q8::transpose(src.data(), 1, 32, 32);
    bool zero = true;
    for (const auto byte : t) zero &= byte == 0U;
    expect(zero, "zero block -> scale 0, codes 0");
    bool threw = false;
    try {
        (void)q8::transpose(src.data(), 1, 32, 48);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "cols not a multiple of 32 refused");
}

}  // namespace

int main() {
    exact_case();
    bound_case();
    zero_and_refuse();
    if (failures == 0) std::printf("q8_0_transpose tests: PASS\n");
    return failures == 0 ? 0 : 1;
}
