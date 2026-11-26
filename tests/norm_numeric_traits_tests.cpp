// Shared norm numeric traits (src/ops/norm/norm_numeric_traits.hpp): host checks of the invariants a binding relies on.
//  1. ReductionOrder::wave_chunk_serial with SquareForm::mul_add equals, bit for bit, the two private wave-chunk sums it
//     replaces: chunk sums started at +0 and added serially (shfl_xor butterfly), and the one-wave sum with the shfl_down
//     tree read at lane 0.
//  2. The full row (factor 1 / CR sqrt, plain output) equals the private chunked kernel's expression.
//  3. SquareForm changes bits (fma vs mul_add), so the trait is load-bearing.
//  4. strix::numeric::rsqrt_correctly_rounded uses the same result through the Gemma4 adapter and agrees with the double oracle away from midpoint ambiguity; the
//     non-finite guard of inv_sqrt_correctly_rounded_finite.
#include "../src/ops/norm/reference.hpp"
#include "../src/gemma4/contracts/g4_identity.hpp"  // pow32_rsqrt: Gemma4 adapter
#include "strix/ds41_moe.hpp"                       // declared_sqrt (host: correctly rounded sqrtf)

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
namespace nm = strix::ops::norm;

unsigned checks = 0;
void require(bool condition, const char* what) {
    ++checks;
    if (!condition) throw std::runtime_error(what);
}
std::uint32_t bits(float x) {
    std::uint32_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}
float from_bits(std::uint32_t b) {
    float x;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}
bool same(float a, float b) { return bits(a) == bits(b); }

template <unsigned Chunk, nm::SquareForm Q, bool CorrectlyRounded>
struct HostChunkTraits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    using Stream = void*;
    static constexpr float epsilon = 1e-6F;
    static constexpr auto reduction_order = nm::ReductionOrder::wave_chunk_serial;
    static constexpr unsigned wave_chunk = Chunk;
    static constexpr auto square_form = Q;
    static constexpr unsigned lanes(unsigned) { return 32U; }
    static float inv_sqrt(float u) {
        if constexpr (CorrectlyRounded) return nm::inv_sqrt_correctly_rounded_finite(u);
        else return 1.0F / strix::ds41::declared_sqrt(u);
    }
    static float load(const float* w, int, unsigned i) { return w[i]; }
    static void launch(const float*, const float*, int, unsigned, float, float*, unsigned, void*) {}
};
using Ds41Chunk = HostChunkTraits<640, nm::SquareForm::mul_add, false>;
using OneWave = HostChunkTraits<0, nm::SquareForm::mul_add, true>;
using OneWaveFma = HostChunkTraits<0, nm::SquareForm::fma, true>;

// Private form A (chunked, 32 lanes per chunk): lane chains acc = acc + x * x, xor butterfly, total = 0 + s0 + s1 + ...
float private_chunk_sum(const std::vector<float>& x, unsigned chunk) {
    const auto n = static_cast<unsigned>(x.size());
    float total = 0.0F;
    for (unsigned c0 = 0; c0 < n; c0 += chunk) {
        float lane[32];
        for (unsigned l = 0; l < 32; ++l) {
            float acc = 0.0F;
            for (unsigned i = c0 + l; i < c0 + chunk && i < n; i += 32U) {
                const float sq = x[i] * x[i];
                acc = acc + sq;
            }
            lane[l] = acc;
        }
        for (unsigned d = 16; d; d >>= 1U) {
            float next[32];
            for (unsigned l = 0; l < 32; ++l) next[l] = lane[l] + lane[l ^ d];
            std::memcpy(lane, next, sizeof(lane));
        }
        total = total + lane[0];
    }
    return total;
}

// Private form B (one wave): lane chains over the whole row, shfl_down tree (out-of-range lanes keep their value), lane 0.
float private_one_wave_sum(const std::vector<float>& x) {
    const auto n = static_cast<unsigned>(x.size());
    float lane[32];
    for (unsigned l = 0; l < 32; ++l) {
        float sum = 0.0F;
        for (unsigned i = l; i < n; i += 32U) {
            const float sq = x[i] * x[i];
            sum = sum + sq;
        }
        lane[l] = sum;
    }
    for (unsigned d = 16; d; d >>= 1U) {
        float next[32];
        for (unsigned l = 0; l < 32; ++l) next[l] = lane[l] + (l + d < 32U ? lane[l + d] : lane[l]);
        std::memcpy(lane, next, sizeof(lane));
    }
    return lane[0];
}

std::vector<float> row(std::mt19937& rng, unsigned n, int spread) {
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-spread, spread);
    std::vector<float> x(n);
    for (auto& v : x) v = std::ldexp(unit(rng), exponent(rng));
    return x;
}

void sums_and_rows() {
    std::mt19937 rng(20261122U);
    unsigned fma_differs = 0;
    for (const unsigned n : {32U, 64U, 512U, 640U, 1280U, 1920U, 5120U, 7680U, 2560U, 4096U}) {
        for (const int spread : {0, 4, 20}) {
            for (unsigned rep = 0; rep < 8; ++rep) {
                const auto x = row(rng, n, spread);
                if (n <= 640U || n % 640U == 0U) {
                    const float shared = nm::wave_chunk_square_sum<Ds41Chunk>(x.data(), n);
                    require(same(shared, private_chunk_sum(x, 640U)), "wave_chunk_serial<640, mul_add> == private chunk sum");
                    // Full row: r = 1 / sqrt_cr(ss / n + eps), y = (x * r) * w.
                    const auto w = row(rng, n, 1);
                    std::vector<float> y(n);
                    nm::reference<Ds41Chunk>(x.data(), w.data(), 0, n, 1e-6F, y.data(), 1);
                    const float r = 1.0F / strix::ds41::declared_sqrt(private_chunk_sum(x, 640U) / static_cast<float>(n) + 1e-6F);
                    for (unsigned i = 0; i < n; ++i) require(same(y[i], (x[i] * r) * w[i]), "row output == private kernel");
                }
                const float one = nm::wave_chunk_square_sum<OneWave>(x.data(), n);
                require(same(one, private_one_wave_sum(x)), "wave_chunk_serial<0, mul_add> == private one-wave sum");
                require(same(one, private_chunk_sum(x, n)), "shfl_down lane 0 == xor butterfly");
                if (!same(one, nm::wave_chunk_square_sum<OneWaveFma>(x.data(), n))) ++fma_differs;
            }
        }
    }
    require(fma_differs > 0, "SquareForm::fma and ::mul_add give different sums on some rows");
    std::cout << "sums: fma differs from mul_add on " << fma_differs << " rows\n";
}

void rsqrt() {
    using strix::numeric::rsqrt_correctly_rounded;
    unsigned oracle_ties = 0;
    auto check = [&](float x) {
        const float r = rsqrt_correctly_rounded(x);
        require(same(r, strix::gemma4::pow32_rsqrt(x)), "Gemma4 adapter returns the shared rsqrt");
        if (std::isfinite(x) && x > 0.0F) {
            const double exact = 1.0 / std::sqrt(static_cast<double>(x));
            const auto oracle = static_cast<float>(exact);
            if (!same(r, oracle)) {
                // Only a double rounding near a float midpoint may disagree: then r is a neighbour of the oracle.
                ++oracle_ties;
                const auto d = static_cast<std::int64_t>(bits(r)) - static_cast<std::int64_t>(bits(oracle));
                require(d == 1 || d == -1, "rsqrt_correctly_rounded within one ulp of the double oracle");
            }
        }
    };
    for (std::uint32_t b = 0x3f800000U; b < 0x40800000U; b += 7U) check(from_bits(b));      // [1, 4): every scale class
    for (std::uint64_t b = 0; b < 0x7f800000ULL; b += 4099U) check(from_bits(static_cast<std::uint32_t>(b)));  // incl. subnormals
    for (const std::uint32_t b : {0x00000001U, 0x007fffffU, 0x00800000U, 0x7f7fffffU, 0x3f800000U, 0x40800000U}) check(from_bits(b));
    require(std::isinf(rsqrt_correctly_rounded(0.0F)) && rsqrt_correctly_rounded(0.0F) > 0.0F, "rsqrt(+0) = +inf");
    require(std::isinf(rsqrt_correctly_rounded(-0.0F)) && rsqrt_correctly_rounded(-0.0F) < 0.0F, "rsqrt(-0) = -inf");
    require(std::isnan(rsqrt_correctly_rounded(-1.0F)), "rsqrt(-1) = NaN");
    require(same(rsqrt_correctly_rounded(INFINITY), 0.0F), "rsqrt(+inf) = +0");
    require(same(rsqrt_correctly_rounded(4.0F), 0.5F), "rsqrt(4) = 0.5");
    require(std::isnan(nm::inv_sqrt_correctly_rounded_finite(INFINITY)), "guard: +inf -> NaN");
    require(std::isnan(nm::inv_sqrt_correctly_rounded_finite(NAN)), "guard: NaN -> NaN");
    require(same(nm::inv_sqrt_correctly_rounded_finite(2.0F), rsqrt_correctly_rounded(2.0F)), "guard: finite passes");
    std::cout << "rsqrt: double-oracle midpoint disagreements " << oracle_ties << " (all within one ulp)\n";
}

}  // namespace

int main() {
    try {
        sums_and_rows();
        rsqrt();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << " after " << checks << " checks\n";
        return 1;
    }
    std::cout << "PASS: " << checks << " norm numeric trait checks\n";
    return 0;
}
