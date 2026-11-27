// Host tests: core/iu4_gptq.hpp (error-compensated IU4 g64 copies) and core/null_ulp.hpp (one-ulp sensitivity null).
#include "core/iu4_gptq.hpp"
#include "core/null_ulp.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

// A random Q8_0 tensor (n rows of k) and its dequantised FP32 values.
std::vector<std::uint8_t> random_q8(std::size_t n, std::size_t k, std::mt19937& rng, std::vector<float>& w) {
    namespace of = strix::core::onfly;
    std::normal_distribution<float> nd(0.0F, 0.02F);
    std::vector<std::uint8_t> q8(n * of::q8_row_bytes(k));
    std::vector<float> row(32);
    for (std::size_t c = 0; c < n; ++c)
        for (std::size_t b = 0; b < k / 32U; ++b) {
            for (auto& v : row) v = nd(rng) * (b % 7U == 3U ? 6.0F : 1.0F);
            of::quantise_q8_0_block(row.data(), q8.data() + c * of::q8_row_bytes(k) + b * 34U);
        }
    w.assign(n * k, 0.0F);
    for (std::size_t c = 0; c < n; ++c)
        for (std::size_t g = 0; g < k / 64U; ++g) of::dequant_q8_group(q8.data() + c * of::q8_row_bytes(k), static_cast<unsigned>(g), w.data() + c * k + g * 64U);
    return q8;
}

void gptq_tests() {
    namespace co = strix::core;
    std::mt19937 rng(7);
    const std::size_t n = 70, k = 192;  // n not a multiple of 64: pad columns stay zero
    std::vector<float> w;
    const auto q8 = random_q8(n, k, rng, w);
    const auto ref = co::onfly::convert_q8_0_host(q8.data(), n, k, co::iu4_scale_rule_mse);
    // RTN path (no factor) == the load-time converter's host reference, byte for byte.
    const auto rtn = co::gptq::quantise(w.data(), n, k, {}, {}, co::iu4_scale_rule_mse, nullptr, 3U);
    check(rtn.packed == ref.packed && rtn.scales == ref.scales, "rtn == convert_q8_0_host");
    // Diagonal Hessian: no error moves, so the compensated copy is the RTN copy (and one dead input is zeroed).
    std::vector<double> h(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) h[i * k + i] = 1.0 + static_cast<double>(i % 5U);
    std::vector<std::uint8_t> dead;
    const auto u = co::gptq::inverse_factor(h, k, 0.01, dead);
    const auto diag = co::gptq::quantise(w.data(), n, k, u, dead, co::iu4_scale_rule_mse, nullptr, 2U);
    check(diag.packed == ref.packed && diag.scales == ref.scales, "diagonal Hessian == convert_q8_0_host");
    h[5 * k + 5] = 0.0;
    (void)co::gptq::inverse_factor(h, k, 0.01, dead);
    check(dead[5] == 1U && dead[4] == 0U, "dead input detected");
    // Correlated inputs (rank-8 structure + noise): the compensated copy has a lower output error than RTN.
    const std::size_t rows = 512, rank = 8;
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> basis(rank * k), x(rows * k);
    for (auto& v : basis) v = nd(rng);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t i = 0; i < k; ++i) {
            float s = 0.05F * nd(rng);
            for (std::size_t j = 0; j < rank; ++j) s += basis[j * k + i] * (j == r % rank ? 1.0F : 0.1F);
            x[r * k + i] = s;
        }
    std::vector<double> hx;
    co::gptq::accumulate_hessian(x.data(), rows, k, hx);
    const auto uf = co::gptq::inverse_factor(hx, k, 0.01, dead);
    std::vector<float> wr, wg;
    (void)co::gptq::quantise(w.data(), n, k, {}, {}, co::iu4_scale_rule_mse, &wr, 1U);
    (void)co::gptq::quantise(w.data(), n, k, uf, dead, co::iu4_scale_rule_mse, &wg, 4U);
    const double er = co::gptq::relative_output_error(w.data(), wr.data(), n, k, x.data(), rows);
    const double eg = co::gptq::relative_output_error(w.data(), wg.data(), n, k, x.data(), rows);
    std::printf("correlated inputs: rel output error rtn %.6f gptq %.6f\n", er, eg);
    check(eg < 0.8 * er, "compensated output error < 0.8 x RTN");
    // Block-diagonal form: one block of K == the full form bitwise; 64-wide blocks still beat RTN on these inputs.
    const auto ub = co::gptq::block_inverse_factor(x.data(), rows, k, k, 0.01, dead, 3U);
    const auto gfull = co::gptq::quantise(w.data(), n, k, uf, dead, co::iu4_scale_rule_mse, nullptr, 2U);
    const auto gblk = co::gptq::quantise(w.data(), n, k, ub, dead, co::iu4_scale_rule_mse, nullptr, 2U, k);
    check(gfull.packed == gblk.packed && gfull.scales == gblk.scales, "block == K equals the full form");
    std::vector<float> wb;
    const auto u64 = co::gptq::block_inverse_factor(x.data(), rows, k, 64U, 0.01, dead, 2U);
    (void)co::gptq::quantise(w.data(), n, k, u64, dead, co::iu4_scale_rule_mse, &wb, 2U, 64U);
    const double eb = co::gptq::relative_output_error(w.data(), wb.data(), n, k, x.data(), rows, 3U);
    std::printf("block 64: rel output error %.6f\n", eb);
    check(eb < er, "block-diagonal compensated output error < RTN");
    // Thread count never changes a bit.
    const auto g1 = co::gptq::quantise(w.data(), n, k, uf, dead, co::iu4_scale_rule_mse, nullptr, 1U);
    const auto g7 = co::gptq::quantise(w.data(), n, k, uf, dead, co::iu4_scale_rule_mse, nullptr, 7U);
    check(g1.packed == g7.packed && g1.scales == g7.scales, "threads bitwise");
}

void null_tests() {
    namespace nu = strix::core::null_ulp;
    const float vals[] = {1.0F, -2.5F, 3.0e-39F, 1.0e30F, -7.0F, 0.15625F};
    int up = 0, down = 0;
    for (std::uint32_t seed = 0; seed < 4U; ++seed)
        for (std::size_t i = 0; i < 6U; ++i) {
            const float f = vals[i], g = nu::value(f, i * 977U + seed, seed);
            const auto a = nu::bits_of(f), b = nu::bits_of(g);
            const auto d = static_cast<std::int64_t>(b) - static_cast<std::int64_t>(a);
            check(d == 1 || d == -1, "one ulp");
            if (seed == 0U) check((a ^ b) == 1U, "seed 0 flips the last bit");
            else (d > 0 ? up : down) += 1;
        }
    check(up > 0 && down > 0, "seeds move both ways");
    const float keep[] = {0.0F, -0.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    for (const float f : keep) check(nu::bits_of(nu::value(f, 3U, 9U)) == nu::bits_of(f), "zero / non-finite untouched");
    // A zero-mantissa value never moves down across the exponent boundary (it goes up).
    check(nu::bits_of(nu::value(4.0F, 0U, 0U)) == (nu::bits_of(4.0F) ^ 1U), "seed 0 on a power of two");
    for (std::uint32_t seed = 1; seed < 9U; ++seed) check(nu::bits_of(nu::value(4.0F, seed, seed)) == nu::bits_of(4.0F) + 1U, "power of two goes up");
}

}  // namespace

int main() {
    gptq_tests();
    null_tests();
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
