// Host rule of the shared eager selected score and value forms (src/attn_core/eager_selected.hpp): one product per
// lane + LDS halving-tree pairing, 1 / sqrtf(head_dim) scale, ascending fused value chain, gated rounded epilogue.
#include "attn_core/eager_selected_rule.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace sel = strix::attn_core::eager_selected;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::uint32_t bits(float x) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

float bf16_round(float x) {  // round to nearest even BF16, quiet NaN
    std::uint32_t u = bits(x);
    if ((u & 0x7fffffffU) > 0x7f800000U)
        u = (u | 0x00400000U) & 0xffff0000U;
    else
        u = (u + 0x7fffU + ((u >> 16U) & 1U)) & 0xffff0000U;
    float y = 0.0F;
    std::memcpy(&y, &u, sizeof y);
    return y;
}

float exp_f(float x) { return std::exp(x); }

std::uint32_t lcg = 12345U;
float next_value() {
    lcg = lcg * 1664525U + 1013904223U;
    return bf16_round(static_cast<float>(static_cast<std::int32_t>(lcg >> 8U) - (1 << 23)) / static_cast<float>(1 << 21));
}

// The block as written per lane (scratch array, level loop over lanes below the stride): the private form.
template <unsigned Threads>
float block_score(const std::vector<float>& q, const std::vector<float>& k, std::uint32_t head_dim) {
    float scratch[Threads];
    for (std::uint32_t f = 0; f < Threads; ++f) scratch[f] = f < head_dim ? q[f] * k[f] : 0.0F;
    for (std::uint32_t stride = Threads / 2U; stride > 0; stride >>= 1U)
        for (std::uint32_t f = 0; f < stride; ++f) scratch[f] += scratch[f + stride];
    return scratch[0] / std::sqrt(static_cast<float>(head_dim));
}

}  // namespace

int main() {
    // Pairing: 4 lanes a, b, c, d -> (a + c) + (b + d), not ((a + b) + c) + d.
    {
        const float q[4] = {1.0F, 1.0F, 1.0F, 1.0F}, k[4] = {1.0e8F, 1.0F, -1.0e8F, 1.0F};
        check(sel::score_rule<4>(q, k, 4) == 1.0F, "halving tree pairs lane l with l + h");
    }
    // The fused chain differs from the unfused one: fma(p, v, s) keeps the exact product.
    {
        const float p[2] = {1.0F, 1.0F + 0x1.0p-12F}, v[2] = {1.0F, -(1.0F + 0x1.0p-12F)};
        const float fused = sel::value_rule(p, v, 2);  // unfused: the product rounds to -(1 + 2^-11), sum -2^-11
        check(bits(fused) == bits(-(0x1.0p-11F + 0x1.0p-24F)), "value chain is fused (fmaf) in ascending slot order");
    }
    // Rule == private block form, bit for bit, over head_dim in {64, 128, 256} with 256 lanes.
    for (std::uint32_t head_dim : {64U, 128U, 256U}) {
        for (unsigned trial = 0; trial < 64U; ++trial) {
            std::vector<float> q(head_dim), k(head_dim);
            for (auto& x : q) x = next_value();
            for (auto& x : k) x = next_value();
            check(bits(sel::score_rule<256>(q.data(), k.data(), head_dim)) == bits(block_score<256>(q, k, head_dim)),
                  "score rule == private block form");
        }
    }
    // Value chain over 8-slot batches plus a tail == one ascending chain (batching is latency only).
    {
        std::vector<float> p(37), v(37);
        for (auto& x : p) x = bf16_round(std::fabs(next_value()) / 64.0F);
        for (auto& x : v) x = next_value();
        float batched = 0.0F;
        std::uint32_t slot = 0;
        for (; slot + 8U <= 37U; slot += 8U)
            for (unsigned i = 0; i < 8U; ++i) batched = std::fma(p[slot + i], v[slot + i], batched);
        for (; slot < 37U; ++slot) batched = std::fma(p[slot], v[slot], batched);
        check(bits(batched) == bits(sel::value_rule(p.data(), v.data(), 37U)), "8-slot batching keeps the chain");
    }
    // Gated epilogue: BF16 attention, BF16 sigmoid gate, BF16 product.
    {
        const float out = sel::gated_output_rule(1.2345678F, 0.5F, exp_f, bf16_round);
        const float gate = bf16_round(1.0F / (1.0F + std::exp(-0.5F)));
        check(bits(out) == bits(bf16_round(bf16_round(1.2345678F) * gate)), "gated epilogue rounding order");
        check(bits(bf16_round(out)) == bits(out), "gated output is BF16-representable");
    }
    if (failures != 0) return 1;
    std::printf("eager_selected: all checks passed\n");
    return 0;
}
