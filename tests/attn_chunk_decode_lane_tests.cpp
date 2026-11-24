// Host-only: the shared lane arithmetic of the chunked decode form (attn_core/chunk_decode.hpp) is two separate FP32
// operations whatever contraction flags the translation unit has. This TU is compiled with -ffp-contract=fast and FMA
// enabled where the compiler supports it (CMake). The references materialise every product, so they are independent of
// the header's helper; score_lane (the body the HIP score block calls) is compared with them for fixed widths and runtime
// widths (a width that is not a multiple of 32 or 64 included) and for an absent key row.
#include "attn_core/chunk_decode.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
namespace cd = strix::attn_core::chunk_decode;

int failures = 0;
void check(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        ++failures;
    }
}
std::uint32_t bits(float x) { return std::bit_cast<std::uint32_t>(x); }

float product(float a, float b) {
    volatile float p = a * b;
    return p;
}
// Lane partial: i = lane, lane + 32, ... < width, from +0, separate multiply and add.
float lane_ref(const float* q, const float* k, unsigned width, unsigned lane) {
    float sum = 0.0F;
    for (unsigned i = lane; i < width; i += 32U)
        sum = sum + product(q[i], k[i]);
    return sum;
}

std::uint32_t state = 1U;
float next() {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(static_cast<std::int32_t>(state >> 8U) - (1 << 23)) / static_cast<float>(1 << 22);
}

volatile float g_one_plus = 1.0F + 0x1p-23F, g_one_minus = 1.0F - 0x1p-23F;

void step_is_separate() {
    // First product -1, then (1 + 2^-23) * (1 - 2^-23): the product rounds to 1, the sum is +0; a fused step is -2^-46.
    const float r = cd::mul_add_separate(-1.0F, g_one_plus, g_one_minus);
    check(bits(r) == bits(0.0F), "mul_add_separate is not fused under contraction");
}

template <unsigned Dim>
void fixed_width(unsigned seed) {
    state = seed;
    std::vector<float> q(Dim), k(Dim);
    for (auto& x : q) x = next();
    for (auto& x : k) x = next();
    for (unsigned lane = 0; lane < 32U; ++lane)
        check(bits(cd::score_lane<Dim>(q.data(), k.data(), 0U, Dim, lane)) == bits(lane_ref(q.data(), k.data(), Dim, lane)),
              "fixed-width lane partial");
}

void runtime_width(unsigned width, unsigned seed) {
    state = seed;
    std::vector<float> q(width), k(width);
    for (auto& x : q) x = next();
    for (auto& x : k) x = next();
    for (unsigned lane = 0; lane < 32U; ++lane)
        check(bits(cd::score_lane<0U>(q.data(), k.data(), 0U, width, lane)) == bits(lane_ref(q.data(), k.data(), width, lane)),
              "runtime-width lane partial");
}

void kv_offset_head() {
    // A head that starts 96 floats into the key row (Dim 64 and runtime width 65, a partial last lane group).
    state = 31U;
    std::vector<float> q(65U), k(96U + 65U);
    for (auto& x : q) x = next();
    for (auto& x : k) x = next();
    for (unsigned lane = 0; lane < 32U; ++lane) {
        check(bits(cd::score_lane<64U>(q.data(), k.data(), 96U, 64U, lane)) ==
                  bits(lane_ref(q.data(), k.data() + 96U, 64U, lane)), "kv_offset fixed");
        check(bits(cd::score_lane<0U>(q.data(), k.data(), 96U, 65U, lane)) ==
                  bits(lane_ref(q.data(), k.data() + 96U, 65U, lane)), "kv_offset runtime");
    }
}

void cancellation_lane() {
    // Lane 0 of a width-65 row: element 0 gives -1, element 32 gives (1 + 2^-23) (1 - 2^-23), element 64 adds 0.
    std::vector<float> q(65U, 0.0F), k(65U, 0.0F);
    q[0] = 1.0F;
    k[0] = -1.0F;
    q[32] = g_one_plus;
    k[32] = g_one_minus;
    check(bits(cd::score_lane<0U>(q.data(), k.data(), 0U, 65U, 0U)) == bits(0.0F), "runtime lane chain is unfused");
}

void absent_key() {
    std::vector<float> q(64U, 1.0F);
    check(bits(cd::score_lane<64U>(q.data(), nullptr, 0U, 64U, 3U)) == bits(0.0F), "absent key row scores 0 (fixed)");
    check(bits(cd::score_lane<0U>(q.data(), nullptr, 0U, 64U, 3U)) == bits(0.0F), "absent key row scores 0 (runtime)");
}

void numerators_are_separate() {
    // acc = -1 then e * v with the cancelling pair: +0 unfused.
    const float e[2] = {1.0F, g_one_plus};
    const float rows[2] = {-1.0F, g_one_minus};
    auto value = [&](std::uint64_t p) -> const float* { return rows + p; };
    const float n = cd::chunk_numerator<1>(e, 0U, 0U, 2U, value, 0U);
    check(bits(n) == bits(0.0F), "chunk_numerator is not fused under contraction");
}

}  // namespace

int main() {
    step_is_separate();
    fixed_width<64U>(11U);
    fixed_width<128U>(12U);
    fixed_width<256U>(13U);
    runtime_width(65U, 21U);   // two lanes see three terms, the others two: a stride of 64 would skip every other term
    runtime_width(200U, 22U);
    runtime_width(32U, 23U);
    runtime_width(1U, 24U);
    kv_offset_head();
    cancellation_lane();
    absent_key();
    numerators_are_separate();
    if (failures != 0) return 1;
    std::printf("attn chunk decode lane: ok\n");
    return 0;
}
