#include "strix/numeric/math_gfx1151.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

namespace gfx = strix::numeric::gfx1151;
using Bits = std::uint32_t;

constexpr float f32(Bits bits) { return std::bit_cast<float>(bits); }

void expect(const char* name, float actual, Bits expected) {
    const auto bits = std::bit_cast<Bits>(actual);
    if (bits == expected) return;
    std::fprintf(stderr, "%s: got %08x, expected %08x\n", name,
                 static_cast<unsigned>(bits), static_cast<unsigned>(expected));
    std::exit(1);
}

void require(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

void set_bit(std::vector<Bits>& words, Bits index) {
    words[index >> 5U] |= 1U << (index & 31U);
}

}  // namespace

int main() {
    constexpr Bits exp_span = 25U << 23U;
    std::vector<Bits> exp((2U * exp_span) / 32U);
    std::vector<Bits> rsq(1U << 19U);
    std::vector<Bits> rcp(1U << 18U);
    gfx::Tables tables{};
    tables.exp_bits = exp.data();
    tables.rsq_bits = rsq.data();
    tables.rcp_bits = rcp.data();

    // Exact probe points and wrapper boundary cases.
    expect("v_exp(+0)", gfx::amd_v_exp(tables, 0.0F), 0x3f800000U);
    expect("v_exp(-0)", gfx::amd_v_exp(tables, -0.0F), 0x3f800000U);
    expect("v_rsq(1/4)", gfx::amd_v_rsq(tables, 0.25F), 0x40000000U);
    expect("v_rsq(16)", gfx::amd_v_rsq(tables, 16.0F), 0x3e800000U);
    expect("v_rcp(-8)", gfx::amd_v_rcp(tables, -8.0F), 0xbe000000U);
    expect("v_rcp(1/2)", gfx::amd_v_rcp(tables, 0.5F), 0x40000000U);
    expect("expf(+inf)", gfx::amd_expf(tables, f32(0x7f800000U)), 0x7f800000U);
    expect("expf(-inf)", gfx::amd_expf(tables, f32(0xff800000U)), 0x00000000U);
    expect("expf(-100)", gfx::amd_expf(tables, -100.0F), 0x0000001bU);
    expect("rsqrtf(-0)", gfx::amd_rsqrtf(tables, -0.0F), 0xff800000U);
    expect("rsqrtf(subnormal power of four)", gfx::amd_rsqrtf(tables, f32(0x00200000U)), 0x5f800000U);

    // Synthetic bitmap round-trip. Check both exp signs and both directions
    // of correction. These entries exercise the format, not probe readings.
    const Bits half = 0x3f000000U - (102U << 23U);
    expect("uncorrected exp", gfx::amd_v_exp(tables, 0.5F), 0x3fb504f3U);
    expect("uncorrected rsq", gfx::amd_v_rsq(tables, 2.0F), 0x3f3504f3U);
    expect("uncorrected rcp", gfx::amd_v_rcp(tables, -3.0F), 0xbeaaaaabU);
    set_bit(exp, half);
    set_bit(rsq, 1U << 23U);
    set_bit(rcp, 1U << 22U);
    expect("exp positive bitmap", gfx::amd_v_exp(tables, 0.5F), 0x3fb504f4U);
    expect("exp sign isolation", gfx::amd_v_exp(tables, -0.5F), 0x3f3504f3U);
    set_bit(exp, exp_span + half);
    expect("exp negative bitmap", gfx::amd_v_exp(tables, -0.5F), 0x3f3504f4U);
    expect("rsq bitmap scale", gfx::amd_v_rsq(tables, 0.5F), 0x3fb504f4U);
    expect("rcp bitmap scale/sign", gfx::amd_v_rcp(tables, -3.0F), 0xbeaaaaaaU);

    // Hard-list values take precedence over the bitmap and are re-scaled
    // after the canonical lookup.
    constexpr std::array<Bits, 2> exp_in{0x3e800000U, 0xbf000000U};
    constexpr std::array<float, 2> exp_out{f32(0x3f9837f1U), f32(0x3f3504f3U)};
    constexpr std::array<Bits, 1> rsq_in{0x40000000U}, rcp_in{0x3fc00000U};
    constexpr std::array<float, 1> rsq_out{f32(0x3f3504f3U)}, rcp_out{f32(0x3f2aaaabU)};
    tables.exp_hard_in = exp_in.data();
    tables.exp_hard_out = exp_out.data();
    tables.exp_hard_n = 2;
    tables.rsq_hard_in = rsq_in.data();
    tables.rsq_hard_out = rsq_out.data();
    tables.rsq_hard_n = 1;
    tables.rcp_hard_in = rcp_in.data();
    tables.rcp_hard_out = rcp_out.data();
    tables.rcp_hard_n = 1;
    expect("exp first hard entry", gfx::amd_v_exp(tables, 0.25F), 0x3f9837f1U);
    expect("exp second hard entry", gfx::amd_v_exp(tables, -0.5F), 0x3f3504f3U);
    expect("exp hard miss", gfx::amd_v_exp(tables, 0.5F), 0x3fb504f4U);
    expect("rsq hard scale", gfx::amd_v_rsq(tables, 0.5F), 0x3fb504f3U);
    expect("rcp hard scale/sign", gfx::amd_v_rcp(tables, -3.0F), 0xbeaaaaabU);

    for (int i = 1; i <= 512; ++i) {
        const float a = static_cast<float>(i) / 512.0F;
        const float b = static_cast<float>(i + 1) / 512.0F;
        require(gfx::amd_expf(tables, 24.0F * (a - 0.5F)) <= gfx::amd_expf(tables, 24.0F * (b - 0.5F)),
                "expf monotonicity");
        require(gfx::amd_v_rsq(tables, a) >= gfx::amd_v_rsq(tables, b), "positive rsq monotonicity");
        require(gfx::amd_v_rcp(tables, a) >= gfx::amd_v_rcp(tables, b), "positive rcp monotonicity");
        require(gfx::amd_v_rcp(tables, -b) >= gfx::amd_v_rcp(tables, -a), "negative rcp monotonicity");
    }
    std::puts("N1 gfx1151 host: probes, tables and monotone samples PASS");
}
