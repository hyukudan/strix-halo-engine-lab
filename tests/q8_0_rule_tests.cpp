// Shared Q8_0 group rule (strix/q8_0_rule.hpp; ggml quantize_row_q8_0_ref): the values the BF16-source draft head gather
// (src/hip_mtp_draft_head.hip) and any load-time Q8_0 builder store. Literal expectations, no re-derivation of the rule.
#include "strix/q8_0_rule.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define Q8_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

namespace q8 = strix::q8_0;

int main() {
    static_assert(q8::group_values == 32U && q8::block_bytes == 34U);
    {  // zero group: d = 0, id = 0, every code 0
        float v[32]{};
        const auto g = q8::quantize_group(v);
        Q8_CHECK(g.d == 0.0F, "zero d %g", static_cast<double>(g.d));
        for (unsigned j = 0; j < 32; ++j) Q8_CHECK(g.q[j] == 0, "zero q[%u] %d", j, g.q[j]);
    }
    {  // amax 127 -> d = 1 exactly: ties round away from zero (roundf), the max maps to +-127
        float v[32]{};
        v[0] = 127.0F; v[1] = 0.5F; v[2] = -0.5F; v[3] = 1.5F; v[4] = -2.5F; v[5] = 2.49F; v[6] = -126.5F; v[7] = 0.25F;
        const auto g = q8::quantize_group(v);
        Q8_CHECK(g.d == 1.0F, "d %g", static_cast<double>(g.d));
        const int want[8]{127, 1, -1, 2, -3, 2, -127, 0};
        for (unsigned j = 0; j < 8; ++j) Q8_CHECK(g.q[j] == want[j], "tie q[%u] %d want %d", j, g.q[j], want[j]);
    }
    {  // the max is negative: d = amax / 127 = 2, codes keep their sign
        float v[32]{};
        v[0] = -254.0F; v[1] = 3.0F; v[2] = -3.0F; v[3] = 100.0F;
        const auto g = q8::quantize_group(v);
        Q8_CHECK(g.d == 2.0F, "neg d %g", static_cast<double>(g.d));
        Q8_CHECK(g.q[0] == -127 && g.q[1] == 2 && g.q[2] == -2 && g.q[3] == 50, "neg q %d %d %d %d", g.q[0], g.q[1], g.q[2], g.q[3]);
    }
    {  // a ramp: d = 16 / 127, codes ~ v * 7.9375 (no value near a tie)
        float v[32];
        for (unsigned j = 0; j < 32; ++j) v[j] = static_cast<float>(static_cast<int>(j) - 16);
        const auto g = q8::quantize_group(v);
        Q8_CHECK(g.d == 16.0F / 127.0F, "ramp d %.9g", static_cast<double>(g.d));
        const int at[6][2]{{0, -127}, {16, 0}, {17, 8}, {18, 16}, {20, 32}, {31, 119}};
        for (const auto& a : at) Q8_CHECK(g.q[a[0]] == a[1], "ramp q[%d] %d want %d", a[0], g.q[a[0]], a[1]);
    }
    {  // BF16 bits -> FP32 exactly
        Q8_CHECK(q8::bf16_value(0x3F80U) == 1.0F && q8::bf16_value(0xC000U) == -2.0F && q8::bf16_value(0U) == 0.0F &&
                     q8::bf16_value(0x3E20U) == 0.15625F,
                 "bf16 values");
    }
    std::printf("q8_0 rule tests PASS\n");
    return 0;
}
