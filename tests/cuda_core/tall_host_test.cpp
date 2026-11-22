#include "cuda/core/tall_k_plan.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace tall = strix::cuda::core::tall;
using enum tall::Reduction;
using enum tall::Store;

using Bf16 = tall::Dense16<false, 3, 4, warp_chain, 1>;
using Compact = tall::Dense16<false, 3, 1, warp_chain, 1, true, 2, 1>;
using Q8 = tall::Dense16<false, 2, 2, q8_group_fold, 2>;
using F16 = tall::Dense16<true, 1, 4, nvfp4_split, 2>;
using Fp8 = tall::Dense8<4, 2, 3>;
using Fp4 = tall::Dense4<false, false>;
using Fp4Bulk = tall::Dense4<true, true>;

static_assert(Bf16::operand == tall::Operand::bf16 && Bf16::prologue == tall::Prologue::bf16_planes);
static_assert(Bf16::weight == tall::WeightPrep::bf16 && Bf16::terms == 3 && Bf16::tile.stages == 3);
static_assert(Bf16::tile.rows == 128 && Bf16::tile.outputs == 128 && Bf16::tile.k == 32 && Bf16::tile.threads == 256);
static_assert(Bf16::admits(plain) && Bf16::admits(residual) && Bf16::admits(silu_mul) && !Bf16::admits(round_bf16));
static_assert(Compact::tile.rows == 32 && Compact::tile.outputs == 32 && Compact::tile.threads == 128);
static_assert(Compact::admits(round_bf16) && !Compact::admits(residual) && !Compact::admits(plain));
static_assert(Q8::operand == tall::Operand::bf16 && Q8::weight == tall::WeightPrep::q8_bf16);
static_assert(Q8::reduction == q8_group_fold && Q8::tile.rows == 64 && Q8::tile.k == 64);
static_assert(F16::operand == tall::Operand::f16 && F16::scale == tall::Scale::tensor && F16::reduction == nvfp4_split);
static_assert(F16::weight == tall::WeightPrep::nvfp4_f16 && F16::admits(silu_mul));
static_assert(Fp8::operand == tall::Operand::fp8 && Fp8::weight == tall::WeightPrep::nvfp4_e4m3);
static_assert(Fp8::tile.rows == 128 && Fp8::tile.outputs == 128 && Fp8::tile.k == 64 && Fp8::SMEM == 98304);
static_assert(Fp8::scale == tall::Scale::row_tensor && Fp8::admits(residual) && !Fp8::admits(glu));
static_assert(Fp4::operand == tall::Operand::fp4 && Fp4::prologue == tall::Prologue::e2m1_row_tile);
static_assert(Fp4::tile.rows == 256 && Fp4::tile.outputs == 128 && Fp4::tile.k == 128 && Fp4::tile.threads == 256);
static_assert(Fp4::scale == tall::Scale::row_tensor && Fp4::glu_outputs == 64 && Fp4::admits(glu) && !Fp4::admits(glu_bf16));
static_assert(Fp4Bulk::tile.threads == 384 && Fp4Bulk::tile.stages == 3 && Fp4Bulk::admits(glu_bf16) && !Fp4Bulk::admits(glu));

static void check(const tall::KPlan& actual, const std::vector<int>& blocks,
                  const std::vector<int>& ends, const char* name) {
    if (actual.blocks != blocks || actual.ends != ends) {
        std::fprintf(stderr, "FAIL %s K plan\n", name);
        std::exit(1);
    }
}

int main() {
    // The 33rd K32 goes back to warp zero after eight four-block groups.
    const auto bf16 = tall::k_plan(warp_chain, 1056, 0, 1);
    check(bf16,
          {0, 1, 2, 3, 32, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
          {5, 9, 13, 17, 21, 25, 29, 33}, "BF16 warp chain");
    // These exact K32 partials distinguish the incumbent warp tree from
    // a full-K sum in ascending order, which would round away the unit.
    std::array<float, 33> partials{};
    partials[0] = 0x1p24F;
    partials[4] = 1.0F;
    partials[32] = -0x1p24F;
    float accumulated = 0.0F;
    std::size_t next = 0;
    for (int end : bf16.ends) {
        float segment = 0.0F;
        for (; next < static_cast<std::size_t>(end); ++next)
            segment += partials[static_cast<std::size_t>(bf16.blocks[next])];
        accumulated += segment;
    }
    if (std::bit_cast<std::uint32_t>(accumulated) != 0x3f800000U) {
        std::fputs("FAIL BF16 K partial cancellation\n", stderr);
        return 1;
    }
    // The 65th through 68th K32s go back to warp zero with the Q8 fold's
    // eight-block group size. Two-block units preserve segment boundaries.
    check(tall::k_plan(q8_group_fold, 2176, 0, 2),
          {0, 2, 4, 6, 64, 66, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
           32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62},
          {6, 10, 14, 18, 22, 26, 30, 34}, "Q8 group fold");
    check(tall::k_plan(warp_chain, 128, 0, 1), {0, 1, 2, 3}, {4, 4, 4, 4, 4, 4, 4, 4}, "empty BF16 warp segments");
    check(tall::k_plan(nvfp4_split, 640, 3, 1),
          {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
          {4, 12, 20}, "F16 uneven NVFP4 split");
    check(tall::k_plan(nvfp4_split, 640, 3, 2),
          {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}, {2, 6, 10}, "F16 two-block pipeline");
    std::puts("PASS K2c CUDA tall host oracle");
}
