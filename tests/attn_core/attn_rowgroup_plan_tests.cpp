#include "attn_core/rowgroup_plan.hpp"

#include <cstdio>
#include <limits>

namespace rg = strix::cuda::core::rowgroup;

namespace {
struct QwenShape {
    static constexpr unsigned D = 256, Heads = 12, Rows = 4, Cap = 2051;
    static constexpr float kScaleLog2 = 0.0625F * 1.44269504088896341F;
    std::uint32_t rows;
};
struct Narrow : QwenShape { static constexpr unsigned D = 64, Heads = 1, Rows = 1, Cap = 3; };
struct Wide : QwenShape { static constexpr unsigned D = 512, Heads = 16, Rows = 8, Cap = 512; };
struct BadDim : QwenShape { static constexpr unsigned D = 96; };
struct EmptyDim : QwenShape { static constexpr unsigned D = 0; };
struct EmptyHeads : QwenShape { static constexpr unsigned Heads = 0; };
struct TooManyHeads : QwenShape { static constexpr unsigned Heads = 17; };
struct TooManyRows : QwenShape { static constexpr unsigned Rows = 9; };
struct EmptyCap : QwenShape { static constexpr unsigned Cap = 0; };
struct TooManyKeys : QwenShape { static constexpr unsigned Cap = 65536; };
struct WrongRowType : QwenShape { float rows; };
struct Nontrivial : QwenShape { ~Nontrivial() {} };
struct BadScale : QwenShape { static constexpr float kScaleLog2 = std::numeric_limits<float>::infinity(); };

static_assert(rg::LaunchPolicy<QwenShape> && rg::LaunchPolicy<Narrow> && rg::LaunchPolicy<Wide>);
static_assert(!rg::LaunchPolicy<BadDim> && !rg::LaunchPolicy<EmptyDim>);
static_assert(!rg::LaunchPolicy<EmptyHeads> && !rg::LaunchPolicy<TooManyHeads> && !rg::LaunchPolicy<TooManyRows>);
static_assert(!rg::LaunchPolicy<EmptyCap> && !rg::LaunchPolicy<TooManyKeys> && !rg::LaunchPolicy<WrongRowType>);
static_assert(!rg::LaunchPolicy<Nontrivial> && !rg::LaunchPolicy<BadScale> && !rg::LaunchPolicy<int>);
constexpr rg::GridLimits large_grid{0x7fffffffU, 65535U};
constexpr auto empty = rg::make_launch_plan<QwenShape>(0, 0, {0, 0});
constexpr auto full = rg::make_launch_plan<QwenShape>(4, 2, large_grid);
constexpr auto partial = rg::make_launch_plan<QwenShape>(7, 2, large_grid);
constexpr auto odd = rg::make_launch_plan<Narrow>(3, 1, large_grid);
static_assert(empty && empty->tiles == 0 && empty->kv_heads == 0);
static_assert(full && full->tiles == 1 && full->kv_heads == 2);
static_assert(partial && partial->tiles == 2 && partial->kv_heads == 2);
static_assert(odd && odd->tiles == 3 && odd->kv_heads == 1);
static_assert(rg::make_launch_plan<Wide>(17, 3, large_grid)->tiles == 3);
static_assert(rg::make_launch_plan<Narrow>(65535, 0x7fffffffU, large_grid).has_value());
static_assert(!rg::make_launch_plan<Narrow>(65536, 1, large_grid));
static_assert(!rg::make_launch_plan<QwenShape>(7, 0, large_grid));
static_assert(!rg::make_launch_plan<QwenShape>(0xffffffffU, 1, large_grid));
static_assert(!rg::make_launch_plan<Narrow>(1, 0x80000000U, large_grid));

constexpr rg::GridLimits small_grid{3, 2};
static_assert(rg::make_launch_plan<QwenShape>(8, 3, small_grid).has_value());
static_assert(!rg::make_launch_plan<QwenShape>(9, 3, small_grid));
static_assert(!rg::make_launch_plan<QwenShape>(8, 4, small_grid));
}  // namespace

int main() {
    std::puts("QSA launch plan PASS");
    return 0;
}
