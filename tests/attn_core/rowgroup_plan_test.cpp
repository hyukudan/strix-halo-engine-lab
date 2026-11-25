#include "attn_core/rowgroup_plan.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
namespace rg = strix::cuda::core::rowgroup;

struct Layout {
    static constexpr unsigned D = 64, Heads = 8, Rows = 3, Cap = 7;
};

struct WiderLayout {
    static constexpr unsigned D = 128, Heads = 16, Rows = 7, Cap = 12;
};

static_assert(rg::Shape<Layout>::Threads == 96);
static_assert(rg::Shape<Layout>::Pitch == 72 && rg::Shape<Layout>::UnionCap == 21);
static_assert(rg::Shape<Layout>::FlashSmem == 9216 && rg::Shape<Layout>::UnionSmem == 132);
static_assert(rg::Shape<WiderLayout>::FlashSmem == 30464 && rg::Shape<WiderLayout>::UnionSmem == 518);
static_assert(rg::workspace_bytes<Layout>(0) == 0);
static_assert(rg::workspace_bytes<Layout>(3) == 328);
}  // namespace

int main() {
    alignas(std::uint32_t) std::array<std::uint8_t, rg::workspace_bytes<Layout>(3)> workspace{};
    const auto layout = rg::carve<Layout>(workspace.data(), 3);
    if (static_cast<void*>(layout.tok) != workspace.data() ||
        layout.mask != workspace.data() + 252 ||
        static_cast<void*>(layout.count) != workspace.data() + 316) {
        std::fputs("rowgroup plan FAIL: workspace offsets\n", stderr);
        return EXIT_FAILURE;
    }
    std::puts("rowgroup plan PASS");
}
