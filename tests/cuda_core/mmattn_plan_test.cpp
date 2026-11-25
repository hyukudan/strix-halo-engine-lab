#include "attn_core/selected_block_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

namespace {
using namespace strix::attn;
using Wide = SelectedBlockShape<128, 128, 128, 16, 16>;
using Narrow = SelectedBlockShape<64, 96, 17, 3, 3>;

unsigned cases = 0;

void expect(bool okay, const char* message) {
    if (!okay) {
        std::fprintf(stderr, "MMATTN plan FAIL %s\n", message);
        std::exit(1);
    }
    ++cases;
}
}  // namespace

int main() {
    constexpr auto warp = SelectedBlockSchedule::warp, staged = SelectedBlockSchedule::staged;
    const auto w = make_selected_block_plan<Wide>(warp, 7U, 4U);
    expect(w && w->grid_x == 56U && w->grid_y == 1U && w->threads == 256U && w->shared_bytes == 0U,
           "wide warp launch");
    const auto s = make_selected_block_plan<Wide>(staged, 7U, 4U);
    expect(s && s->grid_x == 7U && s->grid_y == 4U && s->threads == 512U && s->shared_bytes == 74752U,
           "wide staged launch");
    const auto nw = make_selected_block_plan<Narrow>(warp, 3U, 3U);
    expect(nw && nw->grid_x == 4U && nw->threads == 256U, "partial warp grid");
    const auto ns = make_selected_block_plan<Narrow>(staged, 3U, 3U);
    expect(ns && ns->grid_x == 3U && ns->grid_y == 3U && ns->threads == 96U && ns->shared_bytes == 6352U,
           "narrow staged launch");
    expect(Wide::k_stride == 132U && Narrow::k_stride == 68U, "K row bank padding");
    expect(Narrow::k_elements == 1160U, "V alignment after odd number of K rows");

    for (const auto schedule : {warp, staged}) {
        const auto empty = make_selected_block_plan<Wide>(schedule, 0U, 0U);
        expect(empty && empty->grid_x == 0U && empty->grid_y == 0U &&
               empty->threads == 0U && empty->shared_bytes == 0U, "empty launch");
        expect(!make_selected_block_plan<Wide>(schedule, 1U, 0U), "nonempty launch requires groups");
    }
    constexpr auto max_x = 0x7fffffffU;
    expect(make_selected_block_plan<Wide>(staged, max_x, 65535U).has_value(), "last staged grid");
    expect(!make_selected_block_plan<Wide>(staged, max_x + 1U, 1U), "staged x overflow");
    expect(!make_selected_block_plan<Wide>(staged, 1U, 65536U), "staged y overflow");
    const auto last = make_selected_block_plan<Wide>(warp, max_x / 2U, 1U);
    expect(last && last->grid_x == max_x - 1U, "last wide warp grid");
    expect(!make_selected_block_plan<Wide>(warp, max_x / 2U + 1U, 1U), "warp grid overflow");
    constexpr auto max = std::numeric_limits<unsigned>::max();
    expect(!make_selected_block_plan<Wide>(warp, max, max), "reject before warp count multiplication");
    expect(!make_selected_block_plan<Wide>(static_cast<SelectedBlockSchedule>(42), 1U, 1U), "unknown schedule");
    std::printf("MMATTN plan PASS cases=%u\n", cases);
}
