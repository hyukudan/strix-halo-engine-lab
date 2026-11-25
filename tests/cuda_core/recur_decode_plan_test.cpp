#include "ops/recur/recur_decode_plan.hpp"

#include <cstdio>
#include <limits>

namespace rec = strix::ops::recur;
constexpr auto kShape = rec::shape::gdn_2560_16x48x128;
using Layout = rec::GdnDecodeLayout<kShape>;
using strix::RecurrentStateType;

static_assert(Layout::hidden == 2560 && Layout::key_width == 2048 && Layout::value_width == 6144);
static_assert(Layout::channels == 10240 && Layout::history_elements == 40960);
static_assert(kShape.matrix_bytes(RecurrentStateType::f32) == 3145728);
static_assert(kShape.matrix_bytes(RecurrentStateType::f16) == 1572864);
static_assert(kShape.matrix_bytes(RecurrentStateType::bf16) == 1572864);
static_assert(Layout::a8_pitch == 256 && Layout::a8_deltas == 49152);

constexpr rec::GdnDecodeShape kOther{512, 2, 8, 64, 3, 8};
static_assert(kOther.valid());
static_assert(kOther.channels() == 768 && kOther.history_elements() == 2304);
static_assert(kOther.matrix_bytes(RecurrentStateType::bf16) == 65536);
static_assert(kOther.a8_deltas() == 128);

static_assert(!rec::GdnDecodeShape{0, 2, 8, 64, 3, 8}.valid());
static_assert(!rec::GdnDecodeShape{512, 0, 8, 64, 3, 8}.valid());
static_assert(!rec::GdnDecodeShape{512, 2, 7, 64, 3, 8}.valid());
static_assert(!rec::GdnDecodeShape{512, 2, 8, 63, 3, 8}.valid());
static_assert(!rec::GdnDecodeShape{512, 2, 8, 64, 1, 8}.valid());
static_assert(!rec::GdnDecodeShape{512, 2, 8, 64, 3, 0}.valid());
constexpr auto kMax = std::numeric_limits<unsigned>::max();
static_assert(!rec::GdnDecodeShape{kMax, 2, 8, 64, 3, 8}.valid());
static_assert(!rec::GdnDecodeShape{512, kMax, kMax, 64, 3, 8}.valid());
static_assert(!rec::GdnDecodeShape{512, 2, 8, 64, kMax, 8}.valid());
static_assert(!rec::GdnDecodeShape{512, 2, 8, 64, 3, kMax}.valid());

int main() {
    std::puts("K9G decode layout PASS");
}
