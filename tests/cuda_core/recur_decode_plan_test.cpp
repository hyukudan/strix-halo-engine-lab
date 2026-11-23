#include "cuda/core/recur_decode_plan.hpp"

#include <cstdio>

using Layout = strix::cuda::recur::Gfx1151Decode;
using strix::RecurrentStateType;

static_assert(Layout::hidden == 2560 && Layout::key_width == 2048 && Layout::value_width == 6144);
static_assert(Layout::channels == 10240 && Layout::history_elements == 40960);
static_assert(Layout::matrix_bytes(RecurrentStateType::f32) == 3145728);
static_assert(Layout::matrix_bytes(RecurrentStateType::f16) == 1572864);
static_assert(Layout::matrix_bytes(RecurrentStateType::bf16) == 1572864);
static_assert(Layout::a8_pitch == 256 && Layout::a8_deltas == 49152);

int main() {
    std::puts("K9G decode layout PASS");
}
