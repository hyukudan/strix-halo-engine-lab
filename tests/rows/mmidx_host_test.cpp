#include "core/dense_f32_input_major_plan.hpp"

#include <cstddef>
#include <iostream>
#include <limits>

namespace {

using strix::core::plan_f32_input_major_rows;

struct FixtureTile {
    static constexpr unsigned rows = 16, outputs = 128, k = 64;
};

struct AlternativeTile {
    static constexpr unsigned rows = 2, outputs = 3, k = 4;
};

constexpr auto alternative_grid = plan_f32_input_major_rows<AlternativeTile>(3, 4, 7, 8, 2, 3);
static_assert(alternative_grid.valid() && alternative_grid.row_tiles == 2 && alternative_grid.output_tiles == 3);
static_assert(!plan_f32_input_major_rows<AlternativeTile>(3, 4, 7, 8, 1, 3).valid());
static_assert(!plan_f32_input_major_rows<AlternativeTile>(3, 4, 7, 8, 2, 2).valid());
static_assert(!plan_f32_input_major_rows<AlternativeTile>(3, 6, 7, 8, 2, 3).valid());

struct Case {
    std::size_t rows;
    unsigned in, out, stride, row_tiles, output_tiles;
};

constexpr Case cases[] = {
    {0, 64, 1, 1, 0, 1},
    {1, 64, 1, 1, 1, 1},
    {15, 128, 127, 131, 1, 1},
    {16, 6144, 128, 128, 1, 1},
    {17, 6144, 129, 137, 2, 2},
    {33, 6144, 256, 512, 3, 2},
    {65, 6144, 128, 128, 5, 1},
    {1, 64, 128U * 65535U, 128U * 65535U, 1, 65535},
    {1, 64, 128U * 65535U + 1, 128U * 65535U + 1, 0, 0},
    {static_cast<std::size_t>(0x7fffffffU) * 16, 64, 1, 1, 0x7fffffffU, 1},
    {static_cast<std::size_t>(0x7fffffffU) * 16 + 1, 64, 1, 1, 0, 0},
    {1, 0, 1, 1, 0, 0},
    {1, 63, 1, 1, 0, 0},
    {1, 65, 1, 1, 0, 0},
    {1, 64, 0, 0, 0, 0},
    {1, 64, 129, 128, 0, 0},
    {std::numeric_limits<std::size_t>::max(), 64, 1, 1, 0, 0},
    {static_cast<std::size_t>(0x7fffffffU) * 16, 0xffffffc0U, 1, 1, 0, 0},
    {static_cast<std::size_t>(0x7fffffffU) * 16, 64, 1, 0xffffffffU, 0, 0},
};

}  // namespace

int main() {
    for (const auto& c : cases) {
        const auto grid = plan_f32_input_major_rows<FixtureTile>(
            c.rows, c.in, c.out, c.stride, 0x7fffffffU, 65535U);
        if (grid.row_tiles != c.row_tiles || grid.output_tiles != c.output_tiles ||
            grid.valid() != (c.output_tiles != 0)) {
            std::cerr << "MMIDX plan FAIL for rows=" << c.rows << " in=" << c.in << " out=" << c.out
                      << " stride=" << c.stride << '\n';
            return 1;
        }
    }
    std::cout << "MMIDX plan PASS cases=" << sizeof(cases) / sizeof(*cases) << '\n';
}
