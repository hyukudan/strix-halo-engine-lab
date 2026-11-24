// Host checks of the A4W4 projection-call layout (core/iu4a4_projection_layout.hpp): admission limits, segment
// boundaries on whole 16-column tiles, the owner of every column, and the image byte sizes.
#include "core/iu4a4_projection_layout.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace {
namespace ip = strix::core::iu4a4_projections;
int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
}  // namespace

int main() {
    // Inkling 2 Mini attention call: Q 3,072 + relative 384 + gate 3,072 + K 1,024 + V 1,024 from 2,304 inputs.
    static constexpr std::array<unsigned, 5> attention{3072U, 384U, 3072U, 1024U, 1024U};
    constexpr ip::Layout a = ip::make_layout(2304U, attention);
    static_assert(a.admitted() && a.count == 5U && a.n == 8576U && a.in == 2304U);
    static_assert(a.first_col[0] == 0U && a.first_col[1] == 3072U && a.first_col[2] == 3456U && a.first_col[3] == 6528U &&
                  a.first_col[4] == 7552U && a.first_col[5] == 8576U && a.first_col[8] == 8576U);
    static_assert(a.segment(0U) == 0U && a.segment(3071U) == 0U && a.segment(3072U) == 1U && a.segment(3455U) == 1U &&
                  a.segment(3456U) == 2U && a.segment(8575U) == 4U);
    static_assert(a.weight_code_bytes() == std::size_t{8576U} * 2304U / 2U && a.weight_step_bytes() == std::size_t{8576U} * 18U * 4U);
    // Every column's owner matches the segment ranges, and every 16-column tile lies inside one segment.
    for (unsigned c = 0; c < a.n; ++c) {
        const unsigned s = a.segment(c);
        expect(c >= a.first_col[s] && c < a.first_col[s + 1U], "owner range");
        if (c % 16U == 0U) expect(a.segment(c + 15U) == s, "tile inside one segment");
    }
    for (unsigned s = 0; s < a.count; ++s) expect(a.outputs(s) == attention[s], "segment outputs");
    // Refusals: no segment, too many, a segment off the 16-column grid or empty, inputs off the 128 block or too long.
    const std::vector<unsigned> none;
    const std::vector<unsigned> nine(9U, 16U), eight(8U, 16U);
    const std::array<unsigned, 2> odd{3072U, 24U}, empty{16U, 0U};
    static constexpr std::array<unsigned, 1> one{2304U};
    expect(!ip::make_layout(2304U, none).admitted(), "no segment refused");
    expect(!ip::make_layout(2304U, nine).admitted(), "nine segments refused");
    expect(ip::make_layout(2304U, eight).admitted() && ip::make_layout(2304U, eight).n == 128U, "eight segments admitted");
    expect(!ip::make_layout(2304U, odd).admitted(), "24-output segment refused");
    expect(!ip::make_layout(2304U, empty).admitted(), "empty segment refused");
    expect(!ip::make_layout(2300U, one).admitted(), "inputs off the block refused");
    expect(!ip::make_layout(0U, one).admitted(), "zero inputs refused");
    expect(!ip::make_layout(ip::kMaxInputs + 128U, one).admitted(), "inputs past the contract refused");
    expect(ip::make_layout(ip::kMaxInputs, one).admitted(), "inputs at the contract limit admitted");
    const std::array<unsigned, 2> overflow{0xFFFFFFF0U, 32U};
    expect(!ip::make_layout(128U, overflow).admitted(), "column total overflow refused");
    // The output call: O 3,072 -> 2,304, one segment.
    constexpr ip::Layout o = ip::make_layout(3072U, one);
    static_assert(o.admitted() && o.count == 1U && o.n == 2304U && o.segment(2303U) == 0U);
    std::printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
