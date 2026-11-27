// Rank slices of a Strix pack SP4 matrix (strix/weight/sp4_slice.hpp), CPU only: the output slice is a view of whole
// rows, the input slice gathers the same superblocks of every row byte for byte, bad geometries are refused.
#include "strix/weight/sp4_slice.hpp"

#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace sl = strix::weight::sp4_slice;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// A synthetic SP4 matrix whose every byte tells where it sits (row, superblock, byte).
std::vector<std::uint8_t> synthetic(std::uint64_t rows, std::uint64_t columns) {
    std::vector<std::uint8_t> m(rows * sl::row_bytes(columns));
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = static_cast<std::uint8_t>((i * 2654435761ULL) >> 13);
    return m;
}

void test_slices() {
    const std::uint64_t rows = 6, cols = 5 * 128;
    const auto m = synthetic(rows, cols);
    check(sl::kBlock == 128U && sl::row_bytes(cols) == 5 * 68 && sl::admitted(m.size(), rows, cols), "geometry");
    check(!sl::admitted(m.size(), rows, cols - 64) && !sl::admitted(m.size(), 0, cols) && !sl::admitted(m.size() - 1, rows, cols),
          "admitted refuses a partial superblock, no rows and a short payload");
    const auto out = sl::output_slice(m, rows, cols, 2, 3);
    check(out.size() == 3 * sl::row_bytes(cols) && out.data() == m.data() + 2 * sl::row_bytes(cols), "output slice is a view of rows 2..4");
    const auto in = sl::input_slice(m, rows, cols, 128, 256);
    bool same = in.size() == rows * 2 * 68;
    for (std::uint64_t r = 0; same && r < rows; ++r)
        for (std::uint64_t b = 0; b < 2 * 68; ++b) same = same && in[r * 2 * 68 + b] == m[r * 5 * 68 + 68 + b];
    check(same, "input slice keeps superblocks 1..2 of every row");
    check(sl::input_slice(m, rows, cols, 0, cols) == std::vector<std::uint8_t>(m.begin(), m.end()), "full input slice = the matrix");
    check(throws([&] { (void)sl::input_slice(m, rows, cols, 64, 128); }), "unaligned input slice refused");
    check(throws([&] { (void)sl::input_slice(m, rows, cols, 512, 256); }), "input slice past the end refused");
    check(throws([&] { (void)sl::input_slice(m, rows, cols, 0, 0); }), "empty input slice refused");
    check(throws([&] { (void)sl::output_slice(m, rows, cols, 4, 3); }), "output slice past the end refused");
    check(throws([&] { (void)sl::output_slice(m, rows, cols, 0, 0); }), "empty output slice refused");
    check(throws([&] { (void)sl::output_slice(std::span<const std::uint8_t>(m.data(), m.size() - 1), rows, cols, 0, 1); }), "short payload refused");
}

}  // namespace

int main() {
    test_slices();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("sp4 slice tests passed");
    return 0;
}
