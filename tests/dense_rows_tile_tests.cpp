// Host checks of the rows tile of the head-family dense kernel (strix/dense_rows_tile.hpp):
//  1. tile map: every in-range (row, output) is stored by exactly one (block, value), loads stay in range (clamped);
//  2. tree map: one tile pass over kTreeValues values (work items mapped by tree_item, any item order inside a level)
//     gives, for every value, the bits of the single-row tree `scratch[t] += scratch[t + d]`, d = 128 .. 1.
#include "strix/dense_rows_tile.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace drt = strix::dense_rows_tile;
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

bool tile_map(std::uint32_t rows, std::uint32_t outputs) {
    const auto g = drt::grid(rows, outputs);
    std::vector<std::uint32_t> stored(std::size_t{rows} * outputs, 0U);
    for (std::uint32_t by = 0; by < g.y; ++by)
        for (std::uint32_t bx = 0; bx < g.x; ++bx)
            for (std::uint32_t v = 0; v < drt::kValues; ++v) {
                const auto o = drt::value_output(bx, v), r = drt::value_row(by, v);
                if (drt::clamp_index(o, outputs) >= outputs || drt::clamp_index(r, rows) >= rows) return false;
                if (o < outputs && r < rows) ++stored[std::size_t{r} * outputs + o];
            }
    for (const auto n : stored)
        if (n != 1U) return false;
    return true;
}

std::uint32_t lcg(std::uint32_t& s) {
    s = s * 1664525U + 1013904223U;
    return s;
}
float random_float(std::uint32_t& s) {
    // Mixed magnitudes and signs so that a wrong pairing changes the rounding.
    const auto u = lcg(s);
    const float m = static_cast<float>(u >> 8U) / 16777216.0F;
    const float scale = (u & 3U) == 0U ? 1.0e3F : (u & 3U) == 1U ? 1.0F : (u & 3U) == 2U ? 1.0e-3F : 3.0e1F;
    return ((u >> 7U) & 1U) != 0U ? -m * scale : m * scale;
}

bool tree_map(std::uint32_t seed, bool reversed_items) {
    std::uint32_t s = seed;
    std::vector<float> leaves(std::size_t{drt::kTreeValues} * drt::kLanes);
    for (auto& f : leaves) f = random_float(s);
    // Single-row reference, one value at a time.
    std::vector<float> want(drt::kTreeValues);
    for (std::uint32_t v = 0; v < drt::kTreeValues; ++v) {
        std::vector<float> one(leaves.begin() + std::ptrdiff_t{v} * drt::kLanes,
                               leaves.begin() + std::ptrdiff_t{v + 1U} * drt::kLanes);
        for (std::uint32_t d = drt::kLanes / 2U; d > 0U; d >>= 1U)
            for (std::uint32_t t = 0; t < d; ++t) one[t] += one[t + d];
        want[v] = one[0];
    }
    // Tile pass: all values at once, per level by work item.
    auto scratch = leaves;
    for (std::uint32_t shift = drt::kLaneShift - 1U;; --shift) {
        const auto n = drt::tree_items(shift);
        for (std::uint32_t k = 0; k < n; ++k) {
            const auto item = reversed_items ? n - 1U - k : k;
            const auto t = drt::tree_item(item, shift);
            if (t.dst >= scratch.size() || t.src >= scratch.size()) return false;
            scratch[t.dst] += scratch[t.src];
        }
        if (shift == 0U) break;
    }
    for (std::uint32_t v = 0; v < drt::kTreeValues; ++v)
        if (std::memcmp(&scratch[std::size_t{v} * drt::kLanes], &want[v], sizeof(float)) != 0) return false;
    return true;
}

}  // namespace

int main() {
    check(tile_map(2048, 512), "tile map rows 2048 x outputs 512 (router): each value stored once, loads in range");
    check(tile_map(37, 16), "tile map rows 37 x outputs 16 (indexer proj, ragged rows)");
    check(tile_map(13, 1003), "tile map rows 13 x outputs 1003 (ragged both)");
    check(tile_map(2, 1), "tile map rows 2 x outputs 1");
    check(tree_map(1U, false), "tree map == single-row tree, bitwise (items ascending)");
    check(tree_map(7U, true), "tree map == single-row tree, bitwise (items descending)");
    check(tree_map(12345U, false), "tree map == single-row tree, bitwise (seed 12345)");
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
