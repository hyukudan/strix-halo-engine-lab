// Host check of the wide tile form of the tensor-core GSQ expert kernels (src/cuda/core/native_mma.hpp; kernels in
// src/cuda/core/moe_a8_mma.cuh). The form is bitwise the tile form because:
//   tiles   : cutting an expert's run at multiples of 128 gives exactly the unions of consecutive pairs of its 64-row tiles,
//             every assignment keeps its m16 block position (offset % 16) and its m16 blocks in order, and the m16 blocks
//             of a wide tile are those of its two 64-row tiles (so the MMAs issued are the same);
//   outputs : the wide warp map (column_block, first_part, warp_parts) owns every (output, part) of a block exactly once,
//             the same set the tile form's 8 warps own (project_output / down_output);
//   capacity: tile_bound bounds the tiles of both cuts; the wide form is chosen by wide_form at the threshold only.
// No GPU. Release build, ctest; failures print and return 1 (no dialogs).
#include "../../src/cuda/core/native_mma.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <tuple>
#include <vector>

namespace nm = strix::cuda::core::native_mma;
using u32 = std::uint32_t;

static int g_fail = 0;
static void check(bool ok, const char* what, u32 a, u32 b) {
    if (!ok && g_fail++ < 10) std::printf("FAIL %s (%u, %u)\n", what, a, b);
}

struct Tile { u32 expert, begin, count; };
// The CSR cut of csr_kernel (qwen_moe.cu): expert e's run starts at the prefix sum of the counts and is cut every cap rows.
static std::vector<Tile> cut(const std::vector<u32>& n, u32 cap) {
    std::vector<Tile> tiles;
    u32 off = 0;
    for (u32 e = 0; e < n.size(); ++e) {
        for (u32 b = 0; b * cap < n[e]; ++b) tiles.push_back({e, off + b * cap, n[e] - b * cap < cap ? n[e] - b * cap : cap});
        off += n[e];
    }
    return tiles;
}

static void check_tiles(const std::vector<u32>& n) {
    const auto narrow = cut(n, nm::kTileRows), wide = cut(n, nm::kWideRows);
    u32 assignments = 0;
    for (u32 x : n) assignments += x;
    const u32 experts = static_cast<u32>(n.size());
    check(narrow.size() <= nm::tile_bound(assignments, experts, nm::kTileRows), "64-row tiles within tile_bound", assignments, 0);
    check(wide.size() <= nm::tile_bound(assignments, experts, nm::kWideRows), "wide tiles within tile_bound", assignments, 0);
    std::size_t k = 0;
    u32 narrow_mblocks = 0, wide_mblocks = 0;
    for (const Tile& t : narrow) narrow_mblocks += nm::m_blocks(t.count);
    for (const Tile& w : wide) {
        wide_mblocks += nm::m_blocks(w.count);
        check(w.count <= nm::kWideRows && nm::m_blocks(w.count) <= nm::kWideMBlocks, "wide tile capacity", w.begin, w.count);
        // The 64-row tiles inside this wide tile, in order, cover it exactly.
        u32 covered = 0;
        while (k < narrow.size() && narrow[k].expert == w.expert && narrow[k].begin < w.begin + w.count) {
            const Tile& t = narrow[k];
            check(t.begin == w.begin + covered, "64-row tiles are consecutive inside the wide tile", t.begin, w.begin);
            for (u32 r = 0; r < t.count; ++r) {
                const u32 in_wide = covered + r;  // the assignment's row in the wide tile, r its row in the 64-row tile
                check(in_wide % 16U == r % 16U, "m16 position kept", t.begin + r, in_wide);
                check(in_wide / 16U == covered / 16U + r / 16U, "m16 block order kept", t.begin + r, in_wide);
            }
            covered += t.count;
            ++k;
        }
        check(covered == w.count, "wide tile = its 64-row tiles", w.begin, covered);
    }
    check(k == narrow.size(), "every 64-row tile is inside a wide tile", static_cast<u32>(k), static_cast<u32>(narrow.size()));
    check(narrow_mblocks == wide_mblocks, "same m16 blocks (same MMAs issued)", narrow_mblocks, wide_mblocks);
}

// (output, part) pairs a block's warps own; part = matrix (project) or n8 block (down).
using Owned = std::set<std::tuple<u32, u32>>;
static Owned project_owned(u32 mb) {
    Owned owned;
    for (u32 warp = 0; warp < nm::block_threads(mb) / 32U; ++warp)
        for (u32 p = 0; p < nm::warp_parts(mb); ++p)
            for (u32 col = 0; col < 8U; ++col) {
                const auto key = std::make_tuple(nm::project_output(3U, nm::column_block(warp), col), nm::first_part(mb, warp) + p);
                check(owned.insert(key).second, "project output owned once", warp, col);
            }
    return owned;
}
static Owned down_owned(u32 mb) {
    Owned owned;
    for (u32 warp = 0; warp < nm::block_threads(mb) / 32U; ++warp)
        for (u32 p = 0; p < nm::warp_parts(mb); ++p)
            for (u32 col = 0; col < 8U; ++col) {
                const u32 n = nm::first_part(mb, warp) + p;
                const auto key = std::make_tuple(nm::down_output(5U, nm::column_block(warp), n, col), 0U);
                check(owned.insert(key).second, "down output owned once", warp, col);
            }
    return owned;
}

int main() {
    // Rule and capacity.
    check(!nm::wide(nm::kTileMBlocks) && !nm::wide(nm::kDecodeMBlocks) && nm::wide(nm::kWideMBlocks), "wide trait", 0, 0);
    check(nm::block_threads(nm::kWideMBlocks) == 512U && nm::min_blocks(nm::kWideMBlocks) == 1U, "wide block shape", 0, 0);
    check(nm::block_threads(nm::kTileMBlocks) == nm::kThreads && nm::min_blocks(nm::kTileMBlocks) == 2U, "tile block shape", 0, 0);
    check(nm::tile_capacity(nm::kWideMBlocks) == 128U && nm::tile_capacity(nm::kTileMBlocks) == 64U, "tile capacity", 0, 0);
    check(nm::wide_form(std::uint64_t{nm::kWideMinAvgRows} * 512U, 512U, nm::kWideMinAvgRows), "wide at the threshold", 0, 0);
    check(!nm::wide_form(std::uint64_t{nm::kWideMinAvgRows} * 512U - 1U, 512U, nm::kWideMinAvgRows), "tile form below the threshold", 0, 0);
    check(nm::wide_form(96ULL * 512U, 512U, 96U) && !nm::wide_form(96ULL * 512U - 1U, 512U, 96U), "planner parameter 96 rows", 0, 0);
    check(nm::kWideMinAvgRows == 80U, "bare on = the measured TILEWIDE of the RTX PRO (80 rows per expert)", 0, 0);
    check(!nm::wide_form(1ULL << 30, 512U, 0U), "off: never wide", 0, 0);
    check(nm::wide_form(16U * 10U, 512U, 1U) == false && nm::wide_form(512U, 512U, 1U), "forced threshold 1", 0, 0);
    // Output ownership: wide and tile forms own the same outputs, each once.
    check(project_owned(nm::kWideMBlocks) == project_owned(nm::kTileMBlocks), "project: wide owns the tile form's outputs", 0, 0);
    check(down_owned(nm::kWideMBlocks) == down_owned(nm::kTileMBlocks), "down: wide owns the tile form's outputs", 0, 0);
    check(project_owned(nm::kWideMBlocks).size() == 2U * nm::kProjectOutputs, "project block outputs", 0, 0);
    check(down_owned(nm::kWideMBlocks).size() == nm::kDownOutputs, "down block outputs", 0, 0);
    // Tile cuts: edge counts and random routings with the Qwen MoE shape (512 experts).
    check_tiles({0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 257, 640});
    std::mt19937 rng(20261119U);
    for (u32 trial = 0; trial < 200U; ++trial) {
        std::vector<u32> n(512);
        const u32 mean = 1U + trial % 320U;
        std::gamma_distribution<double> load(0.8 + 0.01 * (trial % 50U), 1.0);
        for (u32& x : n) x = static_cast<u32>(load(rng) * mean);
        check_tiles(n);
    }
    if (g_fail) { std::printf("native_mma_wide_host_test: %d failure(s)\n", g_fail); return 1; }
    std::printf("native_mma_wide_host_test: PASS\n");
    return 0;
}
