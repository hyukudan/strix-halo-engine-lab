#include "select_oracle.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace {
namespace oracle = select_oracle;
namespace sel = oracle::sel;

void require(bool value, const char* name) {
    if (value) return;
    std::fprintf(stderr, "k5c CPU oracle FAIL: %s\n", name);
    std::exit(EXIT_FAILURE);
}

void strided_score_packing() {
    static_assert(std::is_standard_layout_v<sel::StridedScoreArgs> &&
                  std::is_trivially_copyable_v<sel::StridedScoreArgs>);
    std::array<std::uint16_t, 4096> cache{};
    constexpr std::array row_sequence{0U, 0U, 1U};
    constexpr std::array origins{sel::ScoreSequence{0, 2047}, sel::ScoreSequence{2, 3}};
    constexpr std::array tiles{sel::ScoreTile{0, 2}, sel::ScoreTile{2, 1}};
    alignas(16) std::array<float, 3 * 3 * 128> query{};
    std::array<float, 3 * 3 * 19> scores{};
    constexpr sel::ScoreSpec spec{16, 3, 128, 128};
    sel::StridedScoreArgs args{{cache.data() + 7, 1981, 743, 137}, row_sequence.data(), origins.data(),
        query.data(), scores.data(), 2, 3, 19};
    const auto offset = args.cache.offset(args.row_sequence[2], args.layer, args.sequences[1].first_position);
    require(args.cache.base + offset == cache.data() + 3885 && args.cache.offset(0, 0, 1) == 137,
            "strided cache uses element strides and nonzero base, sequence, layer and position");

    sel::detail::StridedScoreGrid grid;
    using sel::detail::pack_strided_score_grid;
    constexpr auto warps = sel::ScoreForm::block_max, tiled = sel::ScoreForm::block_max_tiled;
    require(pack_strided_score_grid<warps>(args, spec, grid) && grid.x == 43 && grid.y == 1 && grid.z == 1 &&
            grid.threads == 128, "packed warp grid rounds up the final CTA");
    require(!pack_strided_score_grid<tiled>(args, spec, grid), "shared-key scoring requires a tile map");
    args.tiles = tiles.data();
    args.tile_count = 2;
    require(pack_strided_score_grid<tiled>(args, spec, grid) && grid.x == 2 && grid.y == 3 && grid.z == 19 &&
            grid.threads == 256, "shared-key grid uses tile, head and group axes");
    args.query = query.data() + 1;
    require(!pack_strided_score_grid<tiled>(args, spec, grid), "shared-key query loads require 16-byte alignment");
    args.query = query.data();
    args.max_groups = 65536;
    require(!pack_strided_score_grid<tiled>(args, spec, grid), "shared-key group grid overflow is rejected");
    args.row_count = 0xffffffffU;
    require(!pack_strided_score_grid<warps>(args, spec, grid), "packed grid overflow is rejected");
    args.cache.row_stride = 127;
    args.row_count = 3;
    require(!pack_strided_score_grid<warps>(args, spec, grid), "cache rows must contain the full key");
    args.max_groups = 16;
    require(pack_strided_score_grid<tiled>(args, spec, grid) && grid.x == 0, "dense scores skip the launch");
    std::puts("k5c CPU strided score packing PASS");
}
} // namespace

int main() {
    std::array<float, 128> key{};
    std::array<float, 256> query{};
    key.fill(1.0F);
    query[0] = 1.0F;
    query[64] = 2.0F;
    query[32] = 4.0F;
    query[128] = -1024.0F;
    require(oracle::relu_tree(query, key) == 7.0F / std::sqrt(128.0F), "QSA head ReLU and tree sum");
    require(oracle::weighted_fma(query, key, std::array{2.0F, -3.0F}) ==
            (2.0F * 0.17677669529663687F) * (7.0F * 0.088388347648318F), "DSA head ReLU and weighted fold");

    constexpr auto ninf = oracle::ninf;
    constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
    const std::array scores{-0.0F, 0.0F, 4.0F, ninf, 4.0F, nan};
    const auto masked = oracle::pick({3, 4, sel::KeyOrder::eligible_float}, 6, 2, scores);
    require(masked.ids == std::vector<unsigned>{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25},
            "canonical zero, lower-ID tie and tail");
    require(masked.scores[0] == ninf && std::bit_cast<unsigned>(masked.scores[1]) == 0 &&
            masked.scores[2] == ninf && masked.scores[4] == ninf, "mask only the selected scores");
    const auto short_pick = oracle::pick({3, 1, sel::KeyOrder::eligible_float}, 4, 0, std::array{nan, ninf, 5.0F, ninf});
    require(short_pick.ids == std::vector<unsigned>{0, 0, 2} && short_pick.scores[0] == ninf,
            "prefix padding when eligible scores are scarce");
    require(oracle::pick({6, 4, sel::KeyOrder::eligible_float}, 2, 1, {}).ids ==
            std::vector<unsigned>{0, 1, 2, 3, 4, 5, 6, 7, 8}, "dense route skips scores");
    require(oracle::pick({2, 1, sel::KeyOrder::raw_float}, 3, 0, std::array{-0.0F, 0.0F, 2.0F}).ids ==
            std::vector<unsigned>{1, 2}, "raw signed-zero order");
    require(oracle::pick({3, 1, sel::KeyOrder::canonical_float, true}, 5, 0,
                         std::array{4.0F, 4.0F, 4.0F, 9.0F, -9.0F}).ids ==
            std::vector<unsigned>{0, 3, 4}, "forced newest and canonical order");

    const std::array pool_scores{5.0F, 4.0F, 7.0F, 6.0F, 7.0F, ninf, 0.0F};
    const auto pool = oracle::pool(pool_scores, 2, 2);
    require(pool == std::vector<std::uint8_t>{0, 1, 0, 1}, "forced last candidate block and lower-ID tie");
    require(oracle::pick({4, 1, sel::KeyOrder::raw_float}, 7, 0, pool_scores, {}, {}, pool, 2).ids ==
            std::vector<unsigned>{0, 2, 3, 6}, "candidate user can still pick masked negative infinity");
    require(oracle::pool(std::array{ninf, ninf, ninf}, 2, 2) == std::vector<std::uint8_t>{0, 1},
            "candidate source only marks rankable maxima and forced last block");

    const std::array keys{5U, 8U, 7U, 8U, 4U, 7U};
    const auto expected = oracle::rank(keys, 2);
    const auto left = oracle::rank(std::span(keys).first(3), 2);
    const auto right = oracle::rank(std::span(keys).last(3), 2);
    const std::array global{left[0], left[1], right[0] + 3U, right[1] + 3U};
    const std::array survivors{keys[global[0]], keys[global[1]], keys[global[2]], keys[global[3]]};
    require(oracle::pick({2, 1, sel::KeyOrder::ordered}, 4, 0, {}, survivors, global).ids == expected,
            "shard candidates join to the full ordered result");
    strided_score_packing();
    std::puts("k5c CPU oracle PASS");
}
