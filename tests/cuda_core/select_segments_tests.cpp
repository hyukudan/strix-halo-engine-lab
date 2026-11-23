#include "cuda/core/select_segments.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace {
namespace sel = strix::cuda::core::select;
using Map = sel::SegmentScoreMap<8>;
static_assert(std::is_trivially_copyable_v<Map>);
static_assert(std::is_trivially_copyable_v<sel::SegmentRows<Map>>);
static_assert(std::is_trivially_copyable_v<sel::SegmentRows<const Map*>>);
static_assert(Map::capacity > 2);

void require(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "K5B segment producer FAIL: %s\n", message);
    std::exit(EXIT_FAILURE);
}
} // namespace

int main() {
    constexpr sel::ScoreSpec spec{512, 4, 128, 4};
    constexpr sel::SegmentScoreShape shape{96, 70, 550};
    constexpr sel::RowStrides stride{8, 5, 6};
    std::array<float, 96 * 8> query{};
    std::array<float, 96 * 5> scores{};
    std::array<unsigned, 96 * 6> selected{};
    const std::array<float, 2> keys{};
    const sel::RowBases base{query.data(), nullptr, scores.data(), selected.data()};
    const std::array input{sel::ScoreSegmentInput{3, 2045, nullptr},
                           sel::ScoreSegmentInput{69, 2048, keys.data()},
                           sel::ScoreSegmentInput{5, 2100, keys.data() + 1}};
    Map map;
    require(sel::pack_score_segments(input, shape, spec, map), "pack over 32 rows");
    require(map.count == 3 && map.segments[0].wave_end == 0 && map.segments[1].first_row == 3 &&
            map.segments[1].groups == 529 && map.segments[1].wave_end == 306 &&
            map.segments[2].first_row == 72 && map.segments[2].wave_end == 340,
            "row origins and wave prefixes exclude below-threshold segment");

    const sel::SegmentRows eager{base, stride, map};
    const auto* replay_map = &map;
    const sel::SegmentRows replay{base, stride, replay_map};
    sel::detail::VisibleRow row{};
    require(eager.load(3, row) && row.position == 2048 && row.row.keys == keys.data() &&
            row.row.query == query.data() + 3U * stride.query && row.row.scores == scores.data() + 3U * stride.scores &&
            row.row.selected == selected.data() + 3U * stride.selected, "cohort origins and pitches");
    require(eager.load(72, row) && row.position == 2100 && row.row.keys == keys.data() + 1 &&
            row.score(4).visible == 525 && row.pick(4).tail == 1, "segment boundary and partial tail");
    require(!eager.load(77, row) && !replay.load(95, row), "unused capture rows are inactive");

    auto invalid = input;
    invalid[1].rows = 0;
    require(!sel::pack_score_segments(invalid, shape, spec, map), "empty segment rejected");
    invalid[1].rows = 71;
    require(!sel::pack_score_segments(invalid, shape, spec, map), "segment row overflow");
    invalid = input;
    invalid[1].keys = nullptr;
    require(!sel::pack_score_segments(invalid, shape, spec, map), "missing active pool");
    invalid[1].keys = keys.data();
    invalid[1].first_position = std::numeric_limits<std::uint64_t>::max();
    require(!sel::pack_score_segments(invalid, shape, spec, map), "position overflow");
    require(!sel::pack_score_segments(input, {76, 70, 550}, spec, map), "cohort row overflow");
    require(!sel::pack_score_segments(input, {96, 70, 528}, spec, map), "group overflow");
    require(!sel::pack_score_segments(input, shape, {512, 4, 128, 0}, map), "invalid ratio");
    require(!sel::pack_score_segments(std::array<sel::ScoreSegmentInput, 9>{}, shape, spec, map),
            "descriptor capacity overflow");
    require(!sel::pack_score_segments(std::array{sel::ScoreSegmentInput{65536, 1ULL << 27U, keys.data()}},
                                      {65536, 65536, 1U << 27U}, spec, map), "wave prefix overflow");
    require(replay.load(72, row) && row.position == 2100, "failed packing leaves descriptor unchanged");

    const std::array replacement{sel::ScoreSegmentInput{33, 2070, keys.data() + 1}};
    require(sel::pack_score_segments(replacement, shape, spec, map), "replace replay descriptor");
    require(replay.load(32, row) && row.position == 2102 && row.row.keys == keys.data() + 1 &&
            !replay.load(33, row) && eager.load(72, row), "replay sees update and eager owns its value");
    require(sel::pack_score_segments({}, shape, spec, map) && !replay.load(0, row), "idle replay");
    std::puts("K5B CPU segment packing and replay admission PASS");
}
