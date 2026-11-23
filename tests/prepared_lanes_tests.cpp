// The backend-neutral decode lane contract (strix/attn/prepared_lanes.hpp): launch blocks cover the lanes
// once, in caller order (= packed image rows), at most kMaxPreparedBf16Lanes per launch, each block knowing
// its first packed row; the block stays a small kernel argument; lanes must own distinct caches.
#include "strix/attn/prepared_lanes.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace lanes = strix::attn::lanes;

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using Lane = lanes::PreparedBf16Lane;

std::vector<Lane> make_lanes(std::vector<std::uint16_t>& arena, std::size_t count, std::size_t cache) {
    arena.assign(2U * count * cache, 0U);
    std::vector<Lane> out;
    for (std::size_t i = 0; i < count; ++i)
        out.push_back({arena.data() + 2U * i * cache, arena.data() + (2U * i + 1U) * cache,
                       static_cast<unsigned>(100U + 37U * i), 4096U, static_cast<unsigned>(i % 2U)});
    return out;
}

void blocks_cover_lanes_in_order() {
    CHECK(lanes::launch_count(0) == 0U);
    for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{15}, std::size_t{16}, std::size_t{17},
                          std::size_t{32}, std::size_t{33}}) {
        const auto launches = lanes::launch_count(n);
        CHECK(launches == (n + 15U) / 16U);
        std::size_t next = 0;
        for (unsigned i = 0; i < launches; ++i) {
            const auto r = lanes::launch_range(n, i);
            CHECK(r.first == next);
            CHECK(r.count >= 1U && r.count <= lanes::kMaxPreparedBf16Lanes);
            next += r.count;
        }
        CHECK(next == n);
        CHECK(lanes::launch_range(n, launches).count == 0U);
    }
}

void blocks_carry_descriptors_and_rows() {
    std::vector<std::uint16_t> arena;
    const auto all = make_lanes(arena, 19, 8);
    const std::span<const Lane> span(all);
    const auto first = lanes::make_block(span, 0);
    const auto second = lanes::make_block(span, 1);
    CHECK(first.count == 16U && first.first_row == 0U);
    CHECK(second.count == 3U && second.first_row == 16U);
    for (unsigned i = 0; i < first.count; ++i)
        CHECK(first.lane[i].keys == all[i].keys && first.lane[i].first_position == all[i].first_position);
    for (unsigned i = 0; i < second.count; ++i)
        CHECK(second.lane[i].values == all[16U + i].values && second.lane[i].wrapped == all[16U + i].wrapped);
    CHECK(lanes::make_block(span, 2).count == 0U);
    static_assert(sizeof(lanes::LaneBlock) <= 1024U, "lane block is a kernel argument");
}

void lanes_own_their_caches() {
    std::vector<std::uint16_t> arena;
    auto all = make_lanes(arena, 4, 8);
    CHECK(lanes::distinct_lane_caches(std::span<const Lane>(all)));
    CHECK(lanes::distinct_lane_caches(std::span<const Lane>()));
    auto shared = all;
    shared[3].keys = shared[1].keys;
    CHECK(!lanes::distinct_lane_caches(std::span<const Lane>(shared)));
    auto crossed = all;
    crossed[2].values = crossed[0].keys;
    CHECK(!lanes::distinct_lane_caches(std::span<const Lane>(crossed)));
    auto same = all;
    same[1].values = same[1].keys;
    CHECK(!lanes::distinct_lane_caches(std::span<const Lane>(same)));
    auto absent = all;
    absent[0].keys = nullptr;
    CHECK(!lanes::distinct_lane_caches(std::span<const Lane>(absent)));
}

}  // namespace

int main() {
    blocks_cover_lanes_in_order();
    blocks_carry_descriptors_and_rows();
    lanes_own_their_caches();
    if (failures != 0) {
        std::fprintf(stderr, "prepared lanes: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("prepared lanes: ok");
    return 0;
}
