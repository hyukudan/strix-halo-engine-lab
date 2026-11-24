// The backend-neutral decode lane contract (strix/attn/prepared_lanes.hpp): launch blocks cover the lanes
// once, in caller order (= packed image rows), at most kMaxPreparedBf16Lanes per launch, each block knowing
// its first packed row; the block stays a small kernel argument; lanes own distinct caches, or one sequence's
// consecutive rows share one cache (sequence_lane_caches, verify windows).
#include "strix/attn/prepared_lanes.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
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

// One sequence's verify window on one cache (the CUDA and HIP lanes admissions bind this rule): consecutive positions
// of adjacent lanes on a linear cache or on a ring of at least window - 1 + run slots; without a shared cache it is
// distinct_lane_caches.
void sequence_runs_share_one_cache() {
    std::vector<std::uint16_t> arena;
    const auto all = make_lanes(arena, 4, 8);
    for (const unsigned window : {0U, 1024U}) {
        CHECK(lanes::sequence_lane_caches(std::span<const Lane>(all), window) ==
              lanes::distinct_lane_caches(std::span<const Lane>(all)));
        auto shared = all;
        shared[3].keys = shared[1].keys;
        CHECK(!lanes::sequence_lane_caches(std::span<const Lane>(shared), window));
    }
    std::vector<std::uint16_t> cache(64U, 0U), other(64U, 0U);
    const auto run = [&](unsigned first, unsigned count, unsigned slots, unsigned wrapped) {
        std::vector<Lane> out;
        for (unsigned i = 0; i < count; ++i) out.push_back({cache.data(), cache.data() + 32, first + i, slots, wrapped});
        return out;
    };
    const auto linear = run(700U, 16U, 4096U, 0U);
    CHECK(lanes::sequence_lane_caches(std::span<const Lane>(linear), 0U));
    CHECK(!lanes::distinct_lane_caches(std::span<const Lane>(linear)));  // the cohort rule alone refuses a window
    auto gap = linear;
    gap[5].first_position += 1U;
    CHECK(!lanes::sequence_lane_caches(std::span<const Lane>(gap), 0U));
    auto reversed = run(700U, 2U, 4096U, 0U);
    std::swap(reversed[0], reversed[1]);
    CHECK(!lanes::sequence_lane_caches(std::span<const Lane>(reversed), 0U));
    // Ring: window - 1 + run slots hold the run, one fewer does not; a ring needs a window.
    CHECK(lanes::sequence_lane_caches(std::span<const Lane>(run(2000U, 3U, 1026U, 1U)), 1024U));
    CHECK(!lanes::sequence_lane_caches(std::span<const Lane>(run(2000U, 3U, 1025U, 1U)), 1024U));
    CHECK(!lanes::sequence_lane_caches(std::span<const Lane>(run(2000U, 2U, 4096U, 1U)), 0U));
    // A run, then another sequence's lane, then the first cache again: refused (not adjacent).
    auto split = run(10U, 2U, 4096U, 0U);
    split.push_back({other.data(), other.data() + 32, 5U, 4096U, 0U});
    CHECK(lanes::sequence_lane_caches(std::span<const Lane>(split), 0U));
    split.push_back({cache.data(), cache.data() + 32, 12U, 4096U, 0U});
    CHECK(!lanes::sequence_lane_caches(std::span<const Lane>(split), 0U));
}

}  // namespace

int main() {
    blocks_cover_lanes_in_order();
    blocks_carry_descriptors_and_rows();
    lanes_own_their_caches();
    sequence_runs_share_one_cache();
    if (failures != 0) {
        std::fprintf(stderr, "prepared lanes: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("prepared lanes: ok");
    return 0;
}
