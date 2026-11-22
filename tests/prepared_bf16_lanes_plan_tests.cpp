// Host invariants of the decode-lane plan for the prepared BF16 flash (src/attn_core/prepared_bf16_lanes_plan.hpp):
// a lane's split count, workspace region and image rows depend only on the lane itself, never on the cohort width.
#include "attn_core/prepared_bf16_lanes_plan.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace dense = strix::attn_core::dense;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

// The fields prepared_bf16_lane_args reads and writes (PreparedBf16Args is a device header).
struct Bias {
    const float* logits;
    unsigned heads;
    unsigned extent;
};
struct Sinks {
    const float* logits;
    unsigned heads;
};
struct Args {
    const float* queries;
    float* output;
    const std::uint16_t* keys;
    const std::uint16_t* values;
    unsigned rows;
    unsigned first_position;
    unsigned window;
    unsigned cache_slots;
    bool wrapped;
    Bias relative_bias;
    Sinks sinks;
};

constexpr unsigned kHeads = 24U, kDim = 128U, kExtent = 64U;

void splits_follow_the_lane_position() {
    for (const unsigned window : {0U, 1024U}) {
        for (unsigned position = 0; position < 20000U; position += 37U) {
            const unsigned own = dense::lane_key_splits(position, window);
            check(own == dense::small_row_key_splits(position, 1U, window), "lane splits = one-row rule");
            check(own >= 1U && own <= dense::kKeySplitMaxChunks, "lane splits in range");
            // The same lane inside cohorts of any width keeps its own count; the grid depth is the cohort maximum.
            std::vector<dense::PreparedBf16Lane> cohort;
            for (unsigned width = 1; width <= dense::kMaxPreparedBf16Lanes; ++width) {
                cohort.push_back({nullptr, nullptr, width == 1U ? position : (position * width + 911U) % 30000U, 0, 0});
                check(dense::lane_key_splits(cohort.front().first_position, window) == own, "width-independent");
                check(dense::max_lane_key_splits(cohort, window) >= own, "grid covers the lane");
            }
        }
    }
    const std::array<dense::PreparedBf16Lane, 3> lanes{{{nullptr, nullptr, 10U, 0, 0},
                                                        {nullptr, nullptr, 5000U, 0, 0},
                                                        {nullptr, nullptr, 300U, 0, 0}}};
    check(dense::max_lane_key_splits(lanes, 0U) == dense::small_row_key_splits(5000U, 1U, 0U), "grid = max lane");
}

void workspace_regions_are_disjoint_and_complete() {
    const std::size_t per = dense::lane_workspace_floats(kHeads, kDim);
    check(per == std::size_t(dense::kKeySplitMaxChunks) * kHeads * (kDim + 2U), "lane region size");
    // The one-row layout of flash_body / combine_body at any split count stays inside the lane's region.
    for (unsigned splits = 2; splits <= dense::kKeySplitMaxChunks; ++splits) {
        const std::size_t last_acc = ((std::size_t(splits - 1U) * 1U + 0U) * kHeads + (kHeads - 1U)) * kDim + kDim;
        const std::size_t last_stat =
            std::size_t(splits) * kHeads * kDim + ((std::size_t(splits - 1U) * 1U + 0U) * kHeads + kHeads - 1U) * 2U + 2U;
        check(last_acc <= per && last_stat <= per, "one-row layout inside the lane region");
        check(dense::workspace_floats(1U, kHeads, kDim, splits) <= per, "one-lane workspace fits");
    }
    // The resident decoders size the Split-K workspace for min(rows, 16) rows at 16 chunks: that holds as many lanes.
    for (unsigned rows = 1; rows <= 40U; ++rows) {
        const auto ws = dense::key_split_workspace_floats(rows, kHeads, kDim);
        const unsigned expect = rows < dense::kMaxPreparedBf16Lanes ? rows : dense::kMaxPreparedBf16Lanes;
        check(dense::lanes_per_launch(ws, kHeads, kDim) == expect, "lanes per launch from the decoder workspace");
    }
    check(dense::lanes_per_launch(per - 1U, kHeads, kDim) == 0U, "a short workspace holds no lane");
}

void lane_arguments_address_their_own_rows() {
    std::vector<float> queries(4U * kHeads * kDim), output(queries.size());
    std::vector<float> bias(4U * kHeads * kExtent), sinks(4U * kHeads);
    std::array<std::uint16_t, 8> caches{};
    const Args base{queries.data(), output.data(), nullptr, nullptr, 7U, 99U, 1024U, 0U, false,
                    {bias.data(), kHeads, kExtent}, {sinks.data(), kHeads}};
    for (unsigned index = 0; index < 4U; ++index) {
        const dense::PreparedBf16Lane lane{&caches[2U * index], &caches[2U * index + 1U], 100U + index, 4096U, index & 1U};
        const auto a = dense::prepared_bf16_lane_args(base, lane, index, kHeads, kDim);
        check(a.queries == queries.data() + std::size_t(index) * kHeads * kDim, "query row");
        check(a.output == output.data() + std::size_t(index) * kHeads * kDim, "output row");
        check(a.relative_bias.logits == bias.data() + std::size_t(index) * kHeads * kExtent, "relative logits row");
        check(a.sinks.logits == sinks.data() + std::size_t(index) * kHeads, "sink row");
        check(a.keys == lane.keys && a.values == lane.values, "own cache");
        check(a.rows == 1U && a.first_position == lane.first_position && a.cache_slots == 4096U, "one row at own position");
        check(a.wrapped == ((index & 1U) != 0U) && a.window == 1024U, "own wrap, shared window");
    }
    Args no_bias = base;
    no_bias.relative_bias.logits = nullptr;
    no_bias.sinks.logits = nullptr;
    const auto a = dense::prepared_bf16_lane_args(no_bias, {&caches[0], &caches[1], 3U, 8U, 0U}, 2U, kHeads, kDim);
    check(a.relative_bias.logits == nullptr && a.sinks.logits == nullptr, "absent bias and sinks stay absent");
}

void lanes_write_distinct_caches() {
    std::array<std::uint16_t, 6> c{};
    const std::array<dense::PreparedBf16Lane, 2> ok{{{&c[0], &c[1], 1U, 8U, 0U}, {&c[2], &c[3], 1U, 8U, 0U}}};
    check(dense::distinct_lane_caches(ok), "distinct caches admitted");
    const std::array<dense::PreparedBf16Lane, 2> same{{{&c[0], &c[1], 1U, 8U, 0U}, {&c[0], &c[4], 2U, 8U, 0U}}};
    check(!dense::distinct_lane_caches(same), "shared key cache refused");
    const std::array<dense::PreparedBf16Lane, 2> cross{{{&c[0], &c[1], 1U, 8U, 0U}, {&c[2], &c[0], 2U, 8U, 0U}}};
    check(!dense::distinct_lane_caches(cross), "value cache = other key cache refused");
    const std::array<dense::PreparedBf16Lane, 1> null_cache{{{nullptr, &c[1], 1U, 8U, 0U}}};
    check(!dense::distinct_lane_caches(null_cache), "null cache refused");
}

}  // namespace

int main() {
    splits_follow_the_lane_position();
    workspace_regions_are_disjoint_and_complete();
    lane_arguments_address_their_own_rows();
    lanes_write_distinct_caches();
    if (failures != 0) {
        std::printf("prepared_bf16_lanes_plan: %d failures\n", failures);
        return 1;
    }
    std::printf("prepared_bf16_lanes_plan: PASS\n");
    return 0;
}
