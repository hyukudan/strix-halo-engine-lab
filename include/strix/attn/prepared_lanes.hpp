#pragma once

// Decode lanes of a multi-request step for the prepared BF16 attention: the ONE backend-neutral
// lane contract (no device headers). Every backend lowering (CUDA src/cuda/core, HIP
// src/attn_core/prepared_bf16_attention.hpp) and every decode attention form binds this
// descriptor and these launch blocks; only the kernel entry (lane = one grid dimension over the
// unchanged one-row body) is per backend.
//
// A lane is one query row of its own sequence: its own K/V cache, its own absolute position. The
// packed row images (queries, keys, values, outputs, relative logits, sinks) hold lane l at row l.
// One launch set serves up to kMaxPreparedBf16Lanes lanes; each block binds its lane and runs the
// single-row body with exactly the arguments of the one-lane call, so a lane's bits do not depend
// on the cohort width or on the other lanes (EXACT, batch invariant).

#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::attn::lanes {

struct PreparedBf16Lane {
    std::uint16_t* keys;    // [slot][KV head][dim] BF16, this lane's cache
    std::uint16_t* values;  // [slot][KV head][dim] BF16, this lane's cache
    unsigned first_position;
    unsigned cache_slots;
    unsigned wrapped;       // nonzero: slot = absolute position % cache_slots
};

// Bounded so that a launch block travels as a kernel argument (graph-capturable, no descriptor upload).
inline constexpr unsigned kMaxPreparedBf16Lanes = 16U;

[[nodiscard]] constexpr unsigned launch_count(std::size_t lanes) noexcept {
    return static_cast<unsigned>((lanes + kMaxPreparedBf16Lanes - 1U) / kMaxPreparedBf16Lanes);
}

// Lanes [first, first + count) of launch `index`, in caller order (= packed image rows).
struct LaunchRange {
    std::size_t first = 0;
    unsigned count = 0;
};

[[nodiscard]] constexpr LaunchRange launch_range(std::size_t lanes, unsigned index) noexcept {
    const auto first = std::size_t{index} * kMaxPreparedBf16Lanes;
    if (first >= lanes) return {lanes, 0U};
    const auto left = lanes - first;
    return {first, static_cast<unsigned>(left < kMaxPreparedBf16Lanes ? left : kMaxPreparedBf16Lanes)};
}

// One launch's lanes, passed by value: lane i of the block is packed image row first_row + i.
struct LaneBlock {
    PreparedBf16Lane lane[kMaxPreparedBf16Lanes];
    unsigned count;
    unsigned first_row;
};

[[nodiscard]] inline LaneBlock make_block(std::span<const PreparedBf16Lane> lanes, unsigned index) noexcept {
    LaneBlock block{};
    const auto range = launch_range(lanes.size(), index);
    for (unsigned i = 0; i < range.count; ++i) block.lane[i] = lanes[range.first + i];
    block.count = range.count;
    block.first_row = static_cast<unsigned>(range.first);
    return block;
}

// Every lane writes its own caches: two lanes on one cache would see each other's store before their
// flash, which the one-lane sequence orders differently.
[[nodiscard]] inline bool distinct_lane_caches(std::span<const PreparedBf16Lane> lanes) noexcept {
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        if (!lanes[i].keys || !lanes[i].values || lanes[i].keys == lanes[i].values) return false;
        for (std::size_t j = 0; j < i; ++j) {
            const auto* k = lanes[j].keys;
            const auto* v = lanes[j].values;
            if (lanes[i].keys == k || lanes[i].keys == v || lanes[i].values == k || lanes[i].values == v) return false;
        }
    }
    return true;
}

}  // namespace strix::attn::lanes
