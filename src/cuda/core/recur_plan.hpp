#pragma once

#include "strix/state/table.hpp"

#include <cstdint>
#include <limits>
#include <optional>

namespace strix::cuda::recur {

enum class Kind : std::uint8_t { gdn, kda };
enum class Execution : std::uint8_t { committed, speculative };
enum class Write : std::uint8_t { in_place, ring_slot, replay_input };

struct Dims {
    Kind kind;
    std::uint32_t key_heads, key_dim, value_heads, value_dim, taps;
};

struct StatePlan {
    Write write;
    std::uint64_t image_floats;
    std::uint32_t replay_floats;  // matrix: [prepared QKV, a, b]; history: [raw QKV]
};

struct Plan {
    Dims dims;
    std::uint32_t qk_width, value_width, channels;
    StatePlan matrix, history;
};

inline constexpr std::uint32_t max_steps = 64;

// These are the S1 fixed-state layouts [(value, key, head), (tap, channel)].
// Only the donor's FP32 GDN arithmetic and pure replay are admitted here.
[[nodiscard]] inline std::optional<Plan> make_plan(Dims dims, const state::Decl& matrix,
                                                   const state::Decl& history, Execution execution) noexcept {
    const auto warp_width = [](std::uint32_t width) { return width != 0 && width <= 128 && width % 32 == 0; };
    if (dims.kind != Kind::gdn || !dims.key_heads || !dims.value_heads || !warp_width(dims.key_dim) ||
        !warp_width(dims.value_dim) || dims.taps < 2 || dims.taps > std::numeric_limits<std::uint32_t>::max() - max_steps ||
        (execution != Execution::committed && execution != Execution::speculative))
        return std::nullopt;
    const auto qk = std::uint64_t{dims.key_heads} * dims.key_dim;
    const auto value = std::uint64_t{dims.value_heads} * dims.value_dim;
    const auto channels = 2 * qk + value;
    // replay_input_bytes is a uint32_t in S1, even for the widest retained row.
    if (channels + 2 * std::uint64_t{dims.value_heads} > std::numeric_limits<std::uint32_t>::max() / sizeof(float))
        return std::nullopt;

    const auto map = [execution](const state::Decl& decl, state::Kind kind, std::uint32_t x, std::uint32_t y,
                                 std::uint32_t z, std::uint64_t image, std::uint32_t replay)
        -> std::optional<StatePlan> {
        if (decl.kind != kind || decl.dtype != state::Dtype::f32 || decl.planes != 1 || decl.plane_bytes ||
            decl.shadow.planes || decl.checkpoint_interval || decl.dims[0] != x || decl.dims[1] != y || decl.dims[2] != z)
            return std::nullopt;
        if (decl.rollback == state::Rollback::snapshot && decl.replay_input_bytes == 0)
            return StatePlan{execution == Execution::committed ? Write::in_place : Write::ring_slot, image, 0};
        if (decl.rollback == state::Rollback::replay && decl.replay_input_bytes == replay * sizeof(float))
            return StatePlan{execution == Execution::committed ? Write::in_place : Write::replay_input, image, replay};
        return std::nullopt;
    };
    const auto c = static_cast<std::uint32_t>(channels);
    const auto matrix_plan = map(matrix, state::Kind::recurrent, dims.value_dim, dims.key_dim, dims.value_heads,
                                 value * dims.key_dim, c + 2 * dims.value_heads);
    const auto history_plan = map(history, state::Kind::taps, dims.taps - 1, c, 0, channels * (dims.taps - 1), c);
    if (!matrix_plan || !history_plan) return std::nullopt;
    return Plan{dims, static_cast<std::uint32_t>(qk), static_cast<std::uint32_t>(value), c, *matrix_plan, *history_plan};
}

struct StateView {
    float* images;
    std::uint64_t stride;  // floats between S1 state images
    std::uint32_t slots, published;
    float* input;          // S1 replay rows, densely packed
    std::uint32_t input_rows;
};

// slot_base starts at InstanceLayout::slot_offset(slot). The caller supplies
// the published image index and owns publication and synchronization.
[[nodiscard]] inline std::optional<StateView> bind_state(StatePlan plan, const state::InstanceLayout& layout,
                                                        float* slot_base, std::uint32_t published,
                                                        std::uint32_t max_pending) noexcept {
    constexpr auto width = sizeof(float);
    if (!slot_base || layout.master_bytes != plan.image_floats * width || layout.shadow_bytes ||
        !layout.row_stride || layout.row_stride < layout.master_bytes || layout.row_stride % width || layout.rows <= published ||
        layout.rows > std::numeric_limits<std::uint32_t>::max() || layout.rows > layout.slot_bytes / layout.row_stride)
        return std::nullopt;
    StateView view{slot_base, layout.row_stride / width, static_cast<std::uint32_t>(layout.rows), published, nullptr, 0};
    if (plan.write == Write::ring_slot &&
        (layout.rollback != state::Rollback::snapshot || !max_pending || layout.rows <= max_pending))
        return std::nullopt;
    if (plan.write == Write::replay_input) {
        if (layout.rollback != state::Rollback::replay || layout.rows != 3 || layout.checkpoint_interval ||
            layout.replay_input_bytes != plan.replay_floats * width || layout.replay_input_bytes == 0 ||
            layout.input_offset % width || layout.input_offset / layout.row_stride < layout.rows ||
            layout.input_offset >= layout.slot_bytes || !max_pending ||
            max_pending > (layout.slot_bytes - layout.input_offset) / layout.replay_input_bytes)
            return std::nullopt;
        view.input = slot_base + layout.input_offset / width;
        view.input_rows = max_pending;
    }
    return view;
}

// Matches S1 Txn::candidate for snapshots and pure replay. A zero-length
// accept keeps published; row is the zero-based LAST accepted row otherwise.
#if defined(__CUDACC__)
__host__ __device__
#endif
[[nodiscard]] constexpr std::uint32_t image_slot(Write write, StateView view, std::uint32_t row) noexcept {
    if (write == Write::in_place) return view.published;
    const auto advance = write == Write::ring_slot ? row + 1 : 1 + row % 2;
    const auto until_wrap = view.slots - view.published;
    return advance < until_wrap ? view.published + advance : advance - until_wrap;
}

}  // namespace strix::cuda::recur
