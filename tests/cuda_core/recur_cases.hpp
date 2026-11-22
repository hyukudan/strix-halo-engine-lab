#pragma once

#include "cuda/core/recur_plan.hpp"
#include <array>

namespace strix::cuda::recur::test {

inline constexpr Dims qwen27{Kind::gdn, 16, 128, 48, 128, 4};
inline constexpr Dims other_heads{Kind::gdn, 2, 128, 5, 128, 4};
inline constexpr Dims other_dims{Kind::gdn, 2, 64, 3, 96, 5};

inline std::array<state::Decl, 2> declarations(Dims d, state::Rollback matrix, state::Rollback taps) {
    const auto channels = 2 * d.key_heads * d.key_dim + d.value_heads * d.value_dim;
    constexpr auto f32 = static_cast<std::uint32_t>(sizeof(float));
    return {{
        {.id = "matrix", .kind = state::Kind::recurrent, .dtype = state::Dtype::f32,
         .dims = {d.value_dim, d.key_dim, d.value_heads}, .rollback = matrix,
         .replay_input_bytes = matrix == state::Rollback::replay ? (channels + 2 * d.value_heads) * f32 : 0U,
         .row_alignment = 256},
        {.id = "taps", .kind = state::Kind::taps, .dtype = state::Dtype::f32,
         .dims = {d.taps - 1, channels}, .rollback = taps,
         .replay_input_bytes = taps == state::Rollback::replay ? channels * f32 : 0U, .row_alignment = 1024},
    }};
}

state::Layout layout(const std::array<state::Decl, 2>& decls, std::uint32_t steps);

}  // namespace strix::cuda::recur::test
