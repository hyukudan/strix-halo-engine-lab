#pragma once

#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace strix::cuda::core::rowgroup {

// The padded ldmatrix pitch requires a 64-element dimension multiple. Each
// union prefix fits in uint16_t, and its row membership mask fits in uint8_t.
template <class P>
concept PolicyShape = requires {
    typename std::integral_constant<unsigned, P::D>;
    typename std::integral_constant<unsigned, P::Heads>;
    typename std::integral_constant<unsigned, P::Rows>;
    typename std::integral_constant<unsigned, P::Cap>;
} && (P::D > 0U && P::D % 64U == 0U && P::D <= std::numeric_limits<unsigned>::max() / 16U - 8U &&
      P::Heads >= 1U && P::Heads <= 16U && P::Rows >= 1U && P::Rows <= 8U &&
      P::Cap >= 1U && P::Cap <= std::numeric_limits<std::uint16_t>::max());

// The layout and epilogue hooks are checked when the CUDA bodies instantiate.
template <class P>
concept LaunchPolicy = PolicyShape<P> && std::is_trivially_copyable_v<P> && requires(const P& p) {
    { p.rows } -> std::same_as<const std::uint32_t&>;
    requires (P::kScaleLog2 >= -std::numeric_limits<float>::max() &&
              P::kScaleLog2 <= std::numeric_limits<float>::max());
};

inline constexpr unsigned kUnionThreads = 256;

struct LaunchPlan {
    unsigned tiles, kv_heads;
};

template <PolicyShape P>
[[nodiscard]] constexpr std::optional<LaunchPlan> make_launch_plan(std::uint32_t rows, unsigned kv_heads) {
    if (rows == 0U) return LaunchPlan{};
    const unsigned tiles = (rows - 1U) / P::Rows + 1U;
    // CUDA grid x/y limits apply to the flash grid (KV heads, tiles).
    if (kv_heads == 0U || kv_heads > 0x7fffffffU || tiles > 65535U) return std::nullopt;
    return LaunchPlan{tiles, kv_heads};
}

}  // namespace strix::cuda::core::rowgroup
