#pragma once

#include <concepts>
#include <cstddef>
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

// The binding checks layout and epilogue hooks when its kernel bodies instantiate.
template <class P>
concept LaunchPolicy = PolicyShape<P> && std::is_trivially_copyable_v<P> && requires(const P& p) {
    { p.rows } -> std::same_as<const std::uint32_t&>;
    requires (P::kScaleLog2 >= -std::numeric_limits<float>::max() &&
              P::kScaleLog2 <= std::numeric_limits<float>::max());
};

struct GridLimits {
    unsigned max_kv_heads, max_tiles;
};

struct LaunchPlan {
    unsigned tiles, kv_heads;
};

template <PolicyShape P>
[[nodiscard]] constexpr std::optional<LaunchPlan> make_launch_plan(
    std::uint32_t rows, unsigned kv_heads, GridLimits limits) {
    if (rows == 0U) return LaunchPlan{};
    const unsigned tiles = (rows - 1U) / P::Rows + 1U;
    if (kv_heads == 0U || kv_heads > limits.max_kv_heads || tiles > limits.max_tiles) return std::nullopt;
    return LaunchPlan{tiles, kv_heads};
}

template <class P> struct Shape {
    static constexpr unsigned D = P::D, R = P::Rows, Threads = 32U * R, Chunk = 16U, Pitch = D + 8U;
    static constexpr unsigned Slot = Chunk * Pitch;                       // bf16 elements of one K or V chunk
    static constexpr unsigned Cap = P::Cap, UnionCap = R * Cap;           // union entries per tile (concatenation bound)
    static constexpr std::size_t RingBytes = 4U * Slot * sizeof(std::uint16_t);          // K, V of two chunks
    static constexpr std::size_t QBytes = R * 16U * Pitch * sizeof(std::uint16_t);       // Q staging (before the walk)
    static constexpr std::size_t FlashSmem = RingBytes > QBytes ? RingBytes : QBytes;
    static constexpr std::size_t UnionSmem = R * Cap * sizeof(std::uint32_t) + R * (Cap + 1U) * sizeof(std::uint16_t);
    static_assert(D % 16U == 0U && P::Heads <= 16U && R >= 1U && R <= 8U, "rowgroup shape");
    static_assert((Pitch * 2U) % 128U == 16U, "ldmatrix conflict-free pitch");
};
// Union workspace layout per call: tok [tiles][UnionCap] u32, mask [tiles][UnionCap] u8, count [tiles] u32.
template <class P> constexpr std::size_t workspace_bytes(unsigned tiles) {
    return ((std::size_t{tiles} * Shape<P>::UnionCap * (sizeof(std::uint32_t) + 1U) + 3U) & ~std::size_t{3}) + std::size_t{tiles} * sizeof(std::uint32_t);
}
struct Union {
    std::uint32_t* tok;
    std::uint8_t* mask;
    std::uint32_t* count;
};
template <class P> inline Union carve(void* workspace, unsigned tiles) {
    auto* base = static_cast<std::uint8_t*>(workspace);
    const std::size_t entries = std::size_t{tiles} * Shape<P>::UnionCap;
    const std::size_t counts = (entries * (sizeof(std::uint32_t) + 1U) + 3U) & ~std::size_t{3};
    return {reinterpret_cast<std::uint32_t*>(base), base + entries * sizeof(std::uint32_t), reinterpret_cast<std::uint32_t*>(base + counts)};
}

}  // namespace strix::cuda::core::rowgroup
