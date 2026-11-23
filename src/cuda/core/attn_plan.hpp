#pragma once

#include "strix/attn/fp8_plan.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace strix::cuda::core::attn {

// Reduction order is part of the identity. The prepared MMA walk has one
// running softmax state; one split is required to preserve BF16 P rounding.
enum class DecodeOrder : std::uint8_t { global_max, online_bf16, online_e4m3 };
enum class KvLayout : std::uint8_t { head_major, slot_major, ring };
enum class OutputGate : std::uint8_t { none, sigmoid_interleaved, sigmoid_bf16, sigmoid_strided };

struct DecodeSpec {
    std::uint32_t head_dim, heads, kv_heads;
    std::uint32_t rows, max_context, cache_slots;
    std::uint32_t window = 0;  // Causal. Zero is the full prefix.
    std::uint32_t relative_extent = 0;
    float scale = 1.0F;
    DecodeOrder order = DecodeOrder::global_max;
    KvLayout layout = KvLayout::head_major;
    OutputGate gate = OutputGate::none;
    bool sinks = false;
};

inline constexpr unsigned kSplitKeys = 256;
inline constexpr unsigned kFlashKeys = 32;
inline constexpr unsigned kMaxGroup = 8;

// The native split ABI uses the same extent for the K/V plane and the score
// workspace. Its key walk always starts at absolute key zero.
[[nodiscard]] inline constexpr bool uses_native_split(const DecodeSpec& s) {
    return s.head_dim == 256 && s.layout == KvLayout::head_major &&
           s.window == 0 && s.max_context == s.cache_slots;
}

struct DecodePlan {
    DecodeSpec spec;
    std::uint32_t group;
    std::uint32_t dim_class;  // 0 selects the runtime-dimension scalar walk.
    std::uint32_t splits;
    std::size_t score_floats, stat_floats, scratch_floats;
};

[[nodiscard]] inline std::optional<DecodePlan> make_decode_plan(const DecodeSpec& s) {
    constexpr auto max_context = std::numeric_limits<std::uint32_t>::max() - kSplitKeys;
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    constexpr auto max_bf16 = std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t);
    if (s.head_dim == 0 || s.head_dim > 512 || s.heads == 0 || s.kv_heads == 0 ||
        s.heads % s.kv_heads != 0 || s.heads / s.kv_heads > kMaxGroup ||
        s.rows == 0 || s.rows > 32 || s.heads > 65535U ||
        s.max_context < s.rows || s.max_context > max_context || s.cache_slots == 0 ||
        !std::isfinite(s.scale) ||
        s.heads > max_floats / s.rows / (2U * s.head_dim) ||
        s.kv_heads > max_bf16 / s.cache_slots / s.head_dim ||
        (s.relative_extent != 0 && s.heads > max_floats / s.rows / s.relative_extent)) return std::nullopt;
    switch (s.layout) {
        case KvLayout::head_major:
        case KvLayout::slot_major:
            if (s.cache_slots < s.max_context) return std::nullopt;
            break;
        case KvLayout::ring:
            // Query rows may share the cache. Preserve the first row's live
            // tail as well as every newly appended key.
            if (s.window == 0 || s.cache_slots <
                (s.window > s.max_context - (s.rows - 1U) ? s.max_context : s.window + s.rows - 1U))
                return std::nullopt;
            break;
        default: return std::nullopt;
    }

    std::uint32_t dim_class = 0;
    switch (s.head_dim) {
        case 64: case 96: case 128: case 192: case 256: case 512: dim_class = s.head_dim; break;
        default: break;
    }
    const auto group = s.heads / s.kv_heads;
    if (s.order == DecodeOrder::online_bf16) {
        if (s.layout == KvLayout::head_major || dim_class == 0 ||
            (s.gate != OutputGate::none && s.gate != OutputGate::sigmoid_bf16))
            return std::nullopt;
        return DecodePlan{s, group, dim_class, 1, 0, 0, 0};
    }
    if (s.order == DecodeOrder::online_e4m3) {
        if (!uses_native_split(s) || s.sinks || s.relative_extent != 0 ||
            (s.gate != OutputGate::none && s.gate != OutputGate::sigmoid_interleaved &&
             s.gate != OutputGate::sigmoid_strided)) return std::nullopt;
        const auto fp8 = strix::attn::fp8::make_plan(s.head_dim, s.heads, s.kv_heads, s.cache_slots);
        if (!fp8 || fp8->scratch_floats(s.rows) == 0) return std::nullopt;
        return DecodePlan{s, group, dim_class, fp8->splits, 0,
                          std::size_t{s.rows} * s.heads * fp8->splits, fp8->scratch_floats(s.rows)};
    }
    if (s.order != DecodeOrder::global_max || s.sinks || s.relative_extent != 0 ||
        (s.gate != OutputGate::none && s.gate != OutputGate::sigmoid_interleaved &&
         !(s.gate == OutputGate::sigmoid_strided && uses_native_split(s)))) return std::nullopt;

    const auto splits = (s.max_context + kSplitKeys - 1U) / kSplitKeys;
    const auto row_heads = std::size_t{s.rows} * s.heads;
    const auto floats_per_head = std::uint64_t{s.max_context} + std::uint64_t{splits} * (s.head_dim + 2U);
    if (row_heads > max_floats / floats_per_head) return std::nullopt;
    return DecodePlan{s, group, dim_class, splits,
                      row_heads * s.max_context, row_heads * splits,
                      row_heads * static_cast<std::size_t>(floats_per_head)};
}

}  // namespace strix::cuda::core::attn
