#pragma once
#include "strix/moe_q4k_a8_contract.hpp"
#include <array>
namespace strix::moe_q4k_a8 {
inline constexpr std::string_view a8_down_common_identity =
    "silu-div-expf-times-up:g32-sym127-f32delta-rne:tiny-lt2^-119-zero1:"
    "blocks-mod4-ascending:tree-xor1-xor2:raw-slot-major:finite-failclosed";
inline constexpr std::string_view q51_a8_down_numerical_identity =
    "q51-a8g32-down-v1:s8xu5-i32:delta*(d*S+m*C):separate-f32:"
    "silu-div-expf-times-up:g32-sym127-f32delta-rne:tiny-lt2^-119-zero1:"
    "blocks-mod4-ascending:tree-xor1-xor2:raw-slot-major:finite-failclosed";
inline constexpr std::string_view q80_a8_down_numerical_identity =
    "q80-a8g32-down-v1:s8xs8-i32:delta*(d*S):separate-f32:"
    "silu-div-expf-times-up:g32-sym127-f32delta-rne:tiny-lt2^-119-zero1:"
    "blocks-mod4-ascending:tree-xor1-xor2:raw-slot-major:finite-failclosed";
struct DownShape { std::uint32_t hidden=0, intermediate=0, experts=0, top_k=0; };
struct DownRequest {
    bool enabled=false;
    std::string_view approved_numerical_identity;
    DownShape shape;
    std::uint32_t weight_type=7;
};
// Exact logical extents. All views, including read-only views, are disjoint.
// intermediates is K slots of [gate I, up I]; producer replaces ONLY gate with h.
// Caller owns all views until stream quiescence, including after enqueue errors.
struct DownBindings {
    DeviceBytes intermediates, weights, routes, codes, deltas, sums, rawdots, status, aggregate_status;
};
inline constexpr std::uint32_t down_invalid_route=1, down_nonfinite_activation=8,
    down_aggregate_error=16;
[[nodiscard]] std::string_view down_numerical_identity(std::uint32_t weight_type);
void validate_down(const DownRequest&,const DownBindings&);
// Independent original-byte CPU reference, no GPU or candidate output dependency.
[[nodiscard]] std::array<std::uint8_t,32> decode_down_q51_codes(std::span<const std::byte> block);
[[nodiscard]] QuantizedRow quantize_down_values(std::span<const float> activated);
[[nodiscard]] float evaluate_down_row(std::span<const std::byte> weights,
    const QuantizedRow&,std::uint32_t weight_type);
} // namespace strix::moe_q4k_a8
