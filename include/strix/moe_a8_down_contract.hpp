#pragma once
#include "strix/moe/activation_policy.hpp"
#include "strix/moe_q4k_a8_contract.hpp"
#include <array>
namespace strix::moe_q4k_a8 {
// K-quant down operand classes (C1 form down_rawdots_kq; Q4_K / Q6_K routed down). Lane layout and reduction of
// `project`: per 256-block, pair p of g32 groups (2p, 2p+1), c_g per group, paired = c_2p + c_2p+1, slot p accumulates
// blocks ascending, result (p0+p1)+(p2+p3). Q4_K c_g is the gate/up term; Q6_K is g4-a8-v1's group term.
inline constexpr std::string_view q4k_a8_down_numerical_identity =
    "q4k-a8g32-down-v1:s8xu4-i32:delta*((d*(sc*I))-(dmin*(m*S))):pair-ascending-sb-tree4:"
    "silu-div-expf-times-up:g32-sym127-f32delta-rne:tiny-lt2^-119-zero1:raw-slot-major:finite-failclosed";
inline constexpr std::string_view q6k_a8_down_numerical_identity =
    "q6k-a8g32-down-v1:s8xs8-i32:delta*(d*(s0*I0+s1*I1)):pair-ascending-sb-tree4:"
    "silu-div-expf-times-up:g32-sym127-f32delta-rne:tiny-lt2^-119-zero1:raw-slot-major:finite-failclosed";
// Activation clamp trait (strix::moe::swiglu_clamp, gate <= L, -L <= up <= L) applied before silu(g) * u; +infinity =
// none (every incumbent identity). A finite limit is part of the arithmetic identity of the producer.
inline constexpr std::string_view swiglu_clamp_identity = "swiglu-clamp-v1:g=min(g,L):u=clamp(u,-L,L):nan-propagates:raw-nonfinite-failclosed";
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
inline constexpr std::string_view q20_a8_down_numerical_identity =
    "q20-a8g32-down-v1:s8xu2-i32:delta*(d*(I-S)):separate-f32:"
    "silu-div-expf-times-up:g32-sym127-f32delta-rne:tiny-lt2^-119-zero1:"
    "blocks-mod4-ascending:tree-xor1-xor2:raw-slot-major:finite-failclosed";
inline constexpr std::string_view iq4nl_a8_down_numerical_identity =
    "iq4nl-a8g32-down-v1:s8xs8-i32:delta*(d*I):separate-f32:"
    "silu-div-expf-times-up:g32-sym127-f32delta-rne:tiny-lt2^-119-zero1:"
    "blocks-mod4-ascending:tree-xor1-xor2:raw-slot-major:finite-failclosed";
// Bytes of one down row of `intermediate` inputs for weight type 7 (Q5_1), 8 (Q8_0) or 42 (Q2_0).
[[nodiscard]] std::uint64_t down_row_bytes(std::uint32_t weight_type, std::uint32_t intermediate);
struct DownShape { std::uint32_t hidden=0, intermediate=0, experts=0, top_k=0; };
struct DownRequest {
    bool enabled=false;
    std::string_view approved_numerical_identity;
    DownShape shape;
    std::uint32_t weight_type=7;
    float swiglu_limit=moe::kSwigluNoClamp;  // finite > 0: clamped producer (e.g. 10)
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
// Union rows form (strix/moe/expert_union.hpp) of the K-quant class (weight type 12 Q4_K / 14 Q6_K): rows = R (1..8).
// intermediates / routes / codes / deltas / sums / rawdots are R contiguous C1 extents (each row slice passes validate_down);
// weights / status / aggregate_status are shared; plan is the expert_union plan built by the probe (read only here).
void validate_down_union(const DownRequest&,const DownBindings&,std::uint32_t rows,DeviceBytes plan);
// Independent original-byte CPU reference, no GPU or candidate output dependency.
[[nodiscard]] std::array<std::uint8_t,32> decode_down_q51_codes(std::span<const std::byte> block);
[[nodiscard]] QuantizedRow quantize_down_values(std::span<const float> activated);
[[nodiscard]] float evaluate_down_row(std::span<const std::byte> weights,
    const QuantizedRow&,std::uint32_t weight_type);
} // namespace strix::moe_q4k_a8
