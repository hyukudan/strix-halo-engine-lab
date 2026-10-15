#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace strix::moe_q4k_a8 {
inline constexpr std::uint32_t max_hidden = 2560, max_rows = 640, max_experts = 512, max_top_k = 10;
inline constexpr std::string_view numerical_identity =
    "strix-moe-q4k-a8g32-probe-v2:f32-div-rne:g32-sym127-zero1-finite-noflush:"
    "dot-s8xu8-i32:combine-separate-f32:pair-ascending-sb-tree4:writer-c0:wave32:proj-nonfinite=propagate-unqualified";
struct QuantizedRow {
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
};
struct Factors { std::uint32_t scale; std::uint32_t minimum; };
struct IntegerGroup { std::int32_t raw; std::int32_t sum; std::int32_t scaled; std::int32_t minimum; };
void require_f32_environment();
[[nodiscard]] QuantizedRow quantize_f32(std::span<const float> values);
[[nodiscard]] QuantizedRow quantize_bf16(std::span<const std::uint16_t> values);
[[nodiscard]] Factors decode_factors(std::span<const std::byte> packed12, std::uint32_t group);
[[nodiscard]] float half_bits_to_float(std::uint16_t bits);
[[nodiscard]] IntegerGroup integer_group(std::span<const std::uint8_t> q,
    std::span<const std::int8_t> a, Factors factors);
// Quantized reference with the versioned FP32 order, not legacy exact arithmetic.
// Qualified scope requires finite headers AND every combine intermediate finite.
[[nodiscard]] float evaluate_row(std::span<const std::byte> q4k, const QuantizedRow& row);

enum class InputKind { bf16, fp32 };
struct Shape { std::uint32_t hidden = 256, rows = 32, experts = 1, top_k = 1, logical_rows = 1; };
struct Request {
    bool enabled = false;
    std::string_view approved_numerical_identity;
    InputKind input = InputKind::bf16;
    Shape shape;
    std::uint32_t gate_type = 12, up_type = 12;
    bool fused_gate_up = false;
};
struct DeviceBytes { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
struct Bindings {
    DeviceBytes input, gate, up, routes, codes, deltas, sums, status, output;
};
// Pure host admission; no allocation, HIP call or buffer mutation.
void validate(const Request& request, const Bindings& bindings);
}  // namespace strix::moe_q4k_a8
