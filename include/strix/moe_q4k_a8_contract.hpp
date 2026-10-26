#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

namespace strix::moe_q4k_a8 {
inline constexpr std::uint32_t max_hidden = 2560, max_rows = 640, max_experts = 512, max_top_k = 10;
inline constexpr std::string_view numerical_identity =
    "strix-moe-q4k-a8g32-probe-v2:f32-div-rne:g32-sym127-zero1-finite-noflush:"
    "dot-s8xu8-i32:combine-separate-f32:pair-ascending-sb-tree4:writer-c0:wave32:proj-nonfinite=propagate-unqualified";
inline constexpr std::string_view q5k_numerical_identity =
    "strix-moe-q5k-a8g32-probe-v1:f32-div-rne:g32-sym127-zero1-finite-noflush:"
    "dot-s8xu5qh-i32:combine-separate-f32:pair-ascending-sb-tree4:writer-c0:wave32:proj-nonfinite=propagate-unqualified";
// GSQ-RCO Q2_0 (ggml type 42: 64 weights, fp16 d, value (q - 1) * d). Lane c of 8 owns the A8 groups
// g = c (mod 8) of a row and adds delta_g * (d * float(I_g - S_g)) in ascending g, I_g = sum q * code
// (s8 x u2 dot4), S_g the group code sum; then the xor 1/2/4 tree. Opt-in (STRIX_GSQ_Q20_A8=1).
inline constexpr std::string_view q20_numerical_identity =
    "strix-moe-q20-a8g32-probe-v1:f32-div-rne:g32-sym127-zero1-finite-noflush:"
    "dot-s8xu2-i32:delta*(d*(I-S)):lanes8-groups-mod8-ascending:tree-xor1-xor2-xor4:writer-c0:wave32:proj-nonfinite=propagate-unqualified";
[[nodiscard]] inline bool q20_a8_enabled() {  // STRIX_GSQ_Q20_A8=1
    const char* value = std::getenv("STRIX_GSQ_Q20_A8");
    return value != nullptr && value[0] == '1' && value[1] == 0;
}
// GSQ-RCO routed experts in the fast prefill profile (fast-prefill-f16wmma-v1, "moe" component): every GSQ gate/up type
// (Q2_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) with a GSQ down type (Q2_0, IQ4_NL) runs the GSQ FP16 WMMA kernels unless
// STRIX_GSQ_PREFILL_F16=0, which keeps the A8 forms (iu8 WMMA == dot4 bytes for Q2_0, dot4 for the IQ types). The choice
// changes the prefill bytes, so it is part of the prefill arithmetic identity (gsq_fast_prefill_moe_name / _identity).
[[nodiscard]] inline bool gsq_prefill_f16_enabled() {
    const char* value = std::getenv("STRIX_GSQ_PREFILL_F16");
    return value == nullptr || value[0] != '0';
}
[[nodiscard]] inline std::string_view gsq_fast_prefill_moe_name() {
    return gsq_prefill_f16_enabled() ? "gsq-moe-prefill:f16wmma-v2:gate=q2_0,iq2_s,iq3_xxs,iq3_s,iq4_xs;down=q2_0,iq4_nl"
                                     : "gsq-moe-prefill:a8";
}
[[nodiscard]] inline std::uint64_t gsq_fast_prefill_moe_identity() {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const char c : gsq_fast_prefill_moe_name()) { hash ^= static_cast<unsigned char>(c); hash *= 0x100000001b3ULL; }
    return hash;
}
// GSQ IQ gate/up (IQ2_S 22, IQ3_XXS 18, IQ3_S 21, IQ4_XS 23; gsq_a8_groups.hpp): Q2_0 lanes and tree,
// group value delta * (s * float(I)) with signed int8 grid weights (IQ2_S: two 16-weight halves).
inline constexpr std::string_view iq2s_numerical_identity =
    "strix-moe-iq2s-a8g32-v1:f32-div-rne:g32-sym127-zero1-finite-noflush:dot-s8xs8-i32:"
    "delta*((s0*I0)+(s1*I1)):s=(d*(0.5+n))*0.25:lanes8-groups-mod8-ascending:tree-xor1-xor2-xor4:writer-c0:wave32";
inline constexpr std::string_view iq3xxs_numerical_identity =
    "strix-moe-iq3xxs-a8g32-v1:f32-div-rne:g32-sym127-zero1-finite-noflush:dot-s8xs8-i32:"
    "delta*(s*I):s=(d*(0.5+n))*0.5:lanes8-groups-mod8-ascending:tree-xor1-xor2-xor4:writer-c0:wave32";
inline constexpr std::string_view iq3s_numerical_identity =
    "strix-moe-iq3s-a8g32-v1:f32-div-rne:g32-sym127-zero1-finite-noflush:dot-s8xs8-i32:"
    "delta*(s*I):s=d*(1+2n):lanes8-groups-mod8-ascending:tree-xor1-xor2-xor4:writer-c0:wave32";
inline constexpr std::string_view iq4xs_numerical_identity =
    "strix-moe-iq4xs-a8g32-v1:f32-div-rne:g32-sym127-zero1-finite-noflush:dot-s8xs8-i32:"
    "delta*(s*I):s=d*(ls-32):lanes8-groups-mod8-ascending:tree-xor1-xor2-xor4:writer-c0:wave32";
[[nodiscard]] std::string_view numerical_identity_for_gate_type(std::uint32_t gate_type);
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

// Separate unsigned-five-bit contract; the Q4K oracle/admission remains closed.
[[nodiscard]] std::array<std::uint8_t,32> decode_q5k_codes(std::span<const std::byte> block,
    std::uint32_t group);
[[nodiscard]] IntegerGroup integer_group_q5k(std::span<const std::uint8_t> q,
    std::span<const std::int8_t> a, Factors factors);
[[nodiscard]] float evaluate_q5k_row(std::span<const std::byte> q5k, const QuantizedRow& row);
[[nodiscard]] float evaluate_q20_row(std::span<const std::byte> q20, const QuantizedRow& row);
// GSQ gate/up row oracle for type 42 (= evaluate_q20_row) or an IQ type: 8 lanes, groups mod 8, xor tree.
[[nodiscard]] float evaluate_gsq_row(std::uint32_t type, std::span<const std::byte> weights, const QuantizedRow& row);
// Signed int8 weights and scales of 32-weight group `group` of a GSQ IQ row (host; gsq_a8_groups.hpp).
struct GsqGroup { std::array<std::int8_t,32> v; float s0, s1; };
[[nodiscard]] GsqGroup gsq_iq_group(std::uint32_t type, std::span<const std::byte> row, std::uint32_t group);

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
