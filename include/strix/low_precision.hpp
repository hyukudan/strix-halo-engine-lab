#pragma once

#include "strix/iq4_xs.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::low_precision {

// These are operand interpretations, not names for checkpoint formats.
enum class CodeEncoding { signed_i4, unsigned_i4, signed_i8, iq4_index };
[[nodiscard]] int decode_code(CodeEncoding encoding, std::uint8_t code);

struct AffineOperand {
    CodeEncoding encoding;
    std::span<const std::uint8_t> codes;
    double scale;
    int zero_point;
};
struct GroupDot {
    std::int32_t raw_dot = 0;
    std::int64_t weight_sum = 0;
    std::int64_t activation_sum = 0;
    std::int32_t corrected_dot = 0;
    double scaled_dot = 0;
};
// No saturation. Raw accumulation and final narrowing fail on INT32 overflow;
// every correction multiply/add is checked in INT64, including intermediates.
[[nodiscard]] std::int32_t correct_dot(
    std::int32_t raw, std::int64_t weight_sum, std::int64_t activation_sum,
    std::uint64_t count, int weight_zero, int activation_zero);
[[nodiscard]] GroupDot affine_group_dot(AffineOperand weights,
                                       AffineOperand activations);
[[nodiscard]] double ordered_group_sum(std::span<const GroupDot> groups);

struct Int8Split { int high; int low; };
[[nodiscard]] Int8Split split_int8(int value);
[[nodiscard]] int reconstruct_int8(Int8Split value);
// Signed I4 weights times signed I8 activations, using signed high and unsigned
// low activation digits. Reconstruction multiplies in INT64, never shifts.
[[nodiscard]] std::int32_t two_i4_dot(std::span<const std::int8_t> weights,
                                    std::span<const std::int8_t> activations);

enum class ActivationEncoding { float32_control, symmetric_a8, symmetric_a4 };
inline constexpr std::string_view row_quantization_contract =
    "strix-row-quant-v1:f64:nearest-even:divide:no-ftz:finite-only:"
    "symmetric-127-or-7:zero-scale-one:exact-groups:ordered-groups";
// Stable FNV-1a identifier over the versioned contract, activation name and
// group size (little-endian U64). This identifies semantics, not input content.
[[nodiscard]] std::uint64_t quantization_identity(ActivationEncoding encoding,
                                                std::size_t group_elements);
struct QuantizedRow {
    ActivationEncoding encoding = ActivationEncoding::symmetric_a8;
    std::size_t group_elements = 32;
    std::vector<std::uint8_t> codes;
    std::vector<double> scales;
};
// IEEE binary64 reference, FE_TONEAREST only, gradual subnormals. A scale that
// underflows to zero is rejected. No batch/slot/request identity is an input.
[[nodiscard]] QuantizedRow quantize_row(std::span<const double> values,
                                      ActivationEncoding encoding,
                                      std::size_t group_elements = 32);
// A bounded reference for exact IQ4_XS nibble-to-I8 interpretation followed by
// quantized activation products. Activation groups must match 32-element scales.
// Floating combination is binary64 and is a distinct experimental identity.
[[nodiscard]] double iq4_xs_quantized_dot(std::span<const std::byte> weights,
                                        const QuantizedRow& activations);

enum class StorageEncoding { q4_0, iq4_nl, iq4_xs };
enum class ExecutionPath { packed_float, integer8, integer4, two_integer4 };
[[nodiscard]] std::string_view name(StorageEncoding encoding);
[[nodiscard]] std::string_view name(ActivationEncoding encoding);
[[nodiscard]] std::string_view name(ExecutionPath path);
struct TileGeometry {
    std::uint64_t rows = 16;
    std::uint64_t columns = 16;
    std::uint64_t inner = 16;
};
struct CandidateRequest {
    StorageEncoding storage = StorageEncoding::iq4_xs;
    ActivationEncoding activation = ActivationEncoding::symmetric_a8;
    ExecutionPath path = ExecutionPath::integer8;
    std::uint64_t actual_rows = 1; // Per operation/expert, never request count.
    std::uint64_t input_features = 2560;
    std::uint64_t output_features = 2560;
    TileGeometry tile; // Accounting hypothesis, not a verified WMMA fragment ABI.
};
struct CandidatePlan {
    bool eligible = false;
    std::string reason;
    std::uint64_t storage_block_elements = 0;
    std::uint64_t scale_group_elements = 32;
    std::uint64_t weight_bytes = 0;
    std::uint64_t serial_weight_read_bytes = 0;
    std::uint64_t ideal_weight_read_bytes = 0;
    std::uint64_t input_float_bytes = 0;
    std::uint64_t prepared_activation_bytes = 0;
    std::uint64_t activation_scale_bytes = 0; // Binary64 reference scales.
    std::uint64_t output_float_bytes = 0;
    std::uint64_t preparation_elements = 0;
    std::uint64_t scale_correction_groups = 0;
    std::uint64_t useful_products = 0;
    std::uint64_t issued_products = 0;
    std::uint64_t matrix_tiles = 0;
    std::uint64_t masked_output_elements = 0;
    std::uint64_t decoded_weight_tile_bytes = 0;
};
// Finite representability/work accounting only. No performance ranking or
// qualification. Invalid geometry and accounting overflow throw.
[[nodiscard]] CandidatePlan plan_candidate(const CandidateRequest& request);

} // namespace strix::low_precision
