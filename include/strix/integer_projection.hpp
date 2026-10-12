#pragma once

#include "strix/low_precision.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strix::integer_projection {
enum class WeightEncoding { q4_0, iq4_xs };
enum class Instruction { iu8, iu4 };
struct Shape { std::size_t rows; std::size_t columns; std::size_t inner; };
// Deliberately bounded laboratory staging/result footprint, not a production
// tensor limit. Checked before allocation; no automatic chunking/requantization.
inline constexpr std::size_t max_working_bytes = 256U * 1024U * 1024U;
// Experimental staging only: each group retains the original nibble codes,
// lower 16 elements in low nibbles, upper 16 in high nibbles. No file format.
struct Weights {
    WeightEncoding encoding;
    std::size_t columns;
    std::size_t inner;
    std::vector<std::uint8_t> packed;
    std::vector<double> scales;
};
struct Activations {
    Shape shape;
    low_precision::ActivationEncoding encoding;
    std::vector<std::uint8_t> codes;
    std::vector<double> scales;
    std::vector<std::int32_t> sums;
};
// Checks all dimensions/products before allocation. K must be exact storage
// blocks; rows/columns may have 16-wide WMMA tails. No fabricated K blocks.
void validate_shape(Shape shape, WeightEncoding encoding);
[[nodiscard]] Weights stage_weights(WeightEncoding encoding, std::size_t columns,
    std::size_t inner, std::span<const std::byte> checkpoint);
[[nodiscard]] Activations prepare(Shape shape, low_precision::ActivationEncoding encoding,
    std::span<const double> input);
void validate(const Weights& weights, const Activations& activations, Instruction instruction);
[[nodiscard]] std::string numerical_identity(WeightEncoding weights,
    low_precision::ActivationEncoding activations, Instruction instruction);
[[nodiscard]] int weight_code(const Weights& weights, std::size_t column, std::size_t k);
[[nodiscard]] int activation_code(const Activations& activations, std::size_t row, std::size_t k);
// Independent direct integer formula (qw-zw)*qa, binary64 scale combination in
// increasing group order; this reference does not simulate WMMA instructions.
struct Result { std::vector<double> output; std::vector<std::int32_t> groups; };
[[nodiscard]] Result reference(const Weights& weights, const Activations& activations,
    Instruction instruction);
// Scalar fragment packing model for one K16 panel. Interpreted padding is zero,
// including nonlinear codebooks without a zero entry. Native ABI remains a
// hardware gate: a successful host simulation is not native execution proof.
using Tile = std::array<std::int32_t, 256>;
[[nodiscard]] Tile fragment_panel(const Weights& weights, const Activations& activations,
    Instruction instruction, std::size_t row_base, std::size_t column_base, std::size_t k_base);
} // namespace strix::integer_projection
