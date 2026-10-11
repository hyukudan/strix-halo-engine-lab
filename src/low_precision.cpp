#include "strix/low_precision.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::low_precision {
namespace {

std::int64_t add64(std::int64_t a, std::int64_t b) {
    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b)) {
        throw std::overflow_error("integer correction addition overflow");
    }
    return a + b;
}

std::int64_t mul64(std::int64_t a, std::int64_t b) {
    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    if (a != 0 && b != 0 &&
        ((a > 0 && b > 0 && a > hi / b) ||
         (a > 0 && b < 0 && b < lo / a) ||
         (a < 0 && b > 0 && a < lo / b) ||
         (a < 0 && b < 0 && a < hi / b))) {
        throw std::overflow_error("integer correction multiplication overflow");
    }
    return a * b;
}

std::int32_t narrow32(std::int64_t value) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("INT32 accumulator overflow; saturation disabled");
    }
    return static_cast<std::int32_t>(value);
}

std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("candidate work/byte accounting overflow");
    }
    return a * b;
}

std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) {
    return a / b + static_cast<std::uint64_t>(a % b != 0);
}

void validate_operand(const AffineOperand& operand) {
    if (!std::isfinite(operand.scale)) {
        throw std::invalid_argument("operand scale must be finite");
    }
    int lo = 0;
    int hi = 0;
    switch (operand.encoding) {
    case CodeEncoding::signed_i4: lo = -8; hi = 7; break;
    case CodeEncoding::unsigned_i4: lo = 0; hi = 15; break;
    case CodeEncoding::signed_i8: lo = -128; hi = 127; break;
    case CodeEncoding::iq4_index: lo = 0; hi = 0; break;
    default: throw std::invalid_argument("unknown operand interpretation");
    }
    if (operand.zero_point < lo || operand.zero_point > hi) {
        throw std::invalid_argument("zero point outside operand interpretation");
    }
}

CodeEncoding activation_codes(ActivationEncoding encoding) {
    switch (encoding) {
    case ActivationEncoding::symmetric_a8: return CodeEncoding::signed_i8;
    case ActivationEncoding::symmetric_a4: return CodeEncoding::signed_i4;
    default: throw std::invalid_argument("activation quantization requires A8 or A4");
    }
}

int nearest_even(double value) {
    const double lower = std::floor(value);
    const double fraction = value - lower;
    int result = static_cast<int>(lower);
    if (fraction > 0.5 || (fraction == 0.5 && result % 2 != 0)) {
        ++result;
    }
    return result;
}

void require_reference_environment() {
    if (!std::numeric_limits<double>::is_iec559 ||
        std::fegetround() != FE_TONEAREST) {
        throw std::invalid_argument("reference requires IEEE binary64 nearest-even");
    }
    // Detect common DAZ/FTZ configurations without changing the caller's mode.
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double two = 2.0;
    if (tiny * two == 0.0) {
        throw std::invalid_argument("reference requires gradual subnormals");
    }
}

} // namespace

int decode_code(CodeEncoding encoding, std::uint8_t code) {
    switch (encoding) {
    case CodeEncoding::signed_i8:
        return code < 128U ? static_cast<int>(code) : static_cast<int>(code) - 256;
    case CodeEncoding::signed_i4:
        if (code > 15U) { throw std::invalid_argument("invalid I4 code"); }
        return code < 8U ? static_cast<int>(code) : static_cast<int>(code) - 16;
    case CodeEncoding::unsigned_i4:
        if (code > 15U) { throw std::invalid_argument("invalid U4 code"); }
        return static_cast<int>(code);
    case CodeEncoding::iq4_index:
        if (code > 15U) { throw std::invalid_argument("invalid IQ4 index"); }
        return static_cast<int>(iq4_xs::codebook[code]);
    }
    throw std::invalid_argument("unknown operand interpretation");
}

std::int32_t correct_dot(std::int32_t raw, std::int64_t weight_sum,
                        std::int64_t activation_sum, std::uint64_t count,
                        int weight_zero, int activation_zero) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("group length exceeds signed wide correction");
    }
    if (weight_zero < -128 || weight_zero > 127 ||
        activation_zero < -128 || activation_zero > 127) {
        throw std::invalid_argument("correction zero point outside I8/U4 bounds");
    }
    auto result = add64(raw, mul64(-static_cast<std::int64_t>(weight_zero), activation_sum));
    result = add64(result, mul64(-static_cast<std::int64_t>(activation_zero), weight_sum));
    const auto zeros = mul64(weight_zero, activation_zero);
    result = add64(result, mul64(static_cast<std::int64_t>(count), zeros));
    return narrow32(result);
}

GroupDot affine_group_dot(AffineOperand weights, AffineOperand activations) {
    require_reference_environment();
    validate_operand(weights);
    validate_operand(activations);
    if (weights.codes.empty() || weights.codes.size() != activations.codes.size()) {
        throw std::invalid_argument("group operands need equal nonzero lengths");
    }
    if (activations.encoding == CodeEncoding::iq4_index) {
        throw std::invalid_argument("nonlinear indices are not activation integer codes");
    }
    GroupDot result;
    for (std::size_t i = 0; i < weights.codes.size(); ++i) {
        const auto w = decode_code(weights.encoding, weights.codes[i]);
        const auto a = decode_code(activations.encoding, activations.codes[i]);
        result.raw_dot = narrow32(add64(result.raw_dot, mul64(w, a)));
        result.weight_sum = add64(result.weight_sum, w);
        result.activation_sum = add64(result.activation_sum, a);
    }
    result.corrected_dot = correct_dot(result.raw_dot, result.weight_sum,
        result.activation_sum, weights.codes.size(), weights.zero_point, activations.zero_point);
    const double combined_scale = weights.scale * activations.scale;
    result.scaled_dot = combined_scale * static_cast<double>(result.corrected_dot);
    if (!std::isfinite(combined_scale) || !std::isfinite(result.scaled_dot)) {
        throw std::overflow_error("floating scale combination overflow");
    }
    return result;
}

double ordered_group_sum(std::span<const GroupDot> groups) {
    require_reference_environment();
    double result = 0.0;
    for (const auto& group : groups) {
        if (!std::isfinite(group.scaled_dot)) {
            throw std::invalid_argument("nonfinite group contribution");
        }
        result = result + group.scaled_dot;
        if (!std::isfinite(result)) { throw std::overflow_error("group sum overflow"); }
    }
    return result;
}

Int8Split split_int8(int value) {
    if (value < -128 || value > 127) { throw std::invalid_argument("not signed I8"); }
    int high = value / 16;
    if (value % 16 < 0) { --high; }
    return {high, value - 16 * high};
}

int reconstruct_int8(Int8Split value) {
    if (value.high < -8 || value.high > 7 || value.low < 0 || value.low > 15) {
        throw std::invalid_argument("invalid signed high/unsigned low digits");
    }
    return static_cast<int>(add64(mul64(16, value.high), value.low));
}

std::int32_t two_i4_dot(std::span<const std::int8_t> weights,
                       std::span<const std::int8_t> activations) {
    if (weights.empty() || weights.size() != activations.size()) {
        throw std::invalid_argument("two-I4 dot needs equal nonzero lengths");
    }
    std::int32_t high_dot = 0;
    std::int32_t low_dot = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] < -8 || weights[i] > 7) {
            throw std::invalid_argument("two-I4 weights must be signed I4");
        }
        const auto digits = split_int8(activations[i]);
        high_dot = narrow32(add64(high_dot, mul64(weights[i], digits.high)));
        low_dot = narrow32(add64(low_dot, mul64(weights[i], digits.low)));
    }
    return narrow32(add64(mul64(16, high_dot), low_dot));
}

std::uint64_t quantization_identity(ActivationEncoding encoding, std::size_t group_elements) {
    (void)activation_codes(encoding);
    if (group_elements == 0) { throw std::invalid_argument("quantization identity requires nonzero groups"); }
    std::uint64_t hash = 14695981039346656037ULL;
    const auto update = [&hash](std::uint8_t byte) {
        hash = (hash ^ byte) * 1099511628211ULL;
    };
    for (const auto byte : row_quantization_contract) { update(static_cast<std::uint8_t>(byte)); }
    update(0);
    for (const auto byte : name(encoding)) { update(static_cast<std::uint8_t>(byte)); }
    update(0);
    const auto group = static_cast<std::uint64_t>(group_elements);
    for (unsigned shift = 0; shift < 64; shift += 8) {
        update(static_cast<std::uint8_t>((group >> shift) & 255U));
    }
    return hash;
}

QuantizedRow quantize_row(std::span<const double> values,
                         ActivationEncoding encoding, std::size_t group_elements) {
    require_reference_environment();
    (void)activation_codes(encoding);
    if (values.empty() || group_elements == 0 || values.size() % group_elements != 0) {
        throw std::invalid_argument("row must contain exact nonempty activation groups");
    }
    const int maximum = encoding == ActivationEncoding::symmetric_a8 ? 127 : 7;
    const int modulus = encoding == ActivationEncoding::symmetric_a8 ? 256 : 16;
    QuantizedRow result{encoding, group_elements, {}, {}};
    result.codes.resize(values.size());
    for (std::size_t base = 0; base < values.size(); base += group_elements) {
        double magnitude = 0.0;
        for (std::size_t j = 0; j < group_elements; ++j) {
            const double value = values[base + j];
            if (!std::isfinite(value)) { throw std::invalid_argument("nonfinite row activation"); }
            magnitude = std::max(magnitude, std::abs(value));
        }
        const double scale = magnitude == 0.0 ? 1.0 : magnitude / static_cast<double>(maximum);
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::overflow_error("activation scale is not representable");
        }
        result.scales.push_back(scale);
        for (std::size_t j = 0; j < group_elements; ++j) {
            const double ratio = std::clamp(values[base + j] / scale,
                -static_cast<double>(maximum), static_cast<double>(maximum));
            const int quantized = nearest_even(ratio);
            result.codes[base + j] = static_cast<std::uint8_t>(
                quantized < 0 ? quantized + modulus : quantized);
        }
    }
    return result;
}

double iq4_xs_quantized_dot(std::span<const std::byte> weights,
                          const QuantizedRow& activations) {
    const auto encoding = activation_codes(activations.encoding);
    if (activations.codes.empty() || activations.group_elements != 32 ||
        activations.codes.size() % iq4_xs::block_elements != 0 ||
        activations.scales.size() != activations.codes.size() / 32 ||
        weights.size() != activations.codes.size() / iq4_xs::block_elements * iq4_xs::block_bytes) {
        throw std::invalid_argument("IQ4_XS dot requires exact storage blocks and 32-element scales");
    }
    const int maximum = activations.encoding == ActivationEncoding::symmetric_a8 ? 127 : 7;
    for (const auto code : activations.codes) {
        const auto value = decode_code(encoding, code);
        if (value < -maximum || value > maximum) {
            throw std::invalid_argument("activation code outside symmetric row contract");
        }
    }
    for (const auto scale : activations.scales) {
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::invalid_argument("activation row scales must be positive and finite");
        }
    }
    std::vector<GroupDot> groups;
    groups.reserve(activations.scales.size());
    for (std::size_t b = 0; b < activations.codes.size() / 256; ++b) {
        const std::span<const std::byte, iq4_xs::block_bytes> block(
            weights.data() + b * iq4_xs::block_bytes, iq4_xs::block_bytes);
        const auto global = static_cast<double>(
            iq4_xs::fp16_bits_to_float(iq4_xs::global_scale_bits(block)));
        for (std::size_t g = 0; g < 8; ++g) {
            std::array<std::uint8_t, 32> codes{};
            for (std::size_t j = 0; j < codes.size(); ++j) {
                const auto packed = std::to_integer<std::uint8_t>(block[8 + g * 16 + j % 16]);
                codes[j] = static_cast<std::uint8_t>(j < 16 ? packed & 15U : packed >> 4U);
            }
            const auto group = b * 8 + g;
            const auto local = static_cast<int>(iq4_xs::local_scale_code(block, g)) - 32;
            groups.push_back(affine_group_dot(
                {CodeEncoding::iq4_index, codes, global * static_cast<double>(local), 0},
                {encoding, std::span(activations.codes).subspan(group * 32, 32),
                 activations.scales[group], 0}));
        }
    }
    return ordered_group_sum(groups);
}

std::string_view name(StorageEncoding encoding) {
    switch (encoding) {
    case StorageEncoding::q4_0: return "q4_0";
    case StorageEncoding::iq4_nl: return "iq4_nl";
    case StorageEncoding::iq4_xs: return "iq4_xs";
    }
    throw std::invalid_argument("unsupported storage encoding");
}
std::string_view name(ActivationEncoding encoding) {
    switch (encoding) {
    case ActivationEncoding::float32_control: return "float32_control";
    case ActivationEncoding::symmetric_a8: return "symmetric_a8";
    case ActivationEncoding::symmetric_a4: return "symmetric_a4";
    }
    throw std::invalid_argument("unsupported activation encoding");
}
std::string_view name(ExecutionPath path) {
    switch (path) {
    case ExecutionPath::packed_float: return "packed_float";
    case ExecutionPath::integer8: return "integer8";
    case ExecutionPath::integer4: return "integer4";
    case ExecutionPath::two_integer4: return "two_integer4";
    }
    throw std::invalid_argument("unsupported execution path");
}

CandidatePlan plan_candidate(const CandidateRequest& request) {
    (void)name(request.storage);
    (void)name(request.activation);
    (void)name(request.path);
    if (request.actual_rows == 0 || request.input_features == 0 ||
        request.output_features == 0 || request.tile.rows == 0 ||
        request.tile.columns == 0 || request.tile.inner == 0) {
        throw std::invalid_argument("candidate dimensions and tile geometry must be positive");
    }
    CandidatePlan result;
    result.storage_block_elements = request.storage == StorageEncoding::iq4_xs ? 256 : 32;
    const std::uint64_t block_bytes = request.storage == StorageEncoding::iq4_xs ? 136 : 18;
    if (request.input_features % result.storage_block_elements != 0) {
        result.reason = "K is not divisible by the physical storage block; no synthetic tail";
        return result;
    }
    const bool floating = request.path == ExecutionPath::packed_float;
    if (floating != (request.activation == ActivationEncoding::float32_control)) {
        result.reason = "packed control requires floating activations; integer paths require A8/A4";
        return result;
    }
    if ((request.path == ExecutionPath::integer4 || request.path == ExecutionPath::two_integer4) &&
        request.storage != StorageEncoding::q4_0) {
        result.reason = "nonlinear codebook indices cannot be raw I4 operands; decode to I8";
        return result;
    }
    if ((request.path == ExecutionPath::integer4 && request.activation != ActivationEncoding::symmetric_a4) ||
        (request.path == ExecutionPath::two_integer4 && request.activation != ActivationEncoding::symmetric_a8)) {
        result.reason = "I4 requires A4; two-I4 reconstruction requires already quantized A8";
        return result;
    }
    result.eligible = true;
    result.reason = "reference-representable only; native and model qualification pending";
    const auto m = request.actual_rows;
    const auto k = request.input_features;
    const auto n = request.output_features;
    const auto inputs = multiply(m, k);
    const auto outputs = multiply(m, n);
    result.weight_bytes = multiply(multiply(k / result.storage_block_elements, block_bytes), n);
    result.ideal_weight_read_bytes = result.weight_bytes;
    result.serial_weight_read_bytes = multiply(result.weight_bytes, m);
    result.input_float_bytes = multiply(inputs, 4);
    result.output_float_bytes = multiply(outputs, 4);
    result.useful_products = multiply(outputs, k);
    result.scale_correction_groups = multiply(outputs, k / result.scale_group_elements);
    if (floating) {
        result.issued_products = result.useful_products;
        return result;
    }
    result.preparation_elements = inputs;
    result.prepared_activation_bytes = request.activation == ActivationEncoding::symmetric_a4 ? inputs / 2 : inputs;
    result.activation_scale_bytes = multiply(multiply(m, k / result.scale_group_elements), 8);
    const auto padded_m = multiply(ceil_div(m, request.tile.rows), request.tile.rows);
    const auto padded_n = multiply(ceil_div(n, request.tile.columns), request.tile.columns);
    const auto padded_k = multiply(ceil_div(k, request.tile.inner), request.tile.inner);
    result.matrix_tiles = multiply(multiply(padded_m / request.tile.rows,
        padded_n / request.tile.columns), padded_k / request.tile.inner);
    result.issued_products = multiply(multiply(padded_m, padded_n), padded_k);
    if (request.path == ExecutionPath::two_integer4) {
        result.issued_products = multiply(result.issued_products, 2);
        result.matrix_tiles = multiply(result.matrix_tiles, 2);
    }
    result.masked_output_elements = multiply(padded_m, padded_n) - outputs;
    const auto tile_elements = multiply(request.tile.columns, request.tile.inner);
    result.decoded_weight_tile_bytes = request.path == ExecutionPath::integer8 ?
        tile_elements : ceil_div(tile_elements, 2);
    return result;
}

} // namespace strix::low_precision
