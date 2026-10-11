#include "strix/low_precision.hpp"

#include <array>
#include <charconv>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace lp = strix::low_precision;
namespace {

std::uint64_t number(std::string_view text) {
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || result == 0) {
        throw std::invalid_argument("dimensions must be positive decimal integers");
    }
    return result;
}

void print_candidate(lp::CandidateRequest request, bool& first) {
    const auto p = lp::plan_candidate(request);
    if (!first) { std::cout << ",\n"; }
    first = false;
    std::cout << "  {\"storage\":\"" << lp::name(request.storage)
        << "\",\"activation\":\"" << lp::name(request.activation)
        << "\",\"path\":\"" << lp::name(request.path)
        << "\",\"actual_rows\":" << request.actual_rows
        << ",\"K\":" << request.input_features << ",\"N\":" << request.output_features
        << ",\"eligible\":" << (p.eligible ? "true" : "false")
        << ",\"reason\":\"" << p.reason << "\",\"storage_block_elements\":" << p.storage_block_elements
        << ",\"scale_group_elements\":" << p.scale_group_elements
        << ",\"tile\":[" << request.tile.rows << ',' << request.tile.columns << ',' << request.tile.inner << ']'
        << ",\"weight_bytes\":" << p.weight_bytes
        << ",\"ideal_weight_read_bytes\":" << p.ideal_weight_read_bytes
        << ",\"serial_weight_read_bytes\":" << p.serial_weight_read_bytes
        << ",\"input_float_bytes\":" << p.input_float_bytes
        << ",\"prepared_activation_bytes\":" << p.prepared_activation_bytes
        << ",\"activation_scale_bytes\":" << p.activation_scale_bytes
        << ",\"output_float_bytes\":" << p.output_float_bytes
        << ",\"preparation_elements\":" << p.preparation_elements
        << ",\"scale_correction_groups\":" << p.scale_correction_groups
        << ",\"useful_products\":" << p.useful_products
        << ",\"issued_products\":" << p.issued_products
        << ",\"matrix_tiles\":" << p.matrix_tiles
        << ",\"masked_output_elements\":" << p.masked_output_elements
        << ",\"decoded_weight_tile_bytes\":" << p.decoded_weight_tile_bytes << '}';
}

void candidates(int argc, char** argv) {
    lp::CandidateRequest r;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (++i == argc) { throw std::invalid_argument("option requires value"); }
        const std::string_view value(argv[i]);
        if (option == "--rows") { r.actual_rows = number(value); }
        else if (option == "--k") { r.input_features = number(value); }
        else if (option == "--n") { r.output_features = number(value); }
        else if (option == "--tile-m") { r.tile.rows = number(value); }
        else if (option == "--tile-n") { r.tile.columns = number(value); }
        else if (option == "--tile-k") { r.tile.inner = number(value); }
        else if (option == "--encoding") {
            if (value == "q4_0") { r.storage = lp::StorageEncoding::q4_0; }
            else if (value == "iq4_nl") { r.storage = lp::StorageEncoding::iq4_nl; }
            else if (value == "iq4_xs") { r.storage = lp::StorageEncoding::iq4_xs; }
            else { throw std::invalid_argument("encoding must be q4_0, iq4_nl or iq4_xs"); }
        } else { throw std::invalid_argument("unknown candidate option"); }
    }
    // Evaluate before emitting JSON, so an overflow does not emit a partial result.
    constexpr std::array paths{lp::ExecutionPath::packed_float, lp::ExecutionPath::integer8,
        lp::ExecutionPath::integer8, lp::ExecutionPath::integer4, lp::ExecutionPath::two_integer4};
    constexpr std::array activations{lp::ActivationEncoding::float32_control, lp::ActivationEncoding::symmetric_a8,
        lp::ActivationEncoding::symmetric_a4, lp::ActivationEncoding::symmetric_a4, lp::ActivationEncoding::symmetric_a8};
    for (std::size_t i = 0; i < paths.size(); ++i) {
        r.path = paths[i]; r.activation = activations[i]; (void)lp::plan_candidate(r);
    }
    std::cout << "{\"schema\":\"strix-low-precision-candidates-v1\",\"qualification\":\"host-reference-only\","
        "\"tile_contract\":\"caller-supplied-accounting-hypothesis\",\"candidates\":[\n";
    bool first = true;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        r.path = paths[i]; r.activation = activations[i]; print_candidate(r, first);
    }
    std::cout << "\n]}\n";
}

void vectors() {
    const std::array<double, 8> input{-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127, -127};
    const std::array<std::uint8_t, 8> weights{0, 1, 7, 8, 9, 14, 15, 3};
    const auto a8 = lp::quantize_row(input, lp::ActivationEncoding::symmetric_a8, 8);
    const auto a4 = lp::quantize_row(input, lp::ActivationEncoding::symmetric_a4, 8);
    const auto d8 = lp::affine_group_dot({lp::CodeEncoding::unsigned_i4, weights, 0.25, 8},
        {lp::CodeEncoding::signed_i8, a8.codes, a8.scales[0], 0});
    const auto d4 = lp::affine_group_dot({lp::CodeEncoding::unsigned_i4, weights, 0.25, 8},
        {lp::CodeEncoding::signed_i4, a4.codes, a4.scales[0], 0});
    const auto nonlinear = lp::affine_group_dot({lp::CodeEncoding::iq4_index, weights, 0.5, 0},
        {lp::CodeEncoding::signed_i8, a8.codes, a8.scales[0], 0});
    std::cout << std::setprecision(17) << "{\"schema\":\"strix-low-precision-vectors-v1\","
        "\"qualification\":\"synthetic-host-reference-only\",\"quantization_contract\":\""
        << lp::row_quantization_contract << "\",\"input\":[";
    for (std::size_t i = 0; i < input.size(); ++i) { if (i != 0) { std::cout << ','; } std::cout << input[i]; }
    std::cout << "],\"weight_u4_codes\":[";
    for (std::size_t i = 0; i < weights.size(); ++i) { if (i != 0) { std::cout << ','; } std::cout << static_cast<int>(weights[i]); }
    std::cout << ']';
    const auto print_row = [](std::string_view key, const lp::QuantizedRow& row, const lp::GroupDot& dot) {
        std::cout << ",\"" << key << "\":{\"quantization_identity\":"
            << lp::quantization_identity(row.encoding, row.group_elements)
            << ",\"scale\":" << row.scales[0] << ",\"codes\":[";
        for (std::size_t i = 0; i < row.codes.size(); ++i) {
            if (i != 0) { std::cout << ','; } std::cout << static_cast<int>(row.codes[i]);
        }
        std::cout << "],\"raw_dot\":" << dot.raw_dot << ",\"weight_sum\":" << dot.weight_sum
            << ",\"activation_sum\":" << dot.activation_sum << ",\"corrected_dot\":" << dot.corrected_dot
            << ",\"scaled_dot\":" << dot.scaled_dot << '}';
    };
    print_row("affine_w4_a8", a8, d8);
    print_row("affine_w4_a4", a4, d4);
    std::cout << ",\"iq4_lookup_a8\":{\"corrected_dot\":" << nonlinear.corrected_dot
        << ",\"scaled_dot\":" << nonlinear.scaled_dot << "},\"int8_digits\":[";
    bool first = true;
    for (int q = -128; q <= 127; ++q) {
        const auto split = lp::split_int8(q);
        if (!first) { std::cout << ','; } first = false;
        std::cout << '[' << q << ',' << split.high << ',' << split.low << ']';
    }
    std::cout << "]}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string_view(argv[1]) == "candidates") { candidates(argc, argv); }
        else if (argc == 2 && std::string_view(argv[1]) == "vectors") { vectors(); }
        else {
            std::cerr << "usage: strix-low-precision-reference candidates [--rows M --k K --n N "
                "--encoding q4_0|iq4_nl|iq4_xs --tile-m M --tile-n N --tile-k K]\n"
                "       strix-low-precision-reference vectors\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
