#include "strix/packed_projection.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    expect(strix::supported_low_row_class(1),
           "one row is supported");
    expect(strix::supported_low_row_class(2),
           "two rows are supported");
    expect(strix::supported_low_row_class(4),
           "four rows are supported");
    expect(strix::supported_low_row_class(8),
           "eight rows are supported");
    expect(!strix::supported_low_row_class(3),
           "three rows are not an explicit low-row class");

    strix::PackedProjectionRequest q4k;
    q4k.ggml_type = 12;
    q4k.input_features = 2560;
    q4k.output_features = 2560;
    q4k.logical_rows = 1;

    const auto q4k_plan =
        strix::build_packed_projection_plan(q4k);
    expect(q4k_plan.block_elements == 256 &&
               q4k_plan.block_bytes == 144,
           "Q4_K storage geometry is reused from GGUF contract");
    expect(q4k_plan.blocks_per_output_row == 10,
           "Q4_K 2560-wide row contains ten blocks");
    expect(q4k_plan.weight_bytes_per_output_row == 1440,
           "Q4_K physical bytes per output row are exact");
    expect(q4k_plan.total_weight_bytes == 3'686'400,
           "Q4_K 2560x2560 tensor bytes are exact");

    auto q8 = q4k;
    q8.ggml_type = 8;
    q8.logical_rows = 4;
    q8.request_concurrency = 1;
    const auto q8_plan =
        strix::build_packed_projection_plan(q8);
    expect(q8_plan.block_elements == 32 &&
               q8_plan.block_bytes == 34,
           "Q8_0 storage geometry is available");
    expect(q8_plan.weight_bytes_per_output_row == 2720,
           "Q8_0 2560-wide row bytes are exact");
    expect(q8_plan.total_weight_bytes == 6'963'200,
           "Q8_0 2560x2560 tensor bytes are exact");
    expect(q8_plan.serial_weight_traffic_bytes ==
               4ULL * q8_plan.total_weight_bytes,
           "four independent rows would read four tensors worth of weights");
    expect(q8_plan.ideal_reused_weight_traffic_bytes ==
               q8_plan.total_weight_bytes,
           "ideal four-row kernel reads packed weights once");
    expect(q8_plan.input_activation_bytes ==
               4ULL * 2560ULL * sizeof(float),
           "activation input bytes are explicit");
    expect(q8_plan.output_activation_bytes ==
               4ULL * 2560ULL * sizeof(float),
           "activation output bytes are explicit");

    bool invalid_rows = false;
    try {
        auto invalid = q4k;
        invalid.logical_rows = 3;
        static_cast<void>(
            strix::build_packed_projection_plan(invalid));
    } catch (const std::invalid_argument&) {
        invalid_rows = true;
    }
    expect(invalid_rows,
           "unsupported low-row shape is rejected");

    bool invalid_geometry = false;
    try {
        auto invalid = q4k;
        invalid.input_features = 2500;
        static_cast<void>(
            strix::build_packed_projection_plan(invalid));
    } catch (const std::invalid_argument&) {
        invalid_geometry = true;
    }
    expect(invalid_geometry,
           "misaligned quantized row is rejected");

    bool overflow = false;
    try {
        auto invalid = q4k;
        invalid.output_features =
            std::numeric_limits<std::uint64_t>::max();
        static_cast<void>(
            strix::build_packed_projection_plan(invalid));
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    expect(overflow,
           "packed projection byte overflow is rejected");

    if (failures == 0) {
        std::cout << "packed projection tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
