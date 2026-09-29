#include "strix/recurrent_plan.hpp"

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
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::RecurrentWindowRequest request;
    request.logical_tokens = 4;
    request.sessions = 1;

    const auto plan =
        strix::build_recurrent_window_plan(
            model, request);

    expect(plan.key_repeat_factor == 3,
           "16 key heads repeat three times to 48 value heads");
    expect(plan.matrix_state_bytes_per_layer ==
               3ULL * 1024ULL * 1024ULL,
           "one GDN layer has 3 MiB FP32 matrix state");
    expect(plan.conv_state_bytes_per_layer ==
               80ULL * 1024ULL,
           "one GDN layer has 80 KiB BF16 convolution history");
    expect(plan.persistent_state_bytes_all_layers ==
               110ULL * 1024ULL * 1024ULL +
                   832ULL * 1024ULL,
           "all 36 GDN layers total 110.8125 MiB");
    expect(plan.serial_state_read_bytes ==
               4ULL * plan.persistent_state_bytes_all_layers,
           "four-token serial path reads state four times");
    expect(plan.serial_state_write_bytes ==
               plan.serial_state_read_bytes,
           "serial path writes state once per token");
    expect(plan.retained_state_read_bytes ==
               plan.persistent_state_bytes_all_layers,
           "retained path reads state once per window");
    expect(plan.retained_state_write_bytes ==
               plan.persistent_state_bytes_all_layers,
           "retained path writes state once per window");
    expect(plan.state_roundtrip_bytes_saved ==
               6ULL * plan.persistent_state_bytes_all_layers,
           "four-token retained window removes six state traversals");
    expect(plan.full_candidate_snapshot_bytes ==
               4ULL * plan.persistent_state_bytes_all_layers,
           "full candidate snapshots clone state four times");
    expect(plan.pre_window_snapshot_bytes ==
               plan.persistent_state_bytes_all_layers,
           "pre-window rollback needs one state boundary");
    expect(plan.snapshot_bytes_saved ==
               3ULL * plan.persistent_state_bytes_all_layers,
           "pre-window strategy avoids three state clones");

    auto matrix_only = request;
    matrix_only.include_conv_state = false;
    const auto matrix_plan =
        strix::build_recurrent_window_plan(
            model, matrix_only);
    expect(matrix_plan.conv_state_bytes_per_layer == 0,
           "planner can isolate matrix traffic");
    expect(matrix_plan.persistent_state_bytes_all_layers ==
               108ULL * 1024ULL * 1024ULL,
           "matrix-only total is 108 MiB");

    bool invalid = false;
    try {
        auto bad = request;
        bad.logical_tokens = 0;
        static_cast<void>(
            strix::build_recurrent_window_plan(
                model, bad));
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    expect(invalid,
           "zero-token recurrent window is rejected");

    bool overflow = false;
    try {
        auto bad = request;
        bad.sessions =
            std::numeric_limits<std::uint32_t>::max();
        auto huge = model;
        huge.gdn_key_heads = 1;
        huge.gdn_value_heads =
            std::numeric_limits<std::uint64_t>::max();
        static_cast<void>(
            strix::build_recurrent_window_plan(
                huge, bad));
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    expect(overflow,
           "recurrent-plan overflow is rejected");

    if (failures == 0) {
        std::cout << "recurrent plan tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
