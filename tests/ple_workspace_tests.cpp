#include "strix/ple_workspace.hpp"

#include <iostream>
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
    const auto layout =
        strix::build_ple_workspace_layout(
            model);

    expect(
        layout.gathered_bytes == 5'120,
        "PLE workspace stores 16x160 BF16 gathered embedding");
    expect(
        layout.key_bytes == 40'960 &&
            layout.value_bytes == 10'240,
        "PLE workspace stores FP32 key/value projections");
    expect(
        layout.gated_bytes == 40'960 &&
            layout.normalized_bytes == 40'960,
        "PLE workspace stores gated and normalized hc streams");
    expect(
        layout.gate_bytes == 16,
        "PLE workspace stores four FP32 stream gates");
    expect(
        layout.total_bytes == 138'496,
        "PLE workspace total includes deterministic 256-byte alignment");

    const auto second =
        strix::build_ple_workspace_layout(
            model);
    expect(
        second.total_bytes ==
            layout.total_bytes &&
            second.gate_offset ==
                layout.gate_offset,
        "PLE workspace layout is deterministic");

    bool zero_alignment = false;
    try {
        static_cast<void>(
            strix::build_ple_workspace_layout(
                model,
                0));
    } catch (const std::invalid_argument&) {
        zero_alignment = true;
    }
    expect(
        zero_alignment,
        "PLE workspace rejects zero alignment");

    if (failures == 0) {
        std::cout
            << "PLE workspace tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
