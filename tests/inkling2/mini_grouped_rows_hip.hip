#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/grouped_rows.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

namespace grouped = strix::moe_core::grouped_rows;
constexpr unsigned rows = 2, top_k = 3, assignments = rows * top_k, width = 257;
constexpr unsigned input_stride = width + 1U, grouped_stride = width + 7U, output_stride = width + 3U;
constexpr std::uint16_t grouped_guard = 0xff81U;
constexpr float output_guard = -2468.0F;

struct Fixture {
    std::array<float, rows * input_stride> input{};
    std::array<std::uint32_t, assignments> source{0, 1, 0, 1, 0, 1};
    std::array<std::uint32_t, assignments> combine_rows{0, 2, 4, 1, 3, 5};
    std::array<float, assignments> weights{1.0F, 0.5F, 1.0F, 0.25F, 1.0F, 2.0F};
    std::array<std::uint16_t, assignments * grouped_stride> gathered{};
    std::array<float, assignments * grouped_stride> projected{};
    std::array<float, rows * output_stride> output{};
};

void require_hip(hipError_t status) {
    if (status == hipSuccess) return;
    std::cerr << hipGetErrorString(status) << '\n';
    std::exit(1);
}

bool run(Fixture& fixture, hipStream_t stream) {
    for (unsigned column = 0; column < width; ++column) {
        fixture.input[column] = column % 2U == 0U ? -0.0F : 1.00390625F;
        fixture.input[input_stride + column] = column % 2U == 0U ? 1.01171875F : -1.0F;
        // Expert order matters: adding the small term first would lose it.
        fixture.projected[column] = 0x1p30F;
        fixture.projected[2U * grouped_stride + column] = -0x1p30F;
        fixture.projected[4U * grouped_stride + column] = 1.00390625F;
        // Alternate a tie when rounding the down operand with a tie in the final sum.
        fixture.projected[grouped_stride + column] = column % 2U == 0U ? 1.01171875F : 1.0078125F;
        fixture.projected[3U * grouped_stride + column] = 2.0F;
        fixture.projected[5U * grouped_stride + column] = 0.25F;
    }
    fixture.gathered.fill(grouped_guard);
    fixture.output.fill(output_guard);

    const grouped::GatherArgs gather{
        fixture.input.data(), input_stride, fixture.source.data(), assignments, width, fixture.gathered.data(), grouped_stride,
    };
    const grouped::CombineArgs combine{
        fixture.projected.data(), grouped_stride, fixture.combine_rows.data(), fixture.weights.data(),
        rows, top_k, width, fixture.output.data(), output_stride,
    };
    auto invalid_gather = gather;
    invalid_gather.grouped_stride = width - 1U;
    auto invalid_combine = combine;
    invalid_combine.top_k = 0;
    if (grouped::gather(invalid_gather, stream) != hipErrorInvalidValue ||
        grouped::combine(invalid_combine, stream) != hipErrorInvalidValue) return false;
    require_hip(grouped::gather(gather, stream));
    require_hip(grouped::combine(combine, stream));
    require_hip(hipStreamSynchronize(stream));

    for (unsigned assignment = 0; assignment < assignments; ++assignment) {
        for (unsigned column = 0; column < grouped_stride; ++column) {
            std::uint16_t expected = grouped_guard;
            if (column < width) {
                if (fixture.source[assignment] == 0U) expected = column % 2U == 0U ? 0x8000U : 0x3f80U;
                else expected = column % 2U == 0U ? 0x3f82U : 0xbf80U;
            }
            if (fixture.gathered[assignment * grouped_stride + column] != expected) return false;
        }
    }
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned column = 0; column < output_stride; ++column) {
            const float expected_row1 = column % 2U == 0U ? 1.5078125F : 1.5F;
            const float expected = column < width ? (row == 0 ? 1.0F : expected_row1) : output_guard;
            if (fixture.output[row * output_stride + column] != expected) return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    void* memory = nullptr;
    require_hip(hipMallocManaged(&memory, sizeof(Fixture)));
    auto* fixture = std::construct_at(static_cast<Fixture*>(memory));
    hipStream_t stream = nullptr;
    require_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    const bool ok = run(*fixture, stream);
    require_hip(hipStreamDestroy(stream));
    std::destroy_at(fixture);
    require_hip(hipFree(memory));
    if (!ok) std::cerr << "Expert-major gather or ordered combine mismatch\n";
    return ok ? 0 : 1;
}
