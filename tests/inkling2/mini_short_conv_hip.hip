#include "core/hip_short_convolution.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>

namespace {

namespace short_conv = strix::core::hip_short_conv;
using strix::core::ShortConvolution;
using strix::core::causal_short_convolution;

constexpr std::size_t channels = 513, groups = 3, rows = 11, max_taps = 7;

struct Fixture {
    std::array<float, rows * channels> input{}, output{};
    std::array<float, max_taps * channels> static_weights{};
    std::array<float, rows * groups * max_taps> dynamic_weights{};
    std::array<float, (max_taps - 1) * channels> history{}, other_visit{};
};

void require_hip(hipError_t error) {
    if (error == hipSuccess) return;
    std::cerr << hipGetErrorString(error) << '\n';
    std::exit(1);
}

void fill(std::span<float> data, std::size_t modulus, float scale) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        const auto sample = static_cast<int>((i * 29U + i / 7U) % modulus);
        data[i] = static_cast<float>(sample - static_cast<int>(modulus / 2U)) * scale;
    }
}

bool same_bits(std::span<const float> expected, std::span<const float> actual) {
    const auto equal_float = [](float left, float right) {
        return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
    };
    return std::ranges::equal(expected, actual, equal_float);
}

bool run(Fixture& fixture, const ShortConvolution& plan, std::span<const std::size_t> chunks,
         hipStream_t stream) {
    fill(fixture.input, 107, 0.043F);
    fill(fixture.static_weights, 173, 0.027F);
    fill(fixture.dynamic_weights, 41, 0.019F);
    fill(fixture.history, 97, 0.031F);
    fill(fixture.other_visit, 79, 0.17F);
    std::ranges::fill(fixture.output, -999.0F);

    const auto other_before = fixture.other_visit;
    const auto history = std::span(fixture.history).first((plan.taps - 1) * channels);
    const auto stat = std::span(fixture.static_weights).first(plan.taps * channels);
    const auto dyn = std::span(fixture.dynamic_weights).first(rows * plan.dynamic_groups * plan.taps);
    std::array<float, rows * channels> expected{};
    auto expected_history = fixture.history;
    causal_short_convolution(plan, fixture.input, stat, dyn, std::span(expected_history).first(history.size()), expected);

    std::size_t start = 0;
    for (const auto count : chunks) {
        const auto* dynamic = dyn.empty() ? nullptr : dyn.data() + start * plan.dynamic_groups * plan.taps;
        const short_conv::Args args{
            count, fixture.input.data() + start * channels, stat.data(), dynamic,
            history.empty() ? nullptr : history.data(), fixture.output.data() + start * channels,
        };
        require_hip(short_conv::launch(plan, args, stream));
        start += count;
    }
    require_hip(hipStreamSynchronize(stream));
    return start == rows && same_bits(expected, fixture.output) &&
           same_bits(expected_history, fixture.history) && same_bits(other_before, fixture.other_visit);
}

}  // namespace

int main() {
    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, sizeof(Fixture)));
    auto* fixture = std::construct_at(static_cast<Fixture*>(storage));
    hipStream_t stream{};
    require_hip(hipStreamCreate(&stream));

    const ShortConvolution static_plan{channels, 4, 0, 1.0F, std::sqrt(5.0F)};
    const ShortConvolution dynamic_plan{channels, 4, groups, 1.0F, std::sqrt(9.0F)};
    const ShortConvolution long_history{channels, max_taps, groups, 0.6F, std::sqrt(15.0F)};
    const ShortConvolution one_tap{channels, 1};
    constexpr std::array all{rows};
    constexpr std::array<std::size_t, 6> chunks{2, 1, 3, 1, 2, 2};
    const bool pass = run(*fixture, static_plan, all, stream) && run(*fixture, static_plan, chunks, stream) &&
                      run(*fixture, dynamic_plan, all, stream) && run(*fixture, dynamic_plan, chunks, stream) &&
                      run(*fixture, long_history, all, stream) && run(*fixture, long_history, chunks, stream) &&
                      run(*fixture, one_tap, chunks, stream);

    const short_conv::Args missing_dynamic{
        1, fixture->input.data(), fixture->static_weights.data(), nullptr, fixture->history.data(), fixture->output.data(),
    };
    const bool rejects = short_conv::launch(dynamic_plan, missing_dynamic, stream) == hipErrorInvalidValue;
    require_hip(hipStreamDestroy(stream));
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    if (!pass) std::cerr << "HIP causal convolution differs from host reference\n";
    return pass && rejects ? 0 : 1;
}
