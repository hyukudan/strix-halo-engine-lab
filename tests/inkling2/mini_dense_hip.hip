#include "inkling2/inkling2_mini_dense_hip.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

namespace {

namespace mini = strix::inkling2;
namespace bf16 = strix::weight::scalar_float;
constexpr unsigned rows = 3, width = 32, first_outputs = 9, second_outputs = 5, relative_width = 16;
constexpr unsigned norm_width = 96, norm_stride = 101, norm_output_stride = 107;
constexpr float sentinel = -999.0F;

struct Fixture {
    alignas(16) std::array<float, rows * width> input{};
    alignas(16) std::array<float, rows * relative_width> relative_input{};
    std::array<std::uint16_t, first_outputs * width> first_weight{};
    std::array<std::uint16_t, second_outputs * width> second_weight{};
    std::array<std::uint16_t, second_outputs * relative_width> relative_weight{};
    std::array<float, rows * first_outputs> first{};
    std::array<float, rows * second_outputs> second{}, relative{};

    std::array<float, rows * norm_stride> norm_input{};
    std::array<std::uint16_t, norm_width> norm_scale{};
    std::array<float, rows * norm_output_stride> norm_parallel{}, norm_serial{};
    std::array<float, 6> logits{1000.0F, 1200.0F, -1000.0F, 1.00390625F,
                                std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()};
};

void require_hip(hipError_t error) {
    if (error == hipSuccess) return;
    std::cerr << hipGetErrorString(error) << '\n';
    std::exit(1);
}

float widen(std::uint16_t bits) { return bf16::u2f(std::uint32_t{bits} << 16U); }

bool projection_pass(std::span<const float> in, std::span<const std::uint16_t> weights, std::span<const float> out,
                     unsigned in_width) {
    const auto outputs = weights.size() / in_width;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t j = 0; j < outputs; ++j) {
            float expected = 0.0F;
            for (unsigned k = 0; k < in_width; ++k)
                expected = std::fma(in[r * in_width + k], widen(weights[j * in_width + k]), expected);
            if (out[r * outputs + j] != expected) return false;
        }
    }
    return true;
}

bool norm_pass(const Fixture& f) {
    bool distinguishes_double_round = false;
    for (unsigned r = 0; r < rows; ++r) {
        float sum = 0.0F;
        for (unsigned k = 0; k < norm_width; ++k) {
            const auto x = f.norm_input[r * norm_stride + k];
            sum = std::fma(x, x, sum);
        }
        const float inverse = 1.0F / std::sqrt(sum / static_cast<float>(norm_width) + 1e-6F);
        for (unsigned k = 0; k < norm_output_stride; ++k) {
            float expected = sentinel;
            if (k < norm_width) {
                const float normalized = f.norm_input[r * norm_stride + k] * inverse;
                const float weight = widen(f.norm_scale[k]);
                expected = widen(bf16::bf16_encode(normalized * weight));
                const float double_rounded = widen(bf16::bf16_encode(widen(bf16::bf16_encode(normalized)) * weight));
                distinguishes_double_round |= expected != double_rounded;
            }
            if (f.norm_parallel[r * norm_output_stride + k] != expected ||
                f.norm_serial[r * norm_output_stride + k] != expected) return false;
        }
    }
    return distinguishes_double_round;
}

bool softcap_pass(const Fixture& f) {
    const float expected = mini::kLogitSoftcap * std::tanh(1.0F / mini::kLogitSoftcap);
    return f.logits[0] == mini::kLogitSoftcap && f.logits[1] == mini::kLogitSoftcap &&
           f.logits[2] == -mini::kLogitSoftcap && std::abs(f.logits[3] - expected) <= 2.0e-6F &&
           std::isinf(f.logits[4]) && std::isnan(f.logits[5]);
}

}  // namespace

int main() {
    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, sizeof(Fixture)));
    auto* fixture = std::construct_at(static_cast<Fixture*>(storage));
    const auto value = [](std::size_t i) { return 0.0625F * static_cast<float>(static_cast<int>(i % 17U) - 8); };
    for (std::size_t i = 0; i < fixture->input.size(); ++i) fixture->input[i] = value(i);
    for (std::size_t i = 0; i < fixture->relative_input.size(); ++i) fixture->relative_input[i] = value(i + 7U);
    for (std::size_t i = 0; i < fixture->first_weight.size(); ++i)
        fixture->first_weight[i] = bf16::bf16_encode(value(i + 3U));
    for (std::size_t i = 0; i < fixture->second_weight.size(); ++i)
        fixture->second_weight[i] = bf16::bf16_encode(value(2U * i + 1U));
    for (std::size_t i = 0; i < fixture->relative_weight.size(); ++i)
        fixture->relative_weight[i] = bf16::bf16_encode(value(i + 11U));
    for (std::size_t i = 0; i < fixture->norm_input.size(); ++i) fixture->norm_input[i] = value(3U * i + 4U);
    for (std::size_t i = 0; i < fixture->norm_scale.size(); ++i)
        fixture->norm_scale[i] = bf16::bf16_encode(1.0F + value(i));
    fixture->norm_parallel.fill(sentinel);
    fixture->norm_serial.fill(sentinel);

    hipStream_t stream{};
    require_hip(hipStreamCreate(&stream));
    const std::array projections{
        mini::MiniHipDenseProjection{fixture->first_weight.data(), first_outputs, fixture->first.data()},
        mini::MiniHipDenseProjection{fixture->second_weight.data(), second_outputs, fixture->second.data()},
    };
    const std::array relative{mini::MiniHipDenseProjection{
        fixture->relative_weight.data(), second_outputs, fixture->relative.data()}};
    require_hip(mini::enqueue_mini_hip_dense_rows(fixture->input.data(), width, rows, projections, stream));
    require_hip(mini::enqueue_mini_hip_dense_rows(fixture->relative_input.data(), relative_width, rows, relative, stream));
    const mini::MiniHipRmsRows parallel{fixture->norm_input.data(), fixture->norm_scale.data(),
        fixture->norm_parallel.data(), rows, norm_width, norm_stride, norm_output_stride};
    auto serial = parallel;
    serial.output = fixture->norm_serial.data();
    require_hip(mini::enqueue_mini_hip_rms(parallel, stream));
    require_hip(mini::enqueue_mini_hip_rms(serial, stream, mini::MiniHipRmsReduction::serial_reference));
    namespace prep = strix::core::prepared_rows_device;
    require_hip(prep::elements(fixture->logits.size(),
                               prep::Bf16Softcap{fixture->logits.data(), mini::kLogitSoftcap}, stream));
    require_hip(hipStreamSynchronize(stream));

    const bool pass =
        projection_pass(fixture->input, fixture->first_weight, fixture->first, width) &&
        projection_pass(fixture->input, fixture->second_weight, fixture->second, width) &&
        projection_pass(fixture->relative_input, fixture->relative_weight, fixture->relative, relative_width) &&
        norm_pass(*fixture) && softcap_pass(*fixture) &&
        mini::enqueue_mini_hip_dense_rows(fixture->input.data(), width - 1U, rows, projections, stream) == hipErrorInvalidValue &&
        mini::enqueue_mini_hip_rms({}, stream) == hipErrorInvalidValue;
    if (!pass) std::cerr << "Mini HIP dense projections or RMS differ from the host reference\n";
    require_hip(hipStreamDestroy(stream));
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    return pass ? 0 : 1;
}
