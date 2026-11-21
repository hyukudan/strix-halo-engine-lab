#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "inkling2/inkling2_mini_dense_hip.hpp"
#include "core/row_transforms.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

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

float sample(std::size_t i) {
    std::uint32_t bits = static_cast<std::uint32_t>(i) * 0x9e3779b9U;
    bits ^= bits >> 16U;
    return static_cast<float>(static_cast<int>(bits % 511U) - 255) * 0x1p-8F;
}

bool wide_projection_pass(unsigned inputs, std::span<const unsigned> outputs, bool round_input,
                          hipStream_t stream) {
    constexpr unsigned capacity = 200U;
    constexpr std::array live_rows{capacity, 64U, 17U};
    constexpr std::array first_rows{0U, 136U, 0U};
    float* input{};
    const std::size_t input_count = std::size_t{capacity} * inputs;
    require_hip(hipMallocManaged(&input, input_count * sizeof(float)));
    std::vector<float> rounded(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        input[i] = sample(i) + (round_input ? 0x1p-10F : 0.0F);
        rounded[i] = widen(bf16::bf16_encode(input[i]));
    }
    float* short_input{};
    const auto short_count = std::size_t{live_rows.back()} * inputs;
    require_hip(hipMallocManaged(&short_input, short_count * sizeof(float)));
    std::copy_n(input, short_count, short_input);
    const std::array call_inputs{input, input + std::size_t{first_rows[1]} * inputs, short_input};

    std::vector<std::uint16_t*> weights(outputs.size());
    std::array<std::vector<mini::MiniHipDenseProjection>, live_rows.size()> groups;
    for (std::size_t p = 0; p < outputs.size(); ++p) {
        const auto weight_count = std::size_t{outputs[p]} * inputs;
        require_hip(hipMallocManaged(&weights[p], weight_count * sizeof(std::uint16_t)));
        for (std::size_t i = 0; i < weight_count; ++i) {
            const float scale = i % 11U == 0U ? 0x1p-14F : 1.0F;
            weights[p][i] = bf16::bf16_encode(sample(i + 113U * (p + 1U)) * scale);
        }
        for (std::size_t call = 0; call < live_rows.size(); ++call) {
            // One extra row catches stray stores after a short M tile.
            const auto count = std::size_t{live_rows[call] + 1U} * outputs[p];
            float* output{};
            require_hip(hipMallocManaged(&output, count * sizeof(float)));
            std::fill_n(output, count, sentinel);
            groups[call].push_back({weights[p], outputs[p], output});
        }
    }
    for (std::size_t call = 0; call < live_rows.size(); ++call) {
        require_hip(mini::enqueue_mini_hip_dense_rows(call_inputs[call], inputs, live_rows[call],
                                                    groups[call], stream));
    }
    require_hip(hipStreamSynchronize(stream));

    bool pass = true;
    for (std::size_t p = 0; p < outputs.size(); ++p) {
        const unsigned cols = outputs[p];
        const float* full = groups[0][p].output;
        for (std::size_t call = 0; call < live_rows.size(); ++call) {
            const float* got = groups[call][p].output;
            const auto count = std::size_t{live_rows[call]} * cols;
            if (!std::all_of(got + count, got + count + cols, [](float v) { return v == sentinel; }) ||
                (call != 0U && std::memcmp(got, full + std::size_t{first_rows[call]} * cols,
                                           count * sizeof(float)) != 0)) {
                std::cerr << "Wide BF16 row isolation failed at " << inputs << " x " << cols
                          << ", rows " << live_rows[call] << '\n';
                pass = false;
            }
        }
        // The shorter calls have the same bits as these reference-checked rows.
        for (unsigned row = 0; row < capacity && pass; ++row) {
            const auto row_offset = std::size_t{row} * inputs;
            for (unsigned col = 0; col < cols; ++col) {
                const auto weight_offset = std::size_t{col} * inputs;
                float expected = 0.0F, magnitude = 0.0F;
                float original = 0.0F, original_magnitude = 0.0F;
                for (unsigned k = 0; k < inputs; ++k) {
                    const float w = widen(weights[p][weight_offset + k]);
                    const float x = rounded[row_offset + k];
                    expected = std::fma(x, w, expected);
                    magnitude += std::abs(x * w);
                    if (round_input) {
                        const float raw = input[row_offset + k];
                        original = std::fma(raw, w, original);
                        original_magnitude += std::abs(raw * w);
                    }
                }
                const float got = full[std::size_t{row} * cols + col];
                // The K16 WMMA reduction order permits a small accumulation
                // error. The original unrounded input also permits BF16 RNE.
                if (!(std::abs(got - expected) <= 0x1p-18F * magnitude) ||
                    (round_input && !(std::abs(got - original) <= (0x1p-8F + 0x1p-17F) * original_magnitude))) {
                    std::cerr << "Wide BF16 reference mismatch at " << inputs << " x " << cols
                              << ", row " << row << ", column " << col << ", got " << got
                              << ", expected " << expected << '\n';
                    pass = false;
                    break;
                }
            }
        }
    }
    for (const auto& group : groups) {
        for (const auto& projection : group) require_hip(hipFree(projection.output));
    }
    for (auto* w : weights) require_hip(hipFree(w));
    require_hip(hipFree(short_input));
    require_hip(hipFree(input));
    return pass;
}

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
    namespace prep = strix::core::row_transforms;
    require_hip(prep::elements(fixture->logits.size(),
                               prep::Bf16Softcap{fixture->logits.data(), mini::kLogitSoftcap}, stream));
    require_hip(hipStreamSynchronize(stream));

    constexpr std::array qkv_outputs{1024U, 3072U};
    constexpr std::array down_outputs{2304U};
    constexpr std::array tail_outputs{65U, 17U};
    const bool wide_pass = wide_projection_pass(2304U, qkv_outputs, false, stream) &&
        wide_projection_pass(3072U, down_outputs, false, stream) &&
        wide_projection_pass(48U, tail_outputs, true, stream);
    const bool pass = wide_pass &&
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
