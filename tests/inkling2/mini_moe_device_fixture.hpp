#pragma once

#include "inkling2/inkling2_mini_moe_dispatch.hpp"
#include "strix/moe/combine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <span>

namespace strix::tests::mini_moe {

namespace model = strix::inkling2;
namespace packed = strix::weight::microscaled;
using model::kExpertWidth;
using model::kHiddenWidth;
constexpr unsigned tokens = 2;

template <class Device, class T>
struct Managed {
    T* data;
    std::size_t size;
    explicit Managed(std::size_t count) : data(static_cast<T*>(Device::allocate(count * sizeof(T)))), size(count) {}
    ~Managed() { Device::release(data); }
    Managed(const Managed&) = delete;
    Managed& operator=(const Managed&) = delete;
    std::span<T> span() { return {data, size}; }
};

struct Math {
    __host__ __device__ static float exp(float value) { return ::expf(value); }
    __host__ __device__ static float tanh(float value) { return ::tanhf(value); }
};

template <class Device, bool Nvfp4>
struct Matrices {
    static constexpr unsigned code_ratio = Nvfp4 ? 2U : 1U;
    static constexpr unsigned block = Nvfp4 ? 16U : 32U;
    Managed<Device, std::uint8_t> gate_codes{std::size_t{kHiddenWidth} * kHiddenWidth / code_ratio};
    Managed<Device, std::uint8_t> gate_scales{std::size_t{kHiddenWidth} * kHiddenWidth / block};
    Managed<Device, std::uint8_t> down_codes{std::size_t{kExpertWidth} * kHiddenWidth / code_ratio};
    Managed<Device, std::uint8_t> down_scales{std::size_t{kExpertWidth} * kHiddenWidth / block};

    static void set(std::uint8_t* codes, unsigned row, unsigned inputs, unsigned col, bool negative) {
        const auto offset = std::size_t{row} * inputs / code_ratio + col / code_ratio;
        codes[offset] = Nvfp4 ? static_cast<std::uint8_t>((negative ? 10U : 2U) << (4U * (col % 2U)))
                              : static_cast<std::uint8_t>(negative ? 0xb8U : 0x38U);
    }

    explicit Matrices(bool negative) {
        std::ranges::fill(gate_codes.span(), 0);
        std::ranges::fill(down_codes.span(), 0);
        std::ranges::fill(gate_scales.span(), Nvfp4 ? 0x38U : 127U);
        std::ranges::fill(down_scales.span(), Nvfp4 ? 0x38U : 127U);
        for (unsigned col = 0; col < kExpertWidth; ++col) {
            set(gate_codes.data, 2U * col, kHiddenWidth, col, false);
            set(gate_codes.data, 2U * col + 1U, kHiddenWidth, col + kExpertWidth, false);
        }
        for (unsigned col = 0; col < kHiddenWidth; ++col)
            set(down_codes.data, col, kExpertWidth, col % kExpertWidth, negative);
    }

    model::MiniMoeExpert view() {
        return {{gate_codes.data, gate_scales.data, kHiddenWidth, kHiddenWidth, 1.0F,
                 packed::RowOrder::interleaved_halves},
                {down_codes.data, down_scales.data, kHiddenWidth, kExpertWidth, 1.0F, packed::RowOrder::contiguous}};
    }
};

template <class Device, bool Nvfp4>
bool run(typename Device::Stream stream) {
    constexpr unsigned layer = Nvfp4 ? model::kMxFp8First - 1U : model::kMxFp8First;
    constexpr unsigned assignments = tokens * model::kExpertsPerToken;
    const std::array<std::uint32_t, assignments> experts{6, 0, 7, 1, 5, 2, 4, 3, 8, 4, 9, 5, 7, 2, 6, 3};
    std::array<float, assignments> weights;
    for (unsigned i = 0; i < assignments; ++i)
        weights[i] = static_cast<float>((i % model::kExpertsPerToken) + 1U) * 0.125F;
    Matrices<Device, Nvfp4> positive(false), negative(true);
    std::array<model::MiniMoeExpert, model::kExperts> matrices;
    for (unsigned i = 0; i < matrices.size(); ++i) matrices[i] = i % 2U ? negative.view() : positive.view();

    Managed<Device, float> input(tokens * kHiddenWidth), output(tokens * kHiddenWidth);
    for (unsigned i = 0; i < input.size; ++i)
        input.data[i] = static_cast<float>(static_cast<int>((i + i / kHiddenWidth) % 23U) - 11) * 0.1234567F;

    const auto scratch = model::mini_moe_workspace_elements(tokens);
    Managed<Device, std::uint16_t> bf16(scratch);
    Managed<Device, float> fp32(scratch);
    const model::MiniMoeWorkspace workspace{bf16.span(), fp32.span()};
    const auto plan = model::prepare_mini_moe<typename Device::MoeBackend>(layer, tokens, experts, weights, matrices, workspace);
    Managed<Device, std::uint32_t> assignment_rows(assignments), combine_rows(assignments);
    Managed<Device, float> assignment_weights(assignments);
    using Tile = strix::moe_core::grouped_microscaled::TileRows;
    Managed<Device, Tile> gate_up(plan.gate_up.size()), down(plan.down.size());
    std::ranges::copy(plan.routes.grouping.assignment_rows, assignment_rows.data);
    std::ranges::copy(plan.routes.combine_rows, combine_rows.data);
    std::ranges::copy(plan.routes.assignment_weights, assignment_weights.data);
    std::ranges::copy(plan.gate_up, gate_up.data);
    std::ranges::copy(plan.down, down.data);
    const model::MiniMoeDevicePlan device{layer, tokens, assignment_rows.span(), combine_rows.span(),
                                             assignment_weights.span(), gate_up.span(), down.span()};
    const auto short_output = output.span().first(kHiddenWidth);
    if (model::enqueue_mini_moe<typename Device::MoeBackend>(device, input.span(), short_output, workspace, stream) !=
        Device::MoeBackend::invalid_value) return false;
    Device::require(model::enqueue_mini_moe<typename Device::MoeBackend>(
        device, input.span(), output.span(), workspace, stream));
    Device::sync(stream);

    for (unsigned row = 0; row < tokens; ++row) {
        for (unsigned col = 0; col < kHiddenWidth; ++col) {
            using strix::moe::combine_round_bf16;
            const unsigned index = row * kHiddenWidth + col % kExpertWidth;
            const float gate = combine_round_bf16(input.data[index]);
            const float up = combine_round_bf16(input.data[index + kExpertWidth]);
            const float active = combine_round_bf16(model::kMiniExpertActivation<Math>(gate, up));
            float expected = 0.0F;
            for (unsigned expert = 0; expert < model::kExperts; ++expert) {
                for (unsigned slot = 0; slot < model::kExpertsPerToken; ++slot) {
                    const unsigned assignment = row * model::kExpertsPerToken + slot;
                    if (experts[assignment] == expert)
                        expected += weights[assignment] * (expert % 2U ? -active : active);
                }
            }
            if (output.data[row * kHiddenWidth + col] != combine_round_bf16(expected)) {
                std::cerr << "Mini MoE mismatch at row " << row << " and column " << col << '\n';
                return false;
            }
        }
    }
    return true;
}

template <class Device>
int verify() {
    const auto stream = Device::create_stream();
    const bool passed = run<Device, true>(stream) && run<Device, false>(stream);
    Device::destroy_stream(stream);
    return passed ? 0 : 1;
}

}  // namespace strix::tests::mini_moe
