#pragma once

#include "core/linear_microscaled_reference.hpp"
#include "inkling2/inkling2_mini_moe_dispatch.hpp"
#include "moe_core/microscaled_r1.hpp"
#include "strix/moe/combine.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

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
        constexpr auto scale = std::uint8_t{Nvfp4 ? 0x38U : 127U};
        std::ranges::fill(gate_codes.span(), std::uint8_t{0});
        std::ranges::fill(down_codes.span(), std::uint8_t{0});
        std::ranges::fill(gate_scales.span(), scale);
        std::ranges::fill(down_scales.span(), scale);
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
bool run_resident(typename Device::Stream stream) {
    namespace grouped = strix::moe_core::device_grouped;
    constexpr unsigned rows = 131;  // Multiple CSR chunks and a partial router block.
    constexpr unsigned assignments = rows * model::kExpertsPerToken;
    constexpr unsigned tile_rows = Device::MoeBackend::rows_per_tile;
    constexpr unsigned tile_capacity = grouped::tile_bound(rows, model::kExpertsPerToken, model::kExperts, tile_rows);
    constexpr unsigned layer = model::kMxFp8First - 1U;
    constexpr float sentinel = -171.0F;

    Managed<Device, float> input(rows * kHiddenWidth), output(rows * kHiddenWidth);
    for (std::size_t i = 0; i < input.size; ++i) input.data[i] = (i % 3U == 0U) ? -1.0F : 0.5F;
    Managed<Device, float> logits(rows * model::kExperts), bias(model::kExperts);
    std::ranges::fill(logits.span(), -12.0F);
    std::ranges::fill(bias.span(), 0.0F);
    bias.data[0] = 0.25F;  // Changes order, but does not appear in the mixture weights.
    for (unsigned row = 0; row < rows; ++row) {
        auto* values = logits.data + std::size_t(row) * model::kExperts;
        values[0] = -0.5F;  // This expert spans more than one tile for both backends.
        for (unsigned slot = 1; slot < model::kExpertsPerToken; ++slot) {
            values[1U + (row * 7U + slot * 13U) % (model::kExperts - 1U)] = 0.0F;
        }
    }
    const auto oracle_route = model::select_mini_routes(logits.span(), bias.span());
    const auto oracle = strix::moe_core::build_expert_major_plan(
        {rows, model::kExpertsPerToken, model::kExperts, 0, oracle_route.experts}, oracle_route.weights, tile_rows);

    Matrices<Device, true> positive(false), negative(true);
    Managed<Device, model::MiniMoeExpert> matrices(model::kExperts);
    for (unsigned e = 0; e < model::kExperts; ++e) matrices.data[e] = (e % 2U) ? negative.view() : positive.view();
    const auto work_elements = model::mini_moe_workspace_elements(rows);
    Managed<Device, std::uint16_t> bf16(work_elements);
    Managed<Device, float> fp32(work_elements);
    const model::MiniMoeWorkspace workspace{bf16.span(), fp32.span()};
    using Planner = grouped::Planner<model::kExperts, model::kExpertsPerToken>;
    Managed<Device, unsigned> ids(assignments), route_status(rows), scratch(Planner::scratch_bytes(rows) / sizeof(unsigned));
    Managed<Device, float> weights(assignments), assignment_weights(assignments);
    Managed<Device, unsigned> offsets(model::kExperts + 1U), touched(model::kExperts), experts(tile_capacity);
    Managed<Device, unsigned> begins(tile_capacity + 1U), row_ids(assignments), slots(assignments);
    Managed<Device, unsigned> combine_rows(assignments), tile_count(1), failure(1);
    Managed<Device, grouped::TileRows> gate_up(tile_capacity), down(tile_capacity);
    const model::MiniMoeResidentRouteBuffers route{ids.span(), route_status.span(), scratch.span(), weights.span()};
    const grouped::CsrTables csr{offsets.data, touched.data, experts.data, begins.data, row_ids.data, slots.data};
    const grouped::Output out{assignment_weights.data, combine_rows.data, gate_up.data, down.data,
                              tile_count.data, failure.data, tile_capacity};
    const auto enqueue = [&] {
        Device::require(Device::enqueue_resident(layer, rows, input.span(), logits.span(), bias.span(), output.span(),
                                                route, matrices.span(), csr, out, workspace, stream));
        Device::sync(stream);
    };
    enqueue();
    if (*failure.data != 0U || *tile_count.data != oracle.tiles.size() ||
        !std::ranges::all_of(route_status.span(), [](unsigned value) { return value == 0U; }) ||
        !std::ranges::equal(ids.span(), oracle_route.experts) ||
        !std::ranges::equal(row_ids.span(), oracle.grouping.assignment_rows) ||
        !std::ranges::equal(slots.span(), oracle.grouping.assignment_slots) ||
        !std::ranges::equal(offsets.span(), oracle.grouping.expert_offsets) ||
        !std::ranges::equal(combine_rows.span(), oracle.combine_rows)) return false;
    for (unsigned i = 0; i < assignments; ++i) {
        if (!(std::abs(weights.data[i] - oracle_route.weights[i]) <= 1e-6F) ||
            !(std::abs(assignment_weights.data[i] - oracle.assignment_weights[i]) <= 1e-6F)) return false;
    }
    for (unsigned i = 0; i < *tile_count.data; ++i) {
        const auto& tile = oracle.tiles[i];
        if (experts.data[i] != tile.expert || begins.data[i] != tile.first_assignment ||
            gate_up.data[i].rows != tile.rows || down.data[i].rows != tile.rows) return false;
    }

    // Compare the grouped computation with scalar activation and combination.
    // Use device route weights here since the normalization is checked above.
    using strix::moe::combine_round_bf16;
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned col = 0; col < kHiddenWidth; ++col) {
            const auto base = std::size_t(row) * kHiddenWidth + col % kExpertWidth;
            const float active = combine_round_bf16(model::kMiniExpertActivation<Math>(
                input.data[base], input.data[base + kExpertWidth]));
            float expected = 0.0F;
            for (unsigned rank = 0; rank < model::kExpertsPerToken; ++rank) {
                const auto index = std::size_t(row) * model::kExpertsPerToken + rank;
                const auto grouped_row = oracle.combine_rows[index];
                const auto slot = oracle.grouping.assignment_slots[grouped_row];
                const bool neg = oracle_route.experts[std::size_t(row) * model::kExpertsPerToken + slot] % 2U != 0;
                expected += assignment_weights.data[grouped_row] * (neg ? -active : active);
            }
            if (output.data[std::size_t(row) * kHiddenWidth + col] != combine_round_bf16(expected)) return false;
        }
    }

    // A nonfinite route on reuse must neither consume old descriptors nor write output.
    // Put the invalid input on a different lane to exercise the wave vote.
    logits.data[33] = std::numeric_limits<float>::quiet_NaN();
    std::ranges::fill(output.span(), sentinel);
    enqueue();
    return route_status.data[0] == 1U &&
           std::ranges::all_of(route_status.span().subspan(1), [](unsigned value) { return value == 0U; }) &&
           std::ranges::all_of(ids.span().first(model::kExpertsPerToken),
                               [](unsigned id) { return id == model::kExperts; }) &&
           std::ranges::all_of(weights.span().first(model::kExpertsPerToken),
                               [](float weight) { return weight == 0.0F; }) &&
           *failure.data != 0U && *tile_count.data == 0U &&
           std::ranges::all_of(output.span(), [](float v) { return v == sentinel; });
}

template <class Device, bool Nvfp4>
bool run_r1(typename Device::Stream stream) {
    namespace r1 = strix::moe_core::microscaled_r1;
    constexpr unsigned top_k = model::kExpertsPerToken;
    constexpr auto format = Nvfp4 ? packed::MatrixFormat::nvfp4 : packed::MatrixFormat::mxfp8;
    constexpr std::array order{6U, 2U, 0U, 5U, 3U, 4U, 7U, 1U};
    Matrices<Device, Nvfp4> positive(false), negative(true);
    Managed<Device, model::MiniMoeExpert> matrices(top_k);
    Managed<Device, unsigned> ids(top_k), failure(1);
    Managed<Device, float> weights(top_k), input(kHiddenWidth), output(kHiddenWidth);
    Managed<Device, std::uint16_t> activation(std::size_t(top_k) * kExpertWidth);
    std::ranges::copy(order, ids.data);
    for (unsigned slot = 0; slot < top_k; ++slot) {
        matrices.data[slot] = slot % 2U ? negative.view() : positive.view();
        weights.data[slot] = static_cast<float>(slot + 1U) / 8.0F;
    }
    for (unsigned i = 0; i < kHiddenWidth; ++i) input.data[i] = (i % 5U == 0U) ? -1.0F : 0.5F;
    failure.data[0] = 0;
    const r1::Args args{matrices.data, ids.data, weights.data, failure.data, input.data, activation.data, output.data,
                        top_k, kHiddenWidth, kExpertWidth, model::kMlpGateSoftcap, model::kMlpUpSoftcap};
    const auto launch = [&] {
        Device::require(r1::launch<format, top_k>(args, stream));
        Device::sync(stream);
    };
    launch();
    using strix::moe::combine_round_bf16;
    for (unsigned col = 0; col < kHiddenWidth; ++col) {
        const auto from = col % kExpertWidth;
        const float active = combine_round_bf16(model::kMiniExpertActivation<Math>(
            input.data[from], input.data[from + kExpertWidth]));
        float expected = 0.0F;
        for (unsigned expert = 0; expert < top_k; ++expert) {
            const auto slot = std::ranges::find(order, expert) - order.begin();
            expected += weights.data[slot] * (expert % 2U ? -active : active);
        }
        if (output.data[col] != combine_round_bf16(expected)) return false;
    }

    constexpr float sentinel = -171.0F;
    failure.data[0] = 1U;
    std::ranges::fill(output.span(), sentinel);
    launch();
    return std::ranges::all_of(output.span(), [](float v) { return v == sentinel; });
}

// Check the FP32 R1 dot before BF16 epilogues can hide a projection error.
template <packed::MatrixFormat Format, unsigned Outputs, unsigned Columns, class Input>
__global__ void r1_fixture_dot(packed::PackedMatrix matrix, unsigned columns, const Input* input, float* output) {
    namespace r1 = strix::moe_core::microscaled_r1::detail;
    const unsigned col = blockIdx.x * Columns;
    const unsigned lane = threadIdx.x;
    const unsigned live = columns - col;
    float dot[Outputs];
    r1::dot_rows<Format, Columns>(matrix, col, columns, live, input, static_cast<unsigned>(matrix.columns), lane, dot);
    if (lane == 0U) {
        for (unsigned i = 0; i < Outputs; ++i) {
            if (i % Columns < live) output[col + i % Columns + (i / Columns) * columns] = dot[i];
        }
    }
}

template <class Device, packed::MatrixFormat Format, class Input, unsigned Columns, unsigned Groups>
bool run_r1_dot(typename Device::Stream stream, unsigned width) {
    using Operand = packed::PackedOperand<Format>;
    namespace codec = strix::weight::scalar_float;
    constexpr unsigned columns = 5;
    constexpr unsigned rows = columns * Groups;
    constexpr unsigned ratio = Operand::kNvfp4 ? 2U : 1U;
    constexpr auto order = Groups == 2U ? packed::RowOrder::interleaved_halves : packed::RowOrder::contiguous;
    constexpr std::array<std::uint8_t, 5> fp4_scales{0x30U, 0x38U, 0x3aU, 0x42U, 0x47U};
    constexpr std::array<std::uint8_t, 6> fp8_codes{0x01U, 0x30U, 0x35U, 0x3fU, 0x9bU, 0xbcU};
    Managed<Device, std::uint8_t> codes(std::size_t{rows} * width / ratio);
    Managed<Device, std::uint8_t> scales(std::size_t{rows} * width / Operand::kBlockColumns);
    Managed<Device, Input> input(width);
    Managed<Device, float> output(rows + 1U);
    const packed::PackedMatrix matrix{codes.data, scales.data, rows, width, 13.0F / 31.0F, order};
    const Operand operand{matrix};
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned byte = 0; byte < width / ratio; ++byte) {
            const auto index = std::size_t{row} * (width / ratio) + byte;
            const auto seed = 11U * row + 7U * byte + byte / 3U;
            if constexpr (Operand::kNvfp4) codes.data[index] = static_cast<std::uint8_t>(seed & 0xffU);
            else codes.data[index] = fp8_codes[seed % fp8_codes.size()];
        }
        for (unsigned block = 0; block < width / Operand::kBlockColumns; ++block) {
            const auto index = row * (width / Operand::kBlockColumns) + block;
            const auto variation = (row + 3U * block) % fp4_scales.size();
            if constexpr (Operand::kNvfp4) scales.data[index] = fp4_scales[variation];
            else scales.data[index] = static_cast<std::uint8_t>(125U + variation);
        }
    }
    std::vector<float> rounded(width);
    for (unsigned k = 0; k < width; ++k) {
        const float value = static_cast<float>(static_cast<int>((k + k / 13U) % 37U) - 18) / 13.0F;
        const auto bf16 = codec::bf16_encode(value);
        rounded[k] = std::bit_cast<float>(std::uint32_t{bf16} << 16U);
        if constexpr (std::is_same_v<Input, float>) input.data[k] = value;
        else input.data[k] = bf16;
    }
    constexpr float sentinel = -171.0F;
    std::ranges::fill(output.span(), sentinel);
    const auto launch = [&] {
        r1_fixture_dot<Format, Columns * Groups, Columns><<<(columns - 1U) / Columns + 1U, 32U, 0, stream>>>(
            matrix, columns, input.data, output.data);
        Device::sync(stream);
    };
    launch();
    std::array<float, rows> reference{};
    strix::core::microscaled_matvec_reference<Format>(matrix, rounded, reference);
    std::array<std::uint32_t, rows> first{};
    for (unsigned row = 0; row < rows; ++row) {
        float magnitude = 0.0F;
        for (unsigned k = 0; k < width; k += 16U) {
            std::uint32_t weights[8];
            operand.decode_bf16(operand.load_tile16(row, k), weights);
            for (unsigned pair = 0; pair < 8U; ++pair) {
                const float lo = std::bit_cast<float>(weights[pair] << 16U);
                const float hi = std::bit_cast<float>(weights[pair] & 0xffff0000U);
                magnitude += std::abs(lo * rounded[k + 2U * pair]) + std::abs(hi * rounded[k + 2U * pair + 1U]);
            }
        }
        // Scale the tolerance by the term magnitudes even when the sum cancels.
        constexpr float tolerance = 0x1p-15F;
        if (!(std::abs(output.data[row] - reference[row]) <= tolerance * magnitude)) {
            std::cerr << "R1 dot differs from serial reference at width " << width << ", row " << row << '\n';
            return false;
        }
        first[row] = std::bit_cast<std::uint32_t>(output.data[row]);
    }
    if (output.data[rows] != sentinel) return false;
    std::ranges::fill(output.span(), sentinel);
    launch();
    for (unsigned row = 0; row < rows; ++row) {
        if (std::bit_cast<std::uint32_t>(output.data[row]) != first[row]) return false;
    }
    return output.data[rows] == sentinel;
}

template <class Device>
int verify() {
    using packed::MatrixFormat;
    namespace r1 = strix::moe_core::microscaled_r1::detail;
    const auto stream = Device::create_stream();
    const bool passed = run<Device, true>(stream) && run<Device, false>(stream) &&
                        run_resident<Device>(stream) && run_r1<Device, true>(stream) && run_r1<Device, false>(stream) &&
                        run_r1_dot<Device, MatrixFormat::nvfp4, float, r1::kGateColumns, 2U>(stream, kHiddenWidth) &&
                        run_r1_dot<Device, MatrixFormat::mxfp8, float, r1::kGateColumns, 2U>(stream, kHiddenWidth) &&
                        run_r1_dot<Device, MatrixFormat::nvfp4, std::uint16_t, r1::kDownColumns, 1U>(stream, kExpertWidth) &&
                        run_r1_dot<Device, MatrixFormat::mxfp8, std::uint16_t, r1::kDownColumns, 1U>(stream, kExpertWidth) &&
                        run_r1_dot<Device, MatrixFormat::nvfp4, float, r1::kGateColumns, 2U>(stream, 48U);
    Device::destroy_stream(stream);
    return passed ? 0 : 1;
}

}  // namespace strix::tests::mini_moe
