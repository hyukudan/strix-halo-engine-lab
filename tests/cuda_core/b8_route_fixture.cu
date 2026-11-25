#include "cuda/core/moe_route_plan_r1.cuh"
#include "cuda/core/route_select.cuh"
#include "../moe_core/b8_route_cases.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace b8 = strix::tests::b8;
namespace core = strix::cuda::core;
namespace fused = core::moe_route_plan_r1;
namespace plan_r1 = fused::plan_r1;
namespace r1 = fused::experts_r1;
namespace packed = r1::packed;
using packed::MatrixFormat;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
struct Device {
    T* data{};
    std::size_t count;
    explicit Device(std::size_t n) : count(n) { check(cudaMalloc(&data, n * sizeof(T))); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    void write(std::span<const T> values) {
        require(values.size() == count, "device upload size");
        check(cudaMemcpy(data, values.data(), values.size_bytes(), cudaMemcpyHostToDevice));
    }
    void fill(T value) {
        const std::vector<T> values(count, value);
        write(values);
    }
    std::vector<T> read() const {
        std::vector<T> values(count);
        check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

template <class T>
void same(const std::vector<T>& actual, const std::vector<T>& expected, const char* message) {
    require(actual.size() == expected.size() &&
            std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0, message);
}

template <MatrixFormat Format>
struct Matrices {
    using Operand = packed::PackedOperand<Format>;
    unsigned rows, width;
    std::size_t code_stride, scale_stride;
    Device<std::uint8_t> codes, scales;

    Matrices(unsigned r, unsigned w)
        : rows(r), width(w), code_stride(std::size_t{r} * w / (Operand::kNvfp4 ? 2U : 1U)),
          scale_stride(std::size_t{r} * w / Operand::kBlockColumns),
          codes(b8::experts * code_stride), scales(b8::experts * scale_stride) {
        std::vector<std::uint8_t> code_values(codes.count), scale_values(scales.count);
        for (std::size_t i = 0; i < codes.count; ++i) {
            const auto pattern = i + i / 37U + i / code_stride;
            code_values[i] = static_cast<std::uint8_t>(Operand::kNvfp4 ? pattern % 254U : 24U + pattern % 17U);
        }
        for (std::size_t i = 0; i < scales.count; ++i)
            scale_values[i] = static_cast<std::uint8_t>(Operand::kNvfp4 ? 31U + i % 11U : 122U + i % 3U);
        codes.write(code_values);
        scales.write(scale_values);
    }

    packed::PackedMatrix matrix(unsigned expert, packed::RowOrder order) const {
        return {codes.data + expert * code_stride, scales.data + expert * scale_stride, rows, width,
                static_cast<float>(expert + 1U) / 181.0F, order};
    }
};

struct Route {
    static constexpr std::uint32_t id_guard = 0x98765432U;
    static constexpr float weight_guard = -0x1p+120F;
    Device<std::uint32_t> ids{b8::top_k + 1U};
    Device<float> weights{b8::top_k + 1U};
    Device<std::uint32_t> groups{b8::top_k + 1U};

    void reset() {
        ids.fill(id_guard);
        weights.fill(weight_guard);
        groups.fill(id_guard);
    }
};

void compare_route(const Route& candidate, const Route& reference, bool tied) {
    const auto ids = candidate.ids.read();
    const auto weights = candidate.weights.read();
    const auto groups = candidate.groups.read();
    same(ids, reference.ids.read(), "fused route IDs");
    same(weights, reference.weights.read(), "fused route weight bits");
    require(ids.back() == Route::id_guard && weights.back() == Route::weight_guard &&
            groups.back() == Route::id_guard, "route guards");
    std::array<unsigned, b8::top_k> slots;
    std::iota(slots.begin(), slots.end(), 0U);
    std::ranges::sort(slots, [&ids](auto a, auto b) { return ids[a] < ids[b]; });
    for (unsigned rank = 0; rank < b8::top_k; ++rank) {
        const auto slot = slots[rank];
        require(ids[slot] < b8::experts && groups[slot] == rank, "compact group order");
    }
    if (tied)
        require(std::ranges::equal(b8::tied_ids, std::span{ids}.first(b8::top_k)) &&
                std::ranges::equal(b8::tied_groups, std::span{groups}.first(b8::top_k)), "ascending ties");
}

template <MatrixFormat Format>
void run() {
    using ExpertWeights = strix::moe_core::device_grouped::ExpertWeights;
    constexpr unsigned intermediate = Format == MatrixFormat::nvfp4 ? 144U : 160U;
    constexpr std::uint16_t activation_guard = 0x7654U;
    constexpr float output_guard = -0x1p+120F;
    using Dot = core::RouteStridedFmaXor32;

    const Matrices<Format> gate_up(2U * intermediate, b8::hidden);
    const Matrices<Format> down(b8::hidden, intermediate);
    std::array<ExpertWeights, b8::experts> directory;
    for (unsigned e = 0; e < b8::experts; ++e)
        directory[e] = {gate_up.matrix(e, packed::RowOrder::interleaved_halves), down.matrix(e, packed::RowOrder::contiguous)};
    Device<ExpertWeights> matrices(b8::experts);
    matrices.write(directory);

    Device<float> input(b8::hidden), router_weights(b8::experts * b8::hidden), bias(b8::experts);
    Device<float> scores(b8::experts), choices(b8::experts);
    Device<float> actual_output(b8::hidden + 1U), ref_output(b8::hidden + 1U);
    Device<std::uint16_t> actual_activation(b8::top_k * intermediate + 1U);
    Device<std::uint16_t> ref_activation(b8::top_k * intermediate + 1U);
    Route reference, candidate;
    const strix::moe::RouteScorePlanes<1U> planes{scores.data, choices.data};

    for (bool tied : {false, true}) {
        const auto router = b8::router_inputs(tied);
        input.write(router.input);
        router_weights.write(router.weights);
        bias.write(router.bias);
        reference.reset();
        ref_activation.fill(activation_guard);
        ref_output.fill(output_guard);
        check(core::launch_route_sigmoid_projection<b8::hidden, b8::experts, Dot, b8::Math>(
            input.data, router_weights.data, bias.data, 1U, planes, nullptr));
        check(core::launch_route_score_planes<b8::Policy>(planes, 1U, reference.ids.data, reference.weights.data, nullptr));
        const r1::Args incumbent{matrices.data, reference.ids.data, reference.weights.data, nullptr, input.data,
                                 ref_activation.data, ref_output.data, b8::experts, b8::hidden, intermediate, 4.0F, 9.0F};
        check(r1::launch_gate_up<Format, b8::top_k>(incumbent, nullptr));
        check(r1::launch_down_combine<Format, b8::top_k>(incumbent, nullptr));
        const auto expected_activation = ref_activation.read();
        const auto expected_output = ref_output.read();
        require(expected_activation.back() == activation_guard && expected_output.back() == output_guard,
                "expert write guards");
        require(std::ranges::find(expected_activation, activation_guard) ==
                expected_activation.end() - 1, "incumbent activation left an unwritten cell");
        require(std::ranges::find(expected_output, output_guard) ==
                expected_output.end() - 1, "incumbent output left an unwritten cell");

        for (auto limits : {plan_r1::ColumnLimits{1U, 1U}, {2U, 4U}, {4U, 2U}}) {
            const auto plan = plan_r1::make_layer_plan(Format, b8::hidden, intermediate, limits).value();
            candidate.reset();
            actual_activation.fill(activation_guard);
            actual_output.fill(output_guard);
            check(fused::launch_route_plan<b8::hidden, b8::Policy, b8::Math>(plan, input.data, router_weights.data,
                bias.data, {candidate.ids.data, candidate.weights.data, candidate.groups.data}, nullptr));
            const r1::Args args{matrices.data, candidate.ids.data, candidate.weights.data, nullptr, input.data,
                                actual_activation.data, actual_output.data, b8::experts, b8::hidden, intermediate, 4.0F, 9.0F};
            check(fused::launch_experts<b8::top_k>(plan, args, candidate.groups.data, nullptr));
            compare_route(candidate, reference, tied);
            same(actual_activation.read(), expected_activation, "gate/up activation bits");
            same(actual_output.read(), expected_output, "down epilogue output bits");
        }
    }
    std::cout << "B8 CUDA " << (Format == MatrixFormat::nvfp4 ? "NVFP4" : "MXFP8")
              << " route, groups, activation and down bitwise, columns 1/2/4: PASS\n";
}

}  // namespace

int main() {
    try {
        run<MatrixFormat::nvfp4>();
        run<MatrixFormat::mxfp8>();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
