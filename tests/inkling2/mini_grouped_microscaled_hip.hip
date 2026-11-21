#include "inkling2/inkling2_mini_expert_hip.hpp"
#include "moe_core/grouped_microscaled.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>

namespace {

namespace grouped = strix::moe_core::grouped_microscaled;
namespace codec = strix::weight::scalar_float;
namespace packed = strix::weight::microscaled;
constexpr unsigned inputs = 64, outputs = 256;
constexpr unsigned input_stride = inputs + 8U, output_stride = outputs + 7U;
constexpr unsigned capacity = grouped::gemm::kBM;
constexpr std::array active_rows{1U, 19U, capacity};
constexpr float sentinel = -654321.0F;

void require_hip(hipError_t error) {
    if (error == hipSuccess) return;
    std::cerr << hipGetErrorString(error) << '\n';
    std::exit(1);
}

template <packed::MatrixFormat Format>
struct Data {
    static constexpr bool nvfp4 = Format == packed::MatrixFormat::nvfp4;
    static constexpr unsigned code_bytes = nvfp4 ? inputs / 2U : inputs;
    static constexpr unsigned scale_bytes = inputs / (nvfp4 ? 16U : 32U);
    std::array<std::uint8_t, outputs * code_bytes> codes{};
    std::array<std::uint8_t, outputs * scale_bytes> scales{};
    alignas(16) std::array<std::uint16_t, capacity * input_stride> input{};
    std::array<float, capacity * output_stride> output{};
};

template <packed::MatrixFormat Format>
struct Fixture {
    std::array<Data<Format>, active_rows.size()> data{};
    std::array<grouped::TileRows, active_rows.size()> tiles{};
};

template <packed::MatrixFormat Format>
bool run(hipStream_t stream) {
    constexpr bool nvfp4 = Data<Format>::nvfp4;
    constexpr unsigned code_bytes = Data<Format>::code_bytes, scale_bytes = Data<Format>::scale_bytes;
    constexpr std::array<std::uint8_t, 8> fp8{0, 0xb0, 0x38, 0xb8, 0x40, 0xc0, 0x30, 0x80};
    constexpr std::array<std::uint8_t, 3> fp4_scales{0x30, 0x38, 0x40};
    constexpr std::array<float, 3> outer{0.5F, 2.0F, 0.25F};

    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, sizeof(Fixture<Format>)));
    auto* fixture = std::construct_at(static_cast<Fixture<Format>*>(storage));
    for (std::size_t group = 0; group < fixture->data.size(); ++group) {
        auto& data = fixture->data[group];
        for (std::size_t i = 0; i < data.codes.size(); ++i) {
            const auto seed = i * 17U + i / 3U + i / code_bytes + group;
            data.codes[i] = nvfp4 ? static_cast<std::uint8_t>(seed % 256U) : fp8[seed % fp8.size()];
        }
        for (std::size_t i = 0; i < data.scales.size(); ++i) {
            const auto scale = (i + i / scale_bytes + group) % 3U;
            data.scales[i] = nvfp4 ? fp4_scales[scale] : static_cast<std::uint8_t>(126U + scale);
        }
        for (std::size_t i = 0; i < data.input.size(); ++i) {
            const int value = static_cast<int>((i * 17U + i / 3U + group) % 13U) - 6;
            data.input[i] = codec::bf16_encode(static_cast<float>(value) * 0.125F);
        }
        std::ranges::fill(data.output, sentinel);
        auto& tile = fixture->tiles[group];
        tile = {{data.codes.data(), data.scales.data(), outputs, inputs, nvfp4 ? outer[group] : 1.0F,
                 group % 2U == 0U ? packed::RowOrder::interleaved_halves : packed::RowOrder::contiguous},
                data.input.data(), data.output.data(), input_stride, output_stride, active_rows[group]};
        if (!grouped::fits_tile<Format>(tile, inputs, outputs)) {
            std::cerr << "Grouped fixture tile geometry refused\n";
            std::exit(1);
        }
    }
    require_hip(grouped::launch<Format>({fixture->tiles.data(), static_cast<unsigned>(fixture->tiles.size()),
                                        inputs, outputs}, stream));
    require_hip(hipStreamSynchronize(stream));

    bool passed = true;
    std::array<float, inputs> weight{};
    for (std::size_t group = 0; group < fixture->data.size(); ++group) {
        const auto& data = fixture->data[group];
        for (unsigned col = 0; col < output_stride; ++col) {
            if (col < outputs) {
                const auto physical = group % 2U == 0U ? 2U * (col % (outputs / 2U)) + col / (outputs / 2U) : col;
                const auto codes = std::span(data.codes).subspan(physical * code_bytes, code_bytes);
                const auto scales = std::span(data.scales).subspan(physical * scale_bytes, scale_bytes);
                if constexpr (nvfp4) packed::decode_nvfp4_row(codes, scales, outer[group], 1.0F, weight);
                else packed::decode_mxfp8_row(codes, scales, weight);
            }
            for (unsigned row = 0; row < capacity; ++row) {
                float expected = sentinel;
                if (row < active_rows[group] && col < outputs) {
                    expected = 0.0F;
                    for (unsigned k = 0; k < inputs; ++k) {
                        const auto activation = static_cast<std::uint32_t>(data.input[row * input_stride + k]) << 16U;
                        expected += codec::u2f(activation) * weight[k];
                    }
                }
                passed = passed && data.output[row * output_stride + col] == expected;
            }
        }
    }
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    return passed;
}

template <bool Nvfp4>
bool run_mini_down(hipStream_t stream) {
    using namespace strix::inkling2;
    constexpr auto projection = MiniExpertProjection::down;
    constexpr auto geometry = mini_expert_geometry(projection);
    constexpr unsigned stored_columns = Nvfp4 ? geometry.inputs / 2U : geometry.inputs;
    constexpr unsigned scale_columns = Nvfp4 ? geometry.inputs / 16U : geometry.inputs / 32U;
    constexpr unsigned physical_layer = Nvfp4 ? kMxFp8First - 1U : kMxFp8First;
    struct Buffers {
        std::array<std::uint8_t, geometry.outputs * stored_columns> codes{};
        std::array<std::uint8_t, geometry.outputs * scale_columns> scales{};
        alignas(16) std::array<std::uint16_t, geometry.inputs> input{};
        std::array<float, geometry.outputs> output{};
        grouped::TileRows tile;
    };
    void* memory = nullptr;
    require_hip(hipMallocManaged(&memory, sizeof(Buffers)));
    auto* b = std::construct_at(static_cast<Buffers*>(memory));
    for (unsigned i = 0; i < geometry.inputs; ++i)
        b->input[i] = codec::bf16_encode(static_cast<float>(i % 11U) * 0.25F);
    b->scales.fill(Nvfp4 ? 0x38U : 127U);
    for (unsigned row = 0; row < geometry.outputs; ++row) {
        const auto k = row % geometry.inputs;
        const auto offset = row * stored_columns + (Nvfp4 ? k / 2U : k);
        b->codes[offset] = Nvfp4 ? static_cast<std::uint8_t>(2U << (4U * (k % 2U))) : 0x38U;
    }
    b->tile = {{b->codes.data(), b->scales.data(), geometry.outputs, geometry.inputs, 1.0F,
                packed::RowOrder::contiguous}, b->input.data(), b->output.data(), geometry.inputs, geometry.outputs, 1};
    const bool admitted = fits_mini_hip_expert_tile(physical_layer, projection, b->tile) &&
                          !fits_mini_hip_expert_tile(physical_layer, MiniExpertProjection::gate_up, b->tile);
    require_hip(enqueue_mini_hip_expert_projection(physical_layer, projection, &b->tile, 1, stream));
    require_hip(hipStreamSynchronize(stream));
    bool match = true;
    for (unsigned row = 0; row < geometry.outputs; ++row) {
        const auto bits = std::uint32_t{b->input[row % geometry.inputs]} << 16U;
        match = match && b->output[row] == codec::u2f(bits);
    }
    std::destroy_at(b);
    require_hip(hipFree(memory));
    return admitted && match;
}

}  // namespace

int main() {
    hipStream_t stream{};
    require_hip(hipStreamCreate(&stream));
    const bool pass = run<packed::MatrixFormat::nvfp4>(stream) && run<packed::MatrixFormat::mxfp8>(stream) &&
                      run_mini_down<true>(stream) && run_mini_down<false>(stream);
    const bool rejects = grouped::launch<packed::MatrixFormat::nvfp4>({}, stream) == hipErrorInvalidValue;
    require_hip(hipStreamDestroy(stream));
    if (!pass) std::cerr << "Grouped BF16 microscaled HIP GEMM differs from the reference\n";
    return pass && rejects ? 0 : 1;
}
