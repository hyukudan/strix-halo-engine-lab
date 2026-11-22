#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "inkling2/inkling2_mini_expert_hip.hpp"
#include "moe_core/grouped_microscaled.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
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
    float worst = 0.0F;
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
                float magnitude = 0.0F;
                if (row < active_rows[group] && col < outputs) {
                    expected = 0.0F;
                    for (unsigned k = 0; k < inputs; ++k) {
                        const auto activation = static_cast<std::uint32_t>(data.input[row * input_stride + k]) << 16U;
                        const float term = codec::u2f(activation) * weight[k];
                        expected += term;
                        magnitude += std::abs(term);
                    }
                }
                const float got = data.output[row * output_stride + col];
                // Guard rows and columns stay bitwise untouched. gfx11 WMMA
                // accumulates each K16 step in its own hardware order and may
                // differ from the serial FP32 sum by a few units in the last
                // place of the accumulated magnitude; a wrong operand, scale
                // or row order moves a result by at least one whole term.
                const float bound = 0x1p-18F * magnitude;
                const float error = std::abs(got - expected);
                const bool ok = expected == sentinel ? got == sentinel : error <= bound;
                if (magnitude > 0.0F) worst = std::max(worst, error / magnitude);
                if (!ok && passed) {
                    std::cerr << (nvfp4 ? "nvfp4" : "mxfp8") << " group " << group << " row " << row << " col " << col
                              << std::hexfloat << " got " << got << " expected " << expected << std::defaultfloat << '\n';
                }
                passed = passed && ok;
            }
        }
    }
    std::cout << (nvfp4 ? "nvfp4" : "mxfp8") << " worst error / sum|terms| " << worst << '\n';
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    return passed;
}

// Indirect input rows (Args::gather_rows): the same tiles read row r of their gathered copy as row map[first + r] of a
// token-row image. Outputs must be the bits of the run on the gathered copy; the copy is poisoned before the
// indirect run, so a tile that still read it would differ.
template <packed::MatrixFormat Format>
bool run_indirect(hipStream_t stream) {
    constexpr bool nvfp4 = Data<Format>::nvfp4;
    constexpr unsigned code_bytes = Data<Format>::code_bytes, scale_bytes = Data<Format>::scale_bytes;
    constexpr unsigned image_rows = 160U, assigned = 1U + 19U + capacity;
    struct Buffers {
        std::array<std::uint8_t, outputs * code_bytes> codes{};
        std::array<std::uint8_t, outputs * scale_bytes> scales{};
        alignas(16) std::array<std::uint16_t, image_rows * input_stride> image{};
        alignas(16) std::array<std::uint16_t, assigned * input_stride> gathered{};
        std::array<std::uint32_t, assigned> map{};
        std::array<float, assigned * output_stride> direct{}, indirect{};
        std::array<grouped::TileRows, active_rows.size()> direct_tiles{}, indirect_tiles{};
    };
    void* memory = nullptr;
    require_hip(hipMallocManaged(&memory, sizeof(Buffers)));
    auto* b = std::construct_at(static_cast<Buffers*>(memory));
    for (std::size_t i = 0; i < b->codes.size(); ++i)
        b->codes[i] = static_cast<std::uint8_t>(nvfp4 ? (i * 29U + i / 5U) % 256U : 0x30U + (i * 7U) % 0x20U);
    for (std::size_t i = 0; i < b->scales.size(); ++i)
        b->scales[i] = static_cast<std::uint8_t>(nvfp4 ? 0x30U + 8U * (i % 3U) : 126U + i % 3U);
    for (std::size_t i = 0; i < b->image.size(); ++i)
        b->image[i] = codec::bf16_encode(static_cast<float>(static_cast<int>((i * 13U + i / 7U) % 29U) - 14) * 0.0625F);
    for (unsigned r = 0; r < assigned; ++r) {
        b->map[r] = (r * 37U + 11U) % image_rows;
        std::copy_n(b->image.data() + std::size_t{b->map[r]} * input_stride, input_stride,
                    b->gathered.data() + std::size_t{r} * input_stride);
    }
    b->direct.fill(sentinel);
    b->indirect.fill(sentinel);
    unsigned first = 0U;
    for (std::size_t t = 0; t < active_rows.size(); ++t) {
        const packed::PackedMatrix matrix{b->codes.data(), b->scales.data(), outputs, inputs, nvfp4 ? 0.5F : 1.0F,
                                          t % 2U == 0U ? packed::RowOrder::interleaved_halves : packed::RowOrder::contiguous};
        const auto* input = b->gathered.data() + std::size_t{first} * input_stride;
        b->direct_tiles[t] = {matrix, input, b->direct.data() + std::size_t{first} * output_stride, input_stride,
                              output_stride, active_rows[t]};
        // The plan addresses the gathered copy relative to its base; with the map, that base holds the image.
        b->indirect_tiles[t] = {matrix, b->image.data() + std::size_t{first} * input_stride,
                                b->indirect.data() + std::size_t{first} * output_stride, input_stride, output_stride,
                                active_rows[t]};
        first += active_rows[t];
    }
    require_hip(grouped::launch<Format>({b->direct_tiles.data(), static_cast<unsigned>(active_rows.size()), inputs, outputs},
                                        stream));
    require_hip(hipStreamSynchronize(stream));
    b->gathered.fill(0x7fc0U);
    grouped::IndirectArgs indirect{{b->indirect_tiles.data(), static_cast<unsigned>(active_rows.size()), inputs, outputs},
                                   b->map.data(), b->image.data()};
    grouped::IndirectArgs unmapped = indirect;
    unmapped.gather_rows = nullptr;
    const bool refuses = grouped::launch_indirect<Format>(unmapped, stream) == hipErrorInvalidValue;
    require_hip(grouped::launch_indirect<Format>(indirect, stream));
    require_hip(hipStreamSynchronize(stream));
    const bool equal = std::memcmp(b->direct.data(), b->indirect.data(), sizeof(b->direct)) == 0;
    bool written = false;
    for (float v : b->direct) written = written || v != sentinel;
    std::cout << (nvfp4 ? "nvfp4" : "mxfp8") << " indirect rows: outputs " << (equal ? "EQUAL" : "DIFF") << '\n';
    std::destroy_at(b);
    require_hip(hipFree(memory));
    return equal && written && refuses;
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
    if (!admitted || !match) std::cerr << "down projection " << (Nvfp4 ? "nvfp4" : "mxfp8") << " admitted " << admitted << " match " << match << '\n';
    return admitted && match;
}

}  // namespace

int main() {
    hipStream_t stream{};
    require_hip(hipStreamCreate(&stream));
    const bool nvfp4 = run<packed::MatrixFormat::nvfp4>(stream);
    const bool mxfp8 = run<packed::MatrixFormat::mxfp8>(stream);
    const bool down_nvfp4 = run_mini_down<true>(stream);
    const bool down_mxfp8 = run_mini_down<false>(stream);
    const bool indirect_nvfp4 = run_indirect<packed::MatrixFormat::nvfp4>(stream);
    const bool indirect_mxfp8 = run_indirect<packed::MatrixFormat::mxfp8>(stream);
    const bool pass = nvfp4 && mxfp8 && down_nvfp4 && down_mxfp8 && indirect_nvfp4 && indirect_mxfp8;
    const bool rejects = grouped::launch<packed::MatrixFormat::nvfp4>({}, stream) == hipErrorInvalidValue;
    require_hip(hipStreamDestroy(stream));
    if (!pass) std::cerr << "Grouped BF16 microscaled HIP GEMM differs from the reference\n";
    return pass && rejects ? 0 : 1;
}
