#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/resident_q8_rows.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
namespace core = strix::core;
namespace staging = core::q8_f16_staging;
namespace weight = strix::weight;
using strix::hip_check;

template <class Group>
constexpr bool check_projection_panels() {
    constexpr unsigned largest = std::numeric_limits<unsigned>::max() - 31U;
    constexpr unsigned largest_panels = std::numeric_limits<unsigned>::max() / 64U + 1U;
    using Projection = std::remove_cvref_t<decltype(Group{}.projections[0])>;
    const auto projection = [](unsigned outputs) {
        Projection p{};  // every field value-initialised (Bf16Projection::round_bf16 = false)
        p.outputs = outputs;
        return p;
    };
    Group group{};
    group.append(projection(64U));
    group.append(projection(largest));
    group.append(projection(65U));
    return group.count == 3U && group.first_panel[0] == 0U && group.first_panel[1] == 1U &&
        group.first_panel[2] == 1U + largest_panels && group.first_panel[3] == 3U + largest_panels;
}

static_assert(check_projection_panels<core::f16_tile_gemm::Q8ProjectionGroup<3>>());
static_assert(check_projection_panels<core::f16_tile_gemm::Bf16ProjectionGroup<3>>());

template <class T>
struct Buffer {
    T* data{};
    explicit Buffer(std::size_t size) { hip_check(hipMalloc(&data, size * sizeof(T)), "allocate"); }
    ~Buffer() { static_cast<void>(hipFree(data)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void upload(std::span<const T> values) const {
        hip_check(hipMemcpy(data, values.data(), values.size_bytes(), hipMemcpyHostToDevice), "upload");
    }
    void download(std::span<T> values) const {
        hip_check(hipMemcpy(values.data(), data, values.size_bytes(), hipMemcpyDeviceToHost), "download");
    }
};

float half_value(std::uint16_t bits) { return static_cast<float>(std::bit_cast<_Float16>(bits)); }

__global__ void check_staging_kernel(const std::uint8_t* blocks, std::uint32_t* output) {
    const unsigned lane = threadIdx.x;
    const auto* block = blocks + (blockIdx.x * 16U + lane / 2U) * weight::Q8Storage::bytes;
    const auto raw = staging::Q8F16Staging::load(block, 0U, lane & 1U);
    staging::Q8F16Staging::publish(raw, lane, output + (blockIdx.x * 32U + lane) * 8U);
}

std::vector<std::uint8_t> make_weights(std::size_t blocks, std::mt19937& rng,
                                       std::span<const std::uint16_t> scales) {
    std::vector<std::uint8_t> bytes(blocks * weight::Q8Storage::bytes);
    for (std::size_t i = 0; i < blocks; ++i) {
        auto* block = bytes.data() + i * weight::Q8Storage::bytes;
        const auto scale = scales[i % scales.size()];
        block[0] = static_cast<std::uint8_t>(scale);
        block[1] = static_cast<std::uint8_t>(scale >> 8U);
        for (unsigned k = 0; k < weight::Q8Storage::elements; ++k)
            block[weight::Q8Storage::scale_bytes + k] = static_cast<std::uint8_t>(rng());
        block[2] = 0;
        block[3] = 128;
        block[18] = 127;
        block[19] = 255;
    }
    return bytes;
}

void check_staging(std::mt19937& rng) {
    constexpr unsigned blocks = 128U;
    constexpr std::array<std::uint16_t, 14> scales{0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400,
                                                  0x8400, 0x3c01, 0xbc01, 0x7bff, 0xfbff, 0x7c00, 0xfc00};
    const auto bytes = make_weights(blocks, rng, scales);
    std::vector<std::uint32_t> staged(blocks * 16U);
    const Buffer<std::uint8_t> weights(bytes.size());
    const Buffer<std::uint32_t> output(staged.size());
    weights.upload(bytes);
    STRIX_HIP_LAUNCH(Rdna3Layout, check_staging_kernel<<<blocks / 16U, 32U>>>(weights.data, output.data));
    hip_check(hipGetLastError(), "staging launch");
    output.download(staged);
    const auto nan = [](std::uint16_t bits) { return (bits & 0x7fffU) > 0x7c00U; };
    for (unsigned i = 0; i < blocks; ++i) {
        const auto* block = bytes.data() + std::size_t{i} * weight::Q8Storage::bytes;
        const float scale = half_value(weight::Q8Storage::scale_bits(block));
        for (unsigned k = 0; k < 32U; ++k) {
            const auto ref = std::bit_cast<std::uint16_t>(static_cast<_Float16>(
                static_cast<float>(weight::Q8Storage::code(block, k)) * scale));
            const auto got = static_cast<std::uint16_t>(staged[i * 16U + k / 2U] >> (16U * (k & 1U)));
            if (got != ref && !(nan(got) && nan(ref))) throw std::runtime_error("packed FP16 staging differs");
        }
    }
    std::cout << "PASS resident Q8_0 device staging bits\n";
}

void check_projection(std::mt19937& rng) {
    constexpr unsigned rows = 2048U, short_rows = 17U, width = 96U;
    constexpr std::size_t stride = weight::Q8Storage::row_bytes(width);
    constexpr std::array<std::uint16_t, 5> scales{0x1601, 0x9807, 0x1902, 0x1a0f, 0x1b01};
    constexpr unsigned first_n = 137U, second_n = 67U;
    constexpr std::array outputs{first_n, second_n};
    constexpr float sentinel = -1234567.0F;
    std::vector<float> input(std::size_t{rows} * width);
    for (auto& value : input) value = static_cast<float>(static_cast<int>(rng() % 65521U) - 32760) * 0x1p-14F;
    const Buffer<float> x(input.size());
    x.upload(input);

    const std::array bytes{make_weights(first_n * width / 32U, rng, scales),
                           make_weights(second_n * width / 32U, rng, scales)};
    const Buffer<std::uint8_t> wa(bytes[0].size()), wb(bytes[1].size());
    wa.upload(bytes[0]);
    wb.upload(bytes[1]);
    const std::array matrices{
        weight::ResidentMatrixView{{wa.data, bytes[0].size()}, weight::kQ8WeightStorage, first_n, width, stride},
        weight::ResidentMatrixView{{wb.data, bytes[1].size()}, weight::kQ8WeightStorage, second_n, width, stride}};
    const auto first_size = std::size_t{rows + 1U} * first_n;
    const auto second_size = std::size_t{rows + 1U} * second_n;
    const Buffer<float> ya(first_size), yb(second_size);
    const Buffer<float> sa((short_rows + 1U) * first_n), sb((short_rows + 1U) * second_n);
    const Buffer<float> solo((short_rows + 1U) * first_n), tight(16U * first_n);
    const std::vector<float> guards(first_size, sentinel);
    ya.upload(guards);
    yb.upload(std::span(guards).first(second_size));
    sa.upload(std::span(guards).first((short_rows + 1U) * first_n));
    sb.upload(std::span(guards).first((short_rows + 1U) * second_n));

    const std::array long_group{core::ResidentRowProjection{matrices[0], first_n, ya.data},
                                core::ResidentRowProjection{matrices[1], second_n, yb.data}};
    const std::array short_group{core::ResidentRowProjection{matrices[0], first_n, sa.data},
                                 core::ResidentRowProjection{matrices[1], second_n, sb.data}};
    hip_check(core::enqueue_resident_q8_projections<2>(x.data, width, rows, long_group, nullptr), "2048 rows");
    hip_check(core::enqueue_resident_q8_projections<2>(x.data, width, short_rows, short_group, nullptr), "17 rows");
    hip_check(core::enqueue_resident_q8_rows(matrices[0], first_n, x.data, short_rows, solo.data, nullptr), "solo tile");
    std::vector<float> solo_rows(short_rows * first_n);
    solo.download(solo_rows);

    const std::array full_buffers{&ya, &yb};
    const std::array short_buffers{&sa, &sb};
    std::vector<double> activation(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) activation[i] = static_cast<double>(static_cast<_Float16>(input[i]));
    for (unsigned p = 0; p < outputs.size(); ++p) {
        const auto n = outputs[p];
        std::vector<float> full((rows + 1U) * n), small((short_rows + 1U) * n);
        full_buffers[p]->download(full);
        short_buffers[p]->download(small);
        if (!std::ranges::all_of(std::span(full).last(n), [](float value) { return value == sentinel; }) ||
            !std::ranges::all_of(std::span(small).last(n), [](float value) { return value == sentinel; }))
            throw std::runtime_error("stray output store");
        const auto same_bits = [](float a, float b) {
            return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
        };
        small.resize(short_rows * n);
        if (!std::equal(small.begin(), small.end(), full.begin(), same_bits) ||
            (p == 0U && !std::equal(small.begin(), small.end(), solo_rows.begin(), same_bits)))
            throw std::runtime_error("17 and 2048 rows or grouped and solo differ");

        std::vector<double> operands(std::size_t{n} * width);
        for (unsigned o = 0; o < n; ++o) {
            const auto* row = bytes[p].data() + o * stride;
            for (unsigned k = 0; k < width; ++k) {
                const auto* block = weight::Q8Storage::block(row, k / 32U);
                const float w = half_value(weight::Q8Storage::scale_bits(block)) *
                    static_cast<float>(weight::Q8Storage::code(block, k % 32U));
                operands[o * width + k] = static_cast<double>(static_cast<_Float16>(w));
            }
        }
        for (unsigned r = 0; r < rows; ++r) {
            for (unsigned o = 0; o < n; ++o) {
                double sum = 0.0, l1 = 0.0;
                for (unsigned k = 0; k < width; ++k) {
                    const double term = activation[r * width + k] * operands[o * width + k];
                    sum += term;
                    l1 += std::abs(term);
                }
                // FP16 products fit in FP32. Use a conservative FP32 reduction
                // bound (2*K unit roundoffs) scaled by the absolute product sum.
                const double bound = 2.0 * width * 0x1p-24 * l1 + 1e-7;
                const float got = full[r * n + o];
                if (!std::isfinite(got) || std::abs(static_cast<double>(got) - sum) > bound)
                    throw std::runtime_error("FP64 projection bound exceeded");
            }
        }
    }
    std::cout << "PASS resident Q8_0 tile FP64 bound and row-count invariance\n";

    hip_check(core::enqueue_resident_q8_rows(matrices[0], first_n, x.data, 16U, solo.data, nullptr), "tight entry");
    core::detail::launch_q8_tight_rows<16>({wa.data, width, first_n, stride}, x.data, tight.data, 16U, nullptr);
    hip_check(hipGetLastError(), "tight reference launch");
    std::vector<std::uint32_t> primary(16U * first_n), baseline(16U * first_n);
    hip_check(hipMemcpy(primary.data(), solo.data, primary.size() * sizeof(float), hipMemcpyDeviceToHost));
    hip_check(hipMemcpy(baseline.data(), tight.data, baseline.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (primary != baseline) throw std::runtime_error("16-row dispatch differs from the tight kernel");
    std::cout << "PASS resident Q8_0 16-row tight identity\n";
}

void bench_projection(std::mt19937& rng) {
    constexpr unsigned rows = 2048U, width = 2304U, n = 3072U;
    constexpr std::size_t stride = weight::Q8Storage::row_bytes(width);
    constexpr std::array<std::uint16_t, 3> scales{0x1601, 0x9807, 0x1902};
    const auto bytes = make_weights(n * width / 32U, rng, scales);
    std::vector<float> input(std::size_t{rows} * width);
    for (auto& value : input) value = static_cast<float>(static_cast<int>(rng() % 1021U) - 510) * 0x1p-8F;
    const Buffer<std::uint8_t> weights(bytes.size());
    const Buffer<float> x(input.size()), y(std::size_t{rows} * n);
    weights.upload(bytes);
    x.upload(input);
    const weight::ResidentMatrixView matrix{{weights.data, bytes.size()}, weight::kQ8WeightStorage, n, width, stride};
    const core::dense_linear::View tight{weights.data, width, n, stride};
    const auto launch_tile = [&] {
        hip_check(core::enqueue_resident_q8_rows(matrix, n, x.data, rows, y.data, nullptr));
    };
    const auto launch_sweeps = [&] {
        for (unsigned row = 0; row < rows; row += core::gemv_rows::max_rows) {
            core::detail::launch_q8_tight_rows<core::gemv_rows::max_rows>(tight, x.data + std::size_t{row} * width,
                y.data + std::size_t{row} * n, core::gemv_rows::max_rows, nullptr);
            hip_check(hipGetLastError());
        }
    };
    hipEvent_t begin{}, end{};
    hip_check(hipEventCreate(&begin));
    hip_check(hipEventCreate(&end));
    const auto time = [&](const auto& launch) {
        hip_check(hipEventRecord(begin, nullptr));
        launch();
        hip_check(hipEventRecord(end, nullptr));
        hip_check(hipEventSynchronize(end));
        float ms{};
        hip_check(hipEventElapsedTime(&ms, begin, end));
        return ms;
    };
    time(launch_tile);
    time(launch_sweeps);
    std::array<float, 7> tile_ms{}, sweep_ms{};
    for (unsigned i = 0; i < tile_ms.size(); ++i) {
        if ((i & 1U) == 0U) {
            sweep_ms[i] = time(launch_sweeps);
            tile_ms[i] = time(launch_tile);
        } else {
            tile_ms[i] = time(launch_tile);
            sweep_ms[i] = time(launch_sweeps);
        }
    }
    hip_check(hipEventDestroy(end));
    hip_check(hipEventDestroy(begin));
    std::ranges::sort(tile_ms);
    std::ranges::sort(sweep_ms);
    int ordinal{};
    hipDeviceProp_t device{};
    hip_check(hipGetDevice(&ordinal));
    hip_check(hipGetDeviceProperties(&device, ordinal));
    std::cout << "FAST resident Q8_0 synthetic 2048x2304x3072 " << device.gcnArchName
              << " median ms tight=" << sweep_ms[sweep_ms.size() / 2U]
              << " tile=" << tile_ms[tile_ms.size() / 2U] << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::mt19937 rng(20261122U);
        check_staging(rng);
        check_projection(rng);
        if (argc == 2 && std::string_view(argv[1]) == "--bench") bench_projection(rng);
    } catch (const std::exception& error) {
        std::cerr << "FAIL resident Q8_0 tile: " << error.what() << '\n';
        return 1;
    }
}
