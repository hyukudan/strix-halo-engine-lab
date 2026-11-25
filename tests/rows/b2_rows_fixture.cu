// Operator fixture for the existing MXFP8 body with declared split trees.
#include "core/rows_plan.hpp"
#include "cuda/core/dense/rows_fp4.cuh"
#include "strix/cuda_core/nvfp4.hpp"

#include <bit>
#include <cstdio>

namespace cc = strix::cuda_core;
namespace rows = strix::core::rows;
namespace weight = strix::weight;

namespace {

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template<class T>
using Buffer = cc::UniqueHandle<T*, cudaFree>;

template<class T>
Buffer<T> allocate(std::size_t count) {
    Buffer<T> buffer;
    if (count != 0) check(cudaMalloc(buffer.put(), count * sizeof(T)));
    return buffer;
}

template<class T>
Buffer<T> upload(const std::vector<T>& data) {
    auto buffer = allocate<T>(data.size());
    check(cudaMemcpy(buffer.get(), data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    return buffer;
}

void expect_bits(float actual, float expected) {
    if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected))
        throw std::runtime_error("MXFP8 result bits differ");
}

struct Fixture {
    std::uint32_t out, in;
    std::vector<std::uint8_t> codes, scales;
    std::vector<float> x;
    Buffer<std::uint8_t> device_codes, device_scales;
    Buffer<float> device_x, device_y;

    Fixture(std::uint32_t outputs, std::uint32_t chunks, std::uint32_t count)
        : out(outputs), in(chunks * rows::kBlockIn), codes(std::size_t{out} * in),
          scales(std::size_t{out} * chunks), x(std::size_t{count} * in) {
        constexpr std::array<std::uint8_t, 4> weights{0x40, 0x38, 0xb8, 0x3c};
        constexpr std::array values{0x1p26f, 1.0f, -0x1p26f, 1.0f, -0.25f, 2.0f, 4.0f};
        for (std::uint32_t o = 0; o < out; ++o) {
            for (std::uint32_t k = 0; k < in; ++k)
                codes[std::size_t{o} * in + k] = weights[(o + k) % weights.size()];
            for (std::uint32_t c = 0; c < chunks; ++c)
                scales[std::size_t{o} * chunks + c] = static_cast<std::uint8_t>(126U + (o + c) % 3U);
        }
        for (std::uint32_t r = 0; r < count; ++r)
            for (std::uint32_t c = 0; c < chunks; ++c) {
                x[std::size_t{r} * in + c * rows::kBlockIn] = values[(r + c) % values.size()];
                x[std::size_t{r} * in + c * rows::kBlockIn + 9] = values[(r + c + 3U) % values.size()];
            }
        device_codes = upload(codes);
        device_scales = upload(scales);
        device_x = upload(x);
        device_y = allocate<float>(std::size_t{count} * out);
    }

    rows::Job job() const {
        return {{device_codes.get(), rows::kMxfp8Rows, device_scales.get()}, out, in,
                {device_x.get()}, {device_y.get()}, 0, {}};
    }

    std::vector<float> result(std::uint32_t count) const {
        std::vector<float> result(std::size_t{count} * out);
        check(cudaMemcpy(result.data(), device_y.get(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    }

    void compare(const rows::SplitTree& tree, std::uint32_t count, std::span<const float> incumbent) const {
        const auto actual = result(count);
        for (std::uint32_t r = 0; r < count; ++r) {
            for (std::uint32_t o = 0; o < out; ++o) {
                const auto index = std::size_t{r} * out + o;
                const auto expected = rows::mxfp8_rows_reference(tree, std::span{codes}.subspan(std::size_t{o} * in, in),
                    std::span{scales}.subspan(std::size_t{o} * tree.chunks, tree.chunks),
                    std::span{x}.subspan(std::size_t{r} * in, in));
                expect_bits(actual[index], expected);
                if (!incumbent.empty()) expect_bits(actual[index], incumbent[index]);
            }
        }
    }
};

void run(std::uint32_t count) {
    Fixture a(64, 7, count);
    Fixture b(128, 9, count);
    constexpr std::array formats{rows::RoleFormat{weight::Role::attention_q, rows::kMxfp8Rows},
                                 rows::RoleFormat{weight::Role::attention_k, rows::kMxfp8Rows}};
    const std::array jobs{a.job(), b.job()};
    strix::device::Profile device{};
    device.units = 1; // Synthetic planner facts make split coverage deterministic.
    const auto legacy = rows::plan({jobs, count, 0, rows::Packing::fewest_launches}, device);
    const auto old_bytes = cc::rows_split_k_workspace(legacy);
    const auto old_scratch = allocate<float>(old_bytes / sizeof(float));
    check(cc::cuda_rows_grouped(legacy, old_scratch.get(), old_bytes, nullptr));
    const auto old_a = a.result(count), old_b = b.result(count);

    std::array bound{rows::RoleJob{formats[0].role, jobs[0]}, rows::RoleJob{formats[1].role, jobs[1]}};
    for (const auto splits : {0U, 1U, 3U, legacy.launches()[0].splits[0]}) {
        const bool replay = splits == legacy.launches()[0].splits[0];
        bound[0].splits = splits;
        bound[1].splits = replay ? legacy.launches()[0].splits[1] : splits;
        const auto plan = rows::plan_rows(bound, formats, count, device, {65535});
        if (plan.launches().size() != 1) throw std::runtime_error("expected one grouped MXFP8 launch");
        const auto bytes = cc::mxfp8_rows_workspace(plan, 0);
        const auto scratch = allocate<float>(bytes / sizeof(float));
        if (bytes != 0 && cc::cuda_mxfp8_rows(plan, 0, scratch.get(), bytes - 1, nullptr) != cudaErrorInvalidValue)
            throw std::runtime_error("short workspace accepted");
        check(cc::cuda_mxfp8_rows(plan, 0, scratch.get(), bytes, nullptr));
        const auto& trees = plan.launches()[0].trees;
        a.compare(trees[0], count, replay ? std::span<const float>{old_a} : std::span<const float>{});
        b.compare(trees[1], count, replay ? std::span<const float>{old_b} : std::span<const float>{});
    }
    std::printf("PASS B2 CUDA rows M=%u declared=bitwise incumbent=bitwise\n", count);
}

}  // namespace

int main() {
    try {
        for (auto count : {1U, 2U, 4U, 8U, 16U, 32U}) run(count);
        std::puts("PASS B2 CUDA rows");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL B2 CUDA rows: %s\n", error.what());
        return 1;
    }
    return 0;
}
