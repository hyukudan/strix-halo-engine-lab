#include "core/tall_chunk_policy.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace cc = strix::cuda_core;
namespace tall = strix::core::tall;

namespace strix::cuda_core {
cudaError_t dense_big_bf16_fixture(const DenseWeight&, const BigAct&, float*, cudaStream_t, BigCtx*);
}

namespace {

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "FAIL B4 %s\n", message);
    std::exit(1);
}

void check(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "FAIL B4 %s (%s)\n", operation, cudaGetErrorString(error));
    std::exit(1);
}

template <class T>
struct DeviceBuffer {
    T* data = nullptr;

    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "allocate buffer");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(data); }

    void upload(const std::vector<T>& src) const {
        check(cudaMemcpy(data, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "upload buffer");
    }
    void download(std::vector<T>& dest) const {
        check(cudaMemcpy(dest.data(), data, dest.size() * sizeof(T), cudaMemcpyDeviceToHost), "download buffer");
    }
};

constexpr std::size_t guard = 8;
constexpr std::uint32_t canary = 0x7fc13456U;

void compare(const std::vector<float>& actual, const std::vector<float>& expected, std::size_t elements) {
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto got = std::bit_cast<std::uint32_t>(actual[i]);
        const bool written = i >= guard && i < guard + elements;
        const auto want = written ? std::bit_cast<std::uint32_t>(expected[i]) : canary;
        if (got != want || (written && (want & 0x7f800000U) == 0x7f800000U)) {
            std::fprintf(stderr, "FAIL B4 output index %zu expected %08x got %08x\n", i, want, got);
            std::exit(1);
        }
    }
}

std::uint32_t mix(std::size_t index, std::uint32_t seed) {
    auto value = static_cast<std::uint32_t>(index) + seed;
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    return value ^ (value >> 15U);
}

void run(cc::BigCtx* context, std::uint32_t rows, std::uint32_t outputs, std::uint32_t width) {
    std::vector<float> input(std::size_t{rows} * width);
    std::vector<std::uint16_t> weights(std::size_t{outputs} * width);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto bits = mix(i, 119U);
        const auto value = static_cast<int>(bits % 65535U) - 32767;
        input[i] = static_cast<float>(value) * 0x1.23456p-17F;
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const auto value = static_cast<int>(mix(i, 263U) % 513U) - 256;
        weights[i] = strix::weight::scalar_float::bf16_encode(static_cast<float>(value) * 0x1p-8F);
    }

    const std::vector<float> initial(std::size_t{rows} * outputs + 2 * guard, std::bit_cast<float>(canary));
    auto expected = initial;
    auto actual = initial;
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_output(initial.size());
    DeviceBuffer<std::uint16_t> device_weights(weights.size());
    device_input.upload(input);
    device_weights.upload(weights);
    const cc::DenseWeight weight{device_weights.data, cc::WeightFormat::bf16, outputs, width};

    device_output.upload(initial);
    cc::BigAct activation;
    check(cc::dense_big_split(device_input.data, rows, width, weight.format, 3, &activation, nullptr, context),
          "split incumbent planes");
    check(cc::dense_big_gemm(weight, activation, device_output.data + guard, nullptr, cc::BigEpi::none,
                             nullptr, context), "incumbent projection");
    device_output.download(expected);
    compare(expected, expected, std::size_t{rows} * outputs);

    constexpr std::array classes{tall::ChunkClass{32, 32}, tall::ChunkClass{64, 64},
                                 tall::ChunkClass{128, 128}, tall::ChunkClass{256, 256}};
    for (const auto budget : {32U, 64U, 128U, 256U}) {
        device_output.upload(initial);
        for (std::uint32_t first = 0; first < rows;) {
            const auto chunk = tall::resolve_chunk(rows - first, {32, 8, 1, budget}, classes);
            if (chunk.rows == 0) fail("no candidate chunk");
            check(cc::dense_big_split(device_input.data + std::size_t{first} * width, chunk.rows, width,
                                      weight.format, 3, &activation, nullptr, context), "split candidate planes");
            check(cc::dense_big_bf16_fixture(weight, activation,
                                            device_output.data + guard + std::size_t{first} * outputs,
                                            nullptr, context), "candidate projection");
            first += chunk.rows;
            device_output.download(actual);
            compare(actual, expected, std::size_t{first} * outputs);
        }
    }
}

}  // namespace

int main() {
    const std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)>
        context(cc::dense_big_ctx_create(), cc::dense_big_ctx_destroy);
    if (!context) fail("create incumbent scratch context");
    run(context.get(), 67, 80, 32);
    std::puts("PASS B4 CUDA three-plane short K and masked tiles");
    run(context.get(), 197, 144, 1056);
    std::puts("PASS B4 CUDA three-plane K wrap and chunk invariance");
}
