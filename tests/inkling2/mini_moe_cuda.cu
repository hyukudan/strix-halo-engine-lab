#include "inkling2/inkling2_mini_moe_cuda.hpp"
#include "mini_moe_device_fixture.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

namespace {

struct Cuda {
    using MoeBackend = strix::inkling2::MiniCudaMoeBackend;
    using Stream = cudaStream_t;
    static constexpr auto enqueue_resident = strix::inkling2::enqueue_mini_cuda_resident_moe;

    static void require(cudaError_t status) {
        if (status == cudaSuccess) return;
        std::cerr << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
    static void* allocate(std::size_t bytes) {
        void* memory = nullptr;
        require(cudaMallocManaged(&memory, bytes));
        return memory;
    }
    static Stream create_stream() {
        Stream stream{};
        require(cudaStreamCreate(&stream));
        return stream;
    }
    static void sync(Stream stream) { require(cudaStreamSynchronize(stream)); }
    static void destroy_stream(Stream stream) { require(cudaStreamDestroy(stream)); }
    static void release(void* memory) { require(cudaFree(memory)); }
};

template <class T>
auto allocate(std::size_t elements) {
    return std::unique_ptr<T, decltype(&Cuda::release)>(
        static_cast<T*>(Cuda::allocate(elements * sizeof(T))), Cuda::release);
}

bool check_router_bits(unsigned rows) {
    namespace mini = strix::inkling2;
    namespace cc = strix::cuda_core;
    constexpr unsigned width = mini::kHiddenWidth;
    constexpr unsigned experts = mini::kExperts;
    const auto input_elements = std::size_t{rows} * width;
    const auto output_elements = std::size_t{rows} * experts;
    auto input = allocate<float>(input_elements);
    auto expected = allocate<float>(output_elements);
    auto actual = allocate<float>(output_elements);
    auto weight = allocate<std::uint16_t>(std::size_t{width} * experts);
    for (std::size_t i = 0; i < input_elements; ++i)
        input.get()[i] = static_cast<float>(static_cast<int>((i * 31U + i / width) % 973U) - 500) / 997.0F;
    for (std::size_t i = 0; i < std::size_t{width} * experts; ++i)
        weight.get()[i] = static_cast<std::uint16_t>(0x3B00U + (i * 17U + i / width) % 1021U);

    const cc::DenseWeight router{weight.get(), cc::WeightFormat::bf16, experts, width};
    for (unsigned first = 0; first < rows;) {
        const unsigned count = std::min(cc::kMaxRows, rows - first);
        Cuda::require(cc::dense_rows(router, input.get() + std::size_t{first} * width,
                                     expected.get() + std::size_t{first} * experts, count, nullptr, nullptr,
                                     cc::DenseArith::fp32));
        first += count;
    }
    Cuda::require(mini::enqueue_mini_cuda_router_logits(router, rows, {input.get(), input_elements},
                                                        {actual.get(), output_elements}, nullptr));
    Cuda::sync(nullptr);
    if (std::memcmp(expected.get(), actual.get(), output_elements * sizeof(float)) == 0) return true;
    std::cerr << "CUDA Mini FP32 router projection bits differ at " << rows << " rows\n";
    return false;
}

}  // namespace

int main() {
    if (!check_router_bits(33U) || !check_router_bits(2048U)) return 1;
    return strix::tests::mini_moe::verify<Cuda>();
}
