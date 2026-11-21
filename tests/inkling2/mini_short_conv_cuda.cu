#include "cuda/core/short_convolution.cuh"
#include "short_conv_device_fixture.hpp"

#include <cstdlib>
#include <iostream>

namespace {

struct Cuda {
    using Stream = cudaStream_t;
    static constexpr auto name = "Cuda";
    static constexpr auto invalid_value = cudaErrorInvalidValue;
    static constexpr auto launch = &strix::cuda_core::short_conv::launch;
    static constexpr auto launch_rounded = &strix::cuda_core::short_conv::launch_with_store<true>;

    static void require(cudaError_t status) {
        if (status == cudaSuccess) return;
        std::cerr << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
    static void* allocate(std::size_t bytes) {
        void* storage = nullptr;
        require(cudaMallocManaged(&storage, bytes));
        return storage;
    }
    static Stream create_stream() {
        Stream stream{};
        require(cudaStreamCreate(&stream));
        return stream;
    }
    static void sync(Stream stream) { require(cudaStreamSynchronize(stream)); }
    static void destroy_stream(Stream stream) { require(cudaStreamDestroy(stream)); }
    static void release(void* storage) { require(cudaFree(storage)); }
};

}  // namespace

int main() {
    return strix::tests::short_conv::verify<Cuda>();
}
