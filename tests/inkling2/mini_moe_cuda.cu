#include "inkling2/inkling2_mini_moe_cuda.hpp"
#include "mini_moe_device_fixture.hpp"

#include <cstdlib>
#include <iostream>

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

}  // namespace

int main() {
    return strix::tests::mini_moe::verify<Cuda>();
}
