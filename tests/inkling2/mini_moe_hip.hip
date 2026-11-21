#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "inkling2/inkling2_mini_moe_hip.hpp"
#include "mini_moe_device_fixture.hpp"

#include <cstdlib>
#include <iostream>

namespace {

struct Hip {
    using MoeBackend = strix::inkling2::MiniHipMoeBackend;
    using Stream = hipStream_t;
    static constexpr auto enqueue_resident = strix::inkling2::enqueue_mini_hip_resident_moe;

    static void require(hipError_t status) {
        if (status == hipSuccess) return;
        std::cerr << hipGetErrorString(status) << '\n';
        std::exit(1);
    }
    static void* allocate(std::size_t bytes) {
        void* memory = nullptr;
        require(hipMallocManaged(&memory, bytes));
        return memory;
    }
    static Stream create_stream() {
        Stream stream{};
        require(hipStreamCreate(&stream));
        return stream;
    }
    static void sync(Stream stream) { require(hipStreamSynchronize(stream)); }
    static void destroy_stream(Stream stream) { require(hipStreamDestroy(stream)); }
    static void release(void* memory) { require(hipFree(memory)); }
};

}  // namespace

int main() {
    return strix::tests::mini_moe::verify<Hip>();
}
