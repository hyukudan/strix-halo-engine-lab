#include "core/hip_short_convolution.hpp"
#include "short_conv_device_fixture.hpp"

#include <cstdlib>
#include <iostream>

namespace {

struct Hip {
    using Stream = hipStream_t;
    static constexpr auto name = "Hip";
    static constexpr auto invalid_value = hipErrorInvalidValue;
    static constexpr auto launch = &strix::core::hip_short_conv::launch;
    static constexpr auto launch_rounded = &strix::core::hip_short_conv::launch_with_store<true>;

    static void require(hipError_t status) {
        if (status == hipSuccess) return;
        std::cerr << hipGetErrorString(status) << '\n';
        std::exit(1);
    }
    static void* allocate(std::size_t bytes) {
        void* storage = nullptr;
        require(hipMallocManaged(&storage, bytes));
        return storage;
    }
    static Stream create_stream() {
        Stream stream{};
        require(hipStreamCreate(&stream));
        return stream;
    }
    static void sync(Stream stream) { require(hipStreamSynchronize(stream)); }
    static void destroy_stream(Stream stream) { require(hipStreamDestroy(stream)); }
    static void release(void* storage) { require(hipFree(storage)); }
};

}  // namespace

int main() {
    return strix::tests::short_conv::verify<Hip>();
}
