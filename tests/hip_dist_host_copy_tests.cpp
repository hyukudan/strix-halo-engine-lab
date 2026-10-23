#include "strix/dist/host_copy.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
}
int main() try {
    constexpr std::size_t capacity = (std::size_t{1} << 20U) + 128U;
    void *device = nullptr, *host = nullptr, *alias = nullptr;
    check(hipMalloc(&device, capacity));
    check(hipHostMalloc(&host, capacity, hipHostMallocDefault));
    check(hipHostGetDevicePointer(&alias, host, 0));
    hipStream_t stream{};
    hipEvent_t done{};
    check(hipStreamCreate(&stream));
    check(hipEventCreateWithFlags(&done, hipEventDisableTiming));
    std::vector<unsigned char> input(capacity);
    for (std::size_t i = 0; i < capacity; ++i) input[i] = static_cast<unsigned char>((i * 2654435761ULL) >> 13U);
    check(hipMemcpy(device, input.data(), capacity, hipMemcpyHostToDevice));
    unsigned cases = 0;
    for (const std::size_t bytes : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{4}, std::size_t{15}, std::size_t{16},
                                   std::size_t{17}, std::size_t{63}, std::size_t{76}, std::size_t{20480}, std::size_t{65536},
                                   std::size_t{163840}, std::size_t{1048576}}) {
        for (const std::size_t src_offset : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{16}}) {
            for (const std::size_t dst_offset : {std::size_t{16}, std::size_t{17}, std::size_t{23}, std::size_t{32}}) {
                auto* data = static_cast<unsigned char*>(host);
                std::fill(data, data + capacity, static_cast<unsigned char>(0xA5));
                strix::dist::enqueue_host_d2h_copy(static_cast<unsigned char*>(alias) + dst_offset,
                                                 static_cast<unsigned char*>(device) + src_offset, bytes, stream);
                check(hipEventRecord(done, stream));
                check(hipEventSynchronize(done));
                if (!std::equal(data + dst_offset, data + dst_offset + bytes, input.data() + src_offset) ||
                    !std::all_of(data, data + dst_offset, [](auto v) { return v == 0xA5; }) ||
                    !std::all_of(data + dst_offset + bytes, data + capacity, [](auto v) { return v == 0xA5; }))
                    throw std::runtime_error("copy bytes or guard corrupted");
                ++cases;
            }
        }
    }
    check(hipEventDestroy(done));
    check(hipStreamDestroy(stream));
    check(hipHostFree(host));
    check(hipFree(device));
    std::cout << "host D2H kernel byte-exact " << cases << " aligned/unaligned/tail cases; guards intact\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
