#pragma once
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
using hipError_t = int;
inline constexpr hipError_t hipSuccess = 0;
namespace prefill_mock {
inline thread_local int device = 0;
inline std::atomic<unsigned> allocations{0}, frees{0};
}
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipGetDevice(int* device) { *device = prefill_mock::device; return hipSuccess; }
inline const char* hipGetErrorString(hipError_t) { return "mock allocation error"; }
inline hipError_t hipMalloc(void** ptr, std::size_t bytes) {
    *ptr = std::malloc(bytes);
    if (!*ptr) return 1;
    ++prefill_mock::allocations;
    return hipSuccess;
}
inline hipError_t hipFree(void* ptr) { ++prefill_mock::frees; std::free(ptr); return hipSuccess; }
inline hipError_t hipMemset(void* ptr, int value, std::size_t bytes) {
    std::memset(ptr, value, bytes); return hipSuccess;
}
