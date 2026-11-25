#pragma once

#include <cstddef>
#include <cstdlib>

enum hipError_t { hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2, hipErrorInvalidDevice = 101 };

enum hipMemcpyKind { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3 };
enum hipStreamCaptureMode { hipStreamCaptureModeThreadLocal = 1 };
constexpr unsigned hipStreamNonBlocking = 1U;
using hipStream_t = void*;
using hipGraph_t = void*;
using hipGraphExec_t = void*;

struct hipDeviceProp_t {
    char gcnArchName[32];
    int warpSize;
};

namespace hip_archguard_test {
inline int active_device = 0;
inline unsigned property_reads = 0;
inline hipError_t malloc_status = hipSuccess;  // hipMalloc: the status the device-allocation boundary sees
inline constexpr hipDeviceProp_t devices[] = {
    {"gfx1151:xnack-", 32}, {"gfx1200", 32}, {"gfx1201", 32},
    {"gfx942:sramecc+", 64}, {"gfx950", 64}, {"gfx11510", 32},
    {"gfx1151", 64},
};
}

inline hipError_t hipGetDevice(int* device) {
    *device = hip_archguard_test::active_device;
    return *device < 0 ? hipErrorInvalidDevice : hipSuccess;
}

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* properties, int device) {
    ++hip_archguard_test::property_reads;
    *properties = hip_archguard_test::devices[device];
    return hipSuccess;
}

inline const char* hipGetErrorString(hipError_t error) {
    if (error == hipSuccess) return "no error";
    if (error == hipErrorOutOfMemory) return "out of memory";
    return "invalid device";
}

// The device runtime calls of strix/device/runtime.hpp (native section), so the host test can reach the actual
// device-allocation boundary. hipMalloc returns the injected status; memory comes from the C heap.
inline hipError_t hipMalloc(void** pointer, std::size_t bytes) {
    if (hip_archguard_test::malloc_status != hipSuccess) return hip_archguard_test::malloc_status;
    *pointer = std::malloc(bytes);
    return hipSuccess;
}
inline hipError_t hipFree(void* pointer) {
    std::free(pointer);
    return hipSuccess;
}
inline hipError_t hipHostMalloc(void** pointer, std::size_t bytes) {
    *pointer = std::malloc(bytes);
    return hipSuccess;
}
inline hipError_t hipHostFree(void* pointer) {
    std::free(pointer);
    return hipSuccess;
}
inline hipError_t hipSetDevice(int) { return hipSuccess; }
inline hipError_t hipGetLastError() { return hipSuccess; }
inline hipError_t hipMemGetInfo(std::size_t* free, std::size_t* total) {
    *free = *total = 0;
    return hipSuccess;
}
inline hipError_t hipMemcpyAsync(void*, const void*, std::size_t, hipMemcpyKind, hipStream_t) { return hipSuccess; }
inline hipError_t hipMemsetAsync(void*, int, std::size_t, hipStream_t) { return hipSuccess; }
inline hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned) {
    *stream = nullptr;
    return hipSuccess;
}
inline hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
inline hipError_t hipStreamBeginCapture(hipStream_t, hipStreamCaptureMode) { return hipSuccess; }
inline hipError_t hipStreamEndCapture(hipStream_t, hipGraph_t* graph) {
    *graph = nullptr;
    return hipSuccess;
}
inline hipError_t hipGraphInstantiate(hipGraphExec_t* exec, hipGraph_t, void*, char*, std::size_t) {
    *exec = nullptr;
    return hipSuccess;
}
inline hipError_t hipGraphLaunch(hipGraphExec_t, hipStream_t) { return hipSuccess; }
inline hipError_t hipGraphDestroy(hipGraph_t) { return hipSuccess; }
inline hipError_t hipGraphExecDestroy(hipGraphExec_t) { return hipSuccess; }
