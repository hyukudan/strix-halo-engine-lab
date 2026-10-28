#pragma once
#include <cstddef>
using hipStream_t = void*;
using hipEvent_t = void*;
enum hipError_t { hipSuccess = 0, hipErrorOutOfMemory = 2, hipErrorNotReady = 600, hipErrorUnknown = 999 };
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice };
inline constexpr unsigned hipStreamNonBlocking = 1, hipHostMallocDefault = 0, hipEventDisableTiming = 2;
extern "C" {
hipError_t hipStreamCreateWithFlags(hipStream_t*, unsigned);
hipError_t hipStreamDestroy(hipStream_t);
hipError_t hipMalloc(void**, std::size_t);
hipError_t hipFree(void*);
hipError_t hipEventCreateWithFlags(hipEvent_t*, unsigned);
hipError_t hipEventDestroy(hipEvent_t);
hipError_t hipEventRecord(hipEvent_t, hipStream_t);
hipError_t hipEventQuery(hipEvent_t);
hipError_t hipEventSynchronize(hipEvent_t);
hipError_t hipStreamWaitEvent(hipStream_t,hipEvent_t,unsigned);
hipError_t hipDeviceSynchronize();
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipMemcpyAsync(void*, const void*, std::size_t, hipMemcpyKind, hipStream_t);
hipError_t hipMemcpy(void*, const void*, std::size_t, hipMemcpyKind);
hipError_t hipHostMalloc(void**, std::size_t, unsigned);
hipError_t hipHostFree(void*);
const char* hipGetErrorString(hipError_t);
}
