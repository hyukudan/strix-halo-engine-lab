#pragma once
// Synchronous CPU stand-in for the HIP runtime calls of src/dist_communicator.cpp (host contract tests only): device
// memory is host memory, every copy completes at once, events are complete when recorded. No device or DMA evidence.
#include <cstddef>
#include <cstdlib>
#include <cstring>

enum hipError_t { hipSuccess = 0, hipErrorOutOfMemory = 2, hipErrorNotReady = 600 };
struct StubStream {};
struct StubEvent { bool recorded = false; };
using hipStream_t = StubStream*;
using hipEvent_t = StubEvent*;
enum hipMemcpyKind { hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice, hipMemcpyDefault };
enum hipDeviceAttribute_t { hipDeviceAttributeHostNativeAtomicSupported = 1, hipDeviceAttributeIntegrated = 2 };
inline constexpr unsigned hipHostMallocDefault = 0U, hipHostMallocPortable = 1U, hipHostMallocMapped = 2U,
                          hipHostMallocCoherent = 4U, hipHostMallocNonCoherent = 8U;
inline constexpr unsigned hipHostRegisterDefault = 0U, hipHostRegisterMapped = 2U, hipExtHostRegisterCoarseGrained = 8U;
inline constexpr unsigned hipEventDefault = 0U, hipEventDisableTiming = 2U, hipEventReleaseToSystem = 4U;
inline constexpr unsigned hipStreamDefault = 0U, hipStreamNonBlocking = 1U;

inline const char* hipGetErrorString(hipError_t) { return "hip stub error"; }
inline const char* hipGetErrorName(hipError_t) { return "hip stub error"; }
inline hipError_t hipGetLastError() { return hipSuccess; }
inline hipError_t hipMalloc(void** p, std::size_t n) {
    *p = std::calloc(1U, n == 0U ? 1U : n);
    return *p ? hipSuccess : hipErrorOutOfMemory;
}
inline hipError_t hipHostMalloc(void** p, std::size_t n, unsigned = 0U) { return hipMalloc(p, n); }
inline hipError_t hipFree(void* p) { std::free(p); return hipSuccess; }
inline hipError_t hipHostFree(void* p) { std::free(p); return hipSuccess; }
inline hipError_t hipHostRegister(void*, std::size_t, unsigned) { return hipSuccess; }
inline hipError_t hipHostUnregister(void*) { return hipSuccess; }
inline hipError_t hipHostGetDevicePointer(void** p, void* host, unsigned) { *p = host; return hipSuccess; }
inline hipError_t hipGetDevice(int* d) { *d = 0; return hipSuccess; }
inline hipError_t hipSetDevice(int) { return hipSuccess; }
inline hipError_t hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attribute, int) {
    *value = attribute == hipDeviceAttributeHostNativeAtomicSupported ? 1 : 0;
    return hipSuccess;
}
inline hipError_t hipMemset(void* p, int v, std::size_t n) { std::memset(p, v, n); return hipSuccess; }
inline hipError_t hipMemsetAsync(void* p, int v, std::size_t n, hipStream_t) { return hipMemset(p, v, n); }
inline hipError_t hipMemcpy(void* d, const void* s, std::size_t n, hipMemcpyKind) { std::memcpy(d, s, n); return hipSuccess; }
inline hipError_t hipMemcpyAsync(void* d, const void* s, std::size_t n, hipMemcpyKind, hipStream_t) {
    std::memcpy(d, s, n);
    return hipSuccess;
}
inline hipError_t hipStreamCreateWithFlags(hipStream_t* s, unsigned) { *s = new StubStream; return hipSuccess; }
inline hipError_t hipStreamCreate(hipStream_t* s) { return hipStreamCreateWithFlags(s, 0U); }
inline hipError_t hipStreamDestroy(hipStream_t s) { delete s; return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned) { return hipSuccess; }
inline hipError_t hipEventCreateWithFlags(hipEvent_t* e, unsigned) { *e = new StubEvent; return hipSuccess; }
inline hipError_t hipEventCreate(hipEvent_t* e) { return hipEventCreateWithFlags(e, 0U); }
inline thread_local unsigned stub_event_records = 0;
inline hipError_t hipEventRecord(hipEvent_t e, hipStream_t = nullptr) {
    ++stub_event_records;
    e->recorded = true;
    return hipSuccess;
}
inline hipError_t hipEventQuery(hipEvent_t) { return hipSuccess; }
inline hipError_t hipEventSynchronize(hipEvent_t) { return hipSuccess; }
inline hipError_t hipEventDestroy(hipEvent_t e) { delete e; return hipSuccess; }
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t, hipEvent_t) { *ms = 0.0F; return hipSuccess; }
