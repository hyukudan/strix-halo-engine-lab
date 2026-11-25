#pragma once

// Host-only CUDA runtime stand-in for the tests that exercise the native device runtime header
// (strix/device/runtime.hpp) and the CUDA weight uploader without a GPU. Memory comes from the C heap; each
// allocation entry point returns the status the test installs, so an out-of-memory refusal can be injected at the
// exact call a production allocation makes.

#include <cstddef>
#include <cstdlib>

enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal = 1 };
enum { cudaGraphExecUpdateSuccess = 0 };
constexpr unsigned cudaStreamNonBlocking = 1U;

using cudaStream_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
struct cudaGraphExecUpdateResultInfo {
    int result;
};

namespace cuda_status_stub {
inline cudaError_t malloc_status = cudaSuccess;       // cudaMalloc (device memory)
inline cudaError_t malloc_host_status = cudaSuccess;  // cudaMallocHost (pinned host memory)
inline unsigned malloc_calls = 0, malloc_host_calls = 0;
}  // namespace cuda_status_stub

inline cudaError_t cudaMalloc(void** pointer, std::size_t bytes) {
    ++cuda_status_stub::malloc_calls;
    if (cuda_status_stub::malloc_status != cudaSuccess) return cuda_status_stub::malloc_status;
    *pointer = std::malloc(bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void** pointer, std::size_t bytes) {
    ++cuda_status_stub::malloc_host_calls;
    if (cuda_status_stub::malloc_host_status != cudaSuccess) return cuda_status_stub::malloc_host_status;
    *pointer = std::malloc(bytes);
    return cudaSuccess;
}
inline cudaError_t cudaFree(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}
inline cudaError_t cudaFreeHost(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* device) {
    *device = 0;
    return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t error) {
    return error == cudaErrorMemoryAllocation ? "out of memory" : error == cudaSuccess ? "no error" : "invalid value";
}
inline cudaError_t cudaMemGetInfo(std::size_t* free, std::size_t* total) {
    *free = *total = 0;
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void*, int, std::size_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
    *stream = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { return cudaSuccess; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* graph) {
    *graph = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t, unsigned long long) {
    *exec = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaGraphExecUpdate(cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo* info) {
    info->result = cudaGraphExecUpdateSuccess;
    return cudaSuccess;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
